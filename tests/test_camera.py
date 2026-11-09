import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import camera
import camera_settings as settings
from viewfinder import MJPEGReader


class CameraSettingsTests(unittest.TestCase):
    def configured_commands(self, **overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WIGGLEGRAM_")}
        env.update(PYTHONPATH=str(SRC), PYTHON_DOTENV_DISABLED="1", **overrides)
        return subprocess.run([
            sys.executable, "-c",
            "import json; from pathlib import Path; from unittest.mock import patch; "
            "import camera_settings as s; import config as c; "
            "\nwith patch('shutil.which', side_effect=lambda name: name): "
            "print(json.dumps({'still': s.still_command(Path('test photo.jpg')), "
            "'preview': s.viewfinder_command(), 'profiles': str(c.NPZ_DIR)}))",
        ], env=env, text=True, capture_output=True)

    def test_uc873_defaults_preserve_four_view_resolution(self):
        result = self.configured_commands()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        still, preview = data["still"], data["preview"]
        self.assertEqual(still[still.index("--width") + 1], "4656")
        self.assertEqual(still[still.index("--height") + 1], "3496")
        self.assertEqual(still[still.index("--mode") + 1], "4656:3496:10")
        self.assertEqual(still[still.index("--viewfinder-mode") + 1], "2328:1748:10")
        self.assertEqual(preview[preview.index("--mode") + 1], "2328:1748:10")
        for command in (still, preview):
            self.assertEqual(command[command.index("--roi") + 1], "0,0,1,1")
            self.assertEqual(command[command.index("--rotation") + 1], "0")
        self.assertEqual(still[still.index("--autofocus-mode") + 1], "auto")
        self.assertEqual(preview[preview.index("--autofocus-mode") + 1], "continuous")
        self.assertNotIn("--tuning-file", still)
        self.assertNotIn("--autofocus-on-capture", still)
        self.assertEqual(Path(data["profiles"]), SRC / "calibrate" / "out" / "imx519")

    def test_fixed_focus_and_camera_selection_reach_still_and_preview(self):
        result = self.configured_commands(
            WIGGLEGRAM_CAMERA_INDEX="1", WIGGLEGRAM_FOCUS_MODE="manual",
            WIGGLEGRAM_LENS_POSITION="0", WIGGLEGRAM_TUNING_FILE="custom tuning.json",
            WIGGLEGRAM_CALIBRATION_DIR="custom profiles",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        for command in (data["still"], data["preview"]):
            self.assertEqual(command[command.index("--camera") + 1], "1")
            self.assertEqual(command[command.index("--autofocus-mode") + 1], "manual")
            self.assertEqual(command[command.index("--lens-position") + 1], "0")
            self.assertEqual(command[command.index("--tuning-file") + 1], "custom tuning.json")
        self.assertEqual(data["profiles"], "custom profiles")

    def test_conflicting_lens_and_autofocus_settings_fail_early(self):
        result = self.configured_commands(WIGGLEGRAM_LENS_POSITION="1.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WIGGLEGRAM_FOCUS_MODE=manual", result.stderr)

    def test_old_16_by_9_stream_settings_cannot_silently_crop_the_composite(self):
        for name, width, height in (("CAPTURE", "3840", "2160"), ("VF", "1280", "720")):
            with self.subTest(stream=name):
                result = self.configured_commands(**{
                    f"WIGGLEGRAM_{name}_WIDTH": width, f"WIGGLEGRAM_{name}_HEIGHT": height,
                })
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("full sensor aspect ratio", result.stderr)

    def test_camera_command_selection_supports_both_os_families(self):
        for kind in ("still", "vid"):
            with self.subTest(kind=kind):
                with patch.object(settings.shutil, "which", return_value="installed"):
                    self.assertEqual(settings.camera_executable(kind), f"rpicam-{kind}")
                with patch.object(settings.shutil, "which", side_effect=lambda name: name.startswith("libcamera-")):
                    self.assertEqual(settings.camera_executable(kind), f"libcamera-{kind}")
                with patch.object(settings.shutil, "which", return_value=None):
                    with self.assertRaisesRegex(FileNotFoundError, "IMX519"):
                        settings.camera_executable(kind)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.proc = Mock(pid=1234, returncode=None)
        self.proc.poll.side_effect = lambda: self.proc.returncode
        self.stack.enter_context(patch.object(camera.cfg, "INPUT_DIR", self.temp / "input"))
        self.stack.enter_context(patch.object(camera.cfg, "OUTPUT_DIR", self.temp / "output"))
        self.stack.enter_context(patch.object(settings.shutil, "which", return_value="installed"))
        self.popen = self.stack.enter_context(patch.object(camera.subprocess, "Popen", return_value=self.proc))
        self.stack.enter_context(patch.object(camera.signal, "SIGUSR1", 10, create=True))
        self.signal = self.stack.enter_context(patch.object(camera.os, "kill"))
        self.stack.enter_context(patch.object(camera.time, "sleep"))
        self.stack.enter_context(patch.object(camera.threading, "Thread"))

    def write_capture(self, *, partial=False):
        command = self.popen.call_args.args[0]
        path = Path(command[command.index("-o") + 1])
        Image.new("RGB", (64, 48), "red").save(path)
        if partial:
            path.write_bytes(path.read_bytes()[:-2])

    def test_success_returns_completed_jpeg_and_releases_camera(self):
        self.signal.side_effect = lambda *_: self.write_capture()
        captured, output = camera.do_capture()
        self.assertIsNotNone(captured)
        with Image.open(captured) as image:
            image.load()
        self.assertEqual(output, self.temp / "output")
        self.proc.send_signal.assert_called_once_with(camera.signal.SIGINT)
        self.proc.wait.assert_called_once_with(timeout=3)
        self.assertIn("--signal", self.popen.call_args.args[0])

    def test_incomplete_jpeg_times_out_and_kills_a_stuck_camera(self):
        self.signal.side_effect = lambda *_: self.write_capture(partial=True)
        self.proc.wait.side_effect = [subprocess.TimeoutExpired("camera", 3), 0]
        with patch.object(camera.time, "monotonic", side_effect=[0, settings.CAPTURE_TIMEOUT_S + 1]):
            self.assertEqual(camera.do_capture(), (None, None))
        self.proc.kill.assert_called_once()

    def test_driver_startup_failure_does_not_send_capture_signal(self):
        self.proc.returncode = 1
        self.assertEqual(camera.do_capture(), (None, None))
        self.signal.assert_not_called()


class ViewfinderTests(unittest.TestCase):
    def test_mjpeg_frames_are_decoded_and_process_is_released(self):
        stream = io.BytesIO()
        for color in ("red", "blue"):
            Image.new("RGB", (24, 16), color).save(stream, format="JPEG")
        stream.seek(0)
        proc = Mock(stdout=stream, returncode=None)
        proc.poll.side_effect = lambda: proc.returncode
        proc.terminate.side_effect = lambda: setattr(proc, "returncode", 0)
        frames = queue.Queue(maxsize=2)
        reader = MJPEGReader(frames)
        with patch.object(settings.shutil, "which", return_value="installed"), \
                patch("viewfinder.subprocess.Popen", return_value=proc):
            reader.start()
            reader.join(timeout=5)
            reader.stop()
        self.assertFalse(reader.is_alive())
        self.assertTrue(stream.closed)
        image = frames.get_nowait()
        self.assertEqual(image.size, (24, 16))
        self.assertGreater(image.getpixel((0, 0))[2], 240)
        proc.terminate.assert_called_once()

    def test_stopped_reader_cannot_start_a_new_camera_process(self):
        reader = MJPEGReader(queue.Queue())
        reader.stop()
        with patch.object(settings.shutil, "which", return_value="installed"), \
                patch("viewfinder.subprocess.Popen") as popen:
            reader.start()
            reader.join(timeout=5)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
