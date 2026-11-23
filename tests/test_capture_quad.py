import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_quad.py"
sys.path.insert(0, str(SCRIPT.parent))
import capture_quad


class QuadCaptureTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.output = self.directory / "quad.jpg"
        self.stdout = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.stack.enter_context(patch.object(capture_quad.shutil, "which", return_value="rpicam-still"))

    def test_help_runs_without_site_packages_app_configuration_or_a_pi(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
                                cwd=self.directory, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("four-in-one", result.stdout)

    def test_zero_exit_status_with_no_sensors_does_not_attempt_a_capture(self):
        with patch.object(capture_quad.subprocess, "run", return_value=Mock(
                returncode=0, stdout="No cameras available!\n")) as run:
            self.assertEqual(capture_quad.main(["-o", str(self.output)]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.output.exists())
        self.assertIn("dtoverlay=imx519", self.stderr.getvalue())

    def test_single_capture_saves_a_composite_using_either_command_family(self):
        for executable in ("rpicam-still", "libcamera-still"):
            with self.subTest(executable=executable):
                def run(command, **kwargs):
                    if "--list-cameras" in command:
                        return Mock(returncode=0, stdout="Available cameras\n0 : imx519 [4656x3496]\n")
                    # Minimal JPEG markers exercise the file-completion check.
                    Path(command[command.index("--output") + 1]).write_bytes(b"\xff\xd8JPEG\xff\xd9")
                    return Mock(returncode=0)

                with patch.object(capture_quad.shutil, "which", side_effect=lambda name: (
                        executable if name == executable else None)), \
                        patch.object(capture_quad.subprocess, "run", side_effect=run) as capture:
                    self.assertEqual(capture_quad.main(["-o", str(self.output)]), 0)
                self.assertEqual(capture.call_count, 2)  # one enumeration, one capture
                command = capture.call_args.args[0]
                self.assertEqual(command[0], executable)
                self.assertEqual(command[command.index("--mode") + 1], "4656:3496:10")
                self.assertEqual(command[command.index("--roi") + 1], "0,0,1,1")
                self.assertEqual(command[command.index("--rotation") + 1], "0")
                self.assertIn("Saved:", self.stdout.getvalue())

    def test_truncated_jpeg_does_not_report_success(self):
        self.output.write_bytes(b"\xff\xd8unfinished JPEG")
        with patch.object(capture_quad.subprocess, "run", return_value=Mock(
                returncode=0, stdout="0 : imx519 [4656x3496]\n")):
            self.assertEqual(capture_quad.main(["-o", str(self.output)]), 1)
        self.assertIn("complete JPEG", self.stderr.getvalue())
        self.assertNotIn("Saved:", self.stdout.getvalue())

    def test_capture_failure_is_reported_even_if_an_old_output_file_exists(self):
        self.output.write_bytes(b"\xff\xd8old image\xff\xd9")
        with patch.object(capture_quad.subprocess, "run", side_effect=[
                Mock(returncode=0, stdout="0 : imx519 [4656x3496]\n"),
                subprocess.CalledProcessError(1, ["rpicam-still"]),
        ]):
            self.assertEqual(capture_quad.main(["-o", str(self.output)]), 1)
        self.assertNotIn("Saved:", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
