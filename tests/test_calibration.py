import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image

import config as cfg
import camera_settings as settings
from calibrate import (
    color_correct, combine_calibrations, distortion_calibrate,
    rotation_calibrate, wb_calibrate,
)
from framing import camera_views
from pipeline import apply_white_balance, run_pipeline, split_grid


class CalibrationTests(unittest.TestCase):
    def test_profile_writers_share_the_pipeline_paths(self):
        self.assertEqual(wb_calibrate.PROFILE_PATH, cfg.WB_PROFILE_PATH)
        self.assertEqual(color_correct.PROFILE_PATH, cfg.COLOR_PROFILE_PATH)
        self.assertEqual(rotation_calibrate.CALIB_OUT, cfg.ROTATION_CALIBRATION_PATH)

    def test_all_calibration_capture_paths_use_the_imx519_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            def capture(command, **_kwargs):
                out = Path(command[command.index("-o") + 1])
                Image.new("RGB", (128, 96), "white").save(out)

            ui = SimpleNamespace(root=Mock(), _set_status=Mock())
            with patch.object(cfg, "INPUT_DIR", base / "input"), \
                    patch.object(settings.shutil, "which", return_value="installed"), \
                    patch("subprocess.run", side_effect=capture) as run, \
                    patch("builtins.input", return_value=""):
                wb_calibrate.capture_image()
                color_correct.capture_image()
                rotation_calibrate.CaptureUI._capture(ui)
                distortion_calibrate.capture_and_prepare_sets(str(base / "distortion"), capture_count=1)
            self.assertEqual(run.call_count, 4)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--width") + 1], str(settings.CAPTURE_WIDTH))
                self.assertEqual(command[command.index("--height") + 1], str(settings.CAPTURE_HEIGHT))
                self.assertEqual(command[command.index("--camera") + 1], str(settings.CAMERA_INDEX))
                self.assertEqual(command[command.index("--autofocus-mode") + 1], settings.FOCUS_MODE)
                self.assertEqual(command[command.index("--mode") + 1], "4656:3496:10")
                self.assertEqual(command[command.index("--roi") + 1], "0,0,1,1")
                self.assertNotIn("imx219_noir.json", " ".join(command))

    def test_new_white_balance_profile_is_consumed_by_the_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "white-paper.jpg"
            Image.new("RGB", (128, 96), (180, 130, 100)).save(raw)
            profile = base / "new modules" / "white_balance_profile.npz"
            with patch.object(wb_calibrate, "PROFILE_DIR", profile.parent), \
                    patch.object(wb_calibrate, "PROFILE_PATH", profile), \
                    patch.object(cfg, "WB_PROFILE_PATH", profile):
                wb_calibrate.calibrate(raw)
                pieces = wb_calibrate.split_and_rotate(Image.open(raw))
                corrected = apply_white_balance(pieces)
            self.assertTrue(profile.exists())
            self.assertEqual(len(corrected), 4)
            for image in corrected:
                means = np.asarray(image).mean(axis=(0, 1))
                self.assertLess(np.ptp(means), 2)

    def test_white_balance_calibration_does_not_apply_exif_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = base / "quad.jpg"
            image = Image.new("RGB", (256, 192), "white")
            exif = image.getexif()
            exif[274] = 6  # Metadata requests a 90-degree turn; preserve sensor orientation.
            image.save(raw, exif=exif)
            profile = base / "white_balance_profile.npz"
            with patch.object(wb_calibrate, "PROFILE_DIR", base), \
                    patch.object(wb_calibrate, "PROFILE_PATH", profile):
                wb_calibrate.calibrate(raw)
            with np.load(profile) as saved:
                self.assertEqual(saved["gain_maps"].shape, (4, 12, 16, 3))

    def test_rotation_save_creates_the_new_profile_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "new modules" / "rotation_calibration.npz"
            ui = SimpleNamespace(root=Mock(), angles=[0.1, -0.2, 0.3, -0.4])
            with patch.object(rotation_calibrate, "CALIB_OUT", profile):
                rotation_calibrate.SummaryUI._save(ui)
            with np.load(profile) as data:
                np.testing.assert_allclose(data["angles"], ui.angles)

    def test_full_resolution_composite_splits_into_four_native_quad_views(self):
        raw = Image.new("RGB", (4656, 3496))
        pieces = split_grid(raw, 2, 2)
        self.assertEqual([piece.size for piece in pieces], [(2328, 1748)] * 4)
        self.assertEqual([view.size for view in camera_views(raw)], [(2328, 1748)] * 4)

    def test_all_calibration_views_retain_native_sensor_orientation(self):
        pixels = np.random.default_rng(12).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        raw = Image.fromarray(np.tile(pixels, (2, 2, 1)))
        expected = pixels
        for load_views in (wb_calibrate.split_and_rotate, color_correct.split_and_rotate):
            for view in load_views(raw):
                np.testing.assert_array_equal(np.asarray(view), expected)
        for camera in range(4):
            np.testing.assert_array_equal(distortion_calibrate.rotate_piece_for_camera(pixels, camera), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quad.png"
            raw.save(path)
            for view in rotation_calibrate.load_camera_images(path):
                np.testing.assert_array_equal(np.asarray(view), expected)

    def test_saved_rotation_calibrations_do_not_change_the_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            profile = base / "rotation.npz"
            np.savez(profile, angles=[90.0, 180.0, 270.0, 15.0])
            pixels = np.random.default_rng(16).integers(0, 256, (96, 128, 3), dtype=np.uint8)
            raw = Image.fromarray(np.tile(pixels, (2, 2, 1)))
            raw_path = base / "quad.png"
            raw.save(raw_path)
            with patch.object(cfg, "WB_PROFILE_PATH", base / "absent-wb.npz"):
                with patch.object(cfg, "ROTATION_CALIBRATION_PATH", base / "absent-rotation.npz"):
                    run_pipeline(str(raw_path), str(base), "without_rotation_profile")
                with patch.object(cfg, "ROTATION_CALIBRATION_PATH", profile):
                    run_pipeline(str(raw_path), str(base), "with_rotation_profile")
            self.assertEqual((base / "without_rotation_profile.gif").read_bytes(),
                             (base / "with_rotation_profile.gif").read_bytes())

    def test_combining_profiles_does_not_reimport_the_previous_combined_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.savez(base / "rotation_calibration.npz", angles=np.zeros(4))
            with patch.object(combine_calibrations, "INDIVIDUAL_DIR", base):
                self.assertTrue(combine_calibrations.combine_all_calibrations(base))
                self.assertTrue(combine_calibrations.combine_all_calibrations(base))
            with np.load(base / "combined_calibration.npz") as data:
                self.assertEqual(data.files, ["rotation_calibration_angles"])

    def test_composite_still_runs_through_the_gif_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rng = np.random.default_rng(42)
            view = Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
            raw = Image.new("RGB", (256, 192))
            for x, y in ((0, 0), (128, 0), (0, 96), (128, 96)):
                raw.paste(view, (x, y))
            image_path = base / "quad.jpg"
            raw.save(image_path, quality=100)
            with patch.object(cfg, "WB_PROFILE_PATH", base / "absent-wb.npz"), \
                    patch.object(cfg, "ROTATION_CALIBRATION_PATH", base / "absent-rotation.npz"):
                run_pipeline(str(image_path), str(base / "gifs"), "capture")
            with Image.open(base / "gifs" / "capture.gif") as gif:
                self.assertEqual(gif.format, "GIF")
                self.assertEqual(gif.size, (176, 99))
                self.assertEqual(gif.width * 9, gif.height * 16)
                gif.load()


if __name__ == "__main__":
    unittest.main()
