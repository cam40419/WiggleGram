from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framing import aspect_crop_size, crop_to_aspect, preview_view
from pipeline import wigglegram_anchors


class FramingTests(unittest.TestCase):
    def test_largest_exact_16_by_9_crop_uses_all_available_detail(self):
        for source, expected in (((2328, 1748), (2320, 1305)),
                                 ((1748, 2328), (1744, 981)),
                                 ((4656, 3496), (4656, 2619)),
                                 ((3496, 4656), (3488, 1962)),
                                 ((3840, 2160), (3840, 2160))):
            with self.subTest(source=source):
                result = aspect_crop_size(source)
                self.assertEqual(result, expected)
                self.assertEqual(result[0] * 9, result[1] * 16)
                self.assertTrue(result[0] + 16 > source[0] or result[1] + 9 > source[1])

    def test_center_crop_preserves_source_pixels_without_resampling(self):
        pixels = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        cropped = crop_to_aspect(Image.fromarray(pixels))
        self.assertEqual(cropped.size, (128, 72))
        np.testing.assert_array_equal(np.asarray(cropped), pixels[12:84, :, :])

    def test_preview_selects_and_rotates_each_camera_before_cropping(self):
        raw = Image.new("RGB", (256, 192))
        for camera in range(4):
            pixels = np.zeros((96, 128, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(128)[None, :]
            pixels[:, :, 1] = np.arange(96)[:, None]
            pixels[:, :, 2] = camera * 50
            row, column = divmod(camera, 2)
            raw.paste(Image.fromarray(pixels), (column * 128, row * 96))
        for camera in range(4):
            with self.subTest(camera=camera):
                view = preview_view(raw, camera)
                self.assertEqual(view.size, (96, 54))
                self.assertEqual(view.getpixel((0, 0)), (90, 0, camera * 50))
                self.assertEqual(view.getpixel((95, 53)), (37, 95, camera * 50))

    def test_alignment_preserves_exact_16_by_9_and_uses_the_largest_common_crop(self):
        pixels = np.random.default_rng(42).integers(0, 256, (160, 120, 3), dtype=np.uint8)
        images = [Image.fromarray(pixels) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            # Shifts +7, 0, -5, -10 leave 103 native columns in common.
            # Their largest exact 16:9 rectangle is 96x54, after alignment.
            wigglegram_anchors(images, anchors=[(53, 80), (60, 80), (65, 80), (70, 80)],
                               target_xy=(60, 80), save_dir=directory, filename="aligned")
            with Image.open(Path(directory) / "aligned.gif") as gif:
                self.assertEqual(gif.size, (96, 54))
                self.assertEqual(gif.n_frames, 6)
                for frame in range(gif.n_frames):
                    gif.seek(frame)
                    self.assertEqual(gif.width * 9, gif.height * 16)


if __name__ == "__main__":
    unittest.main()
