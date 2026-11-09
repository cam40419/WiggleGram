from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framing import aspect_crop_size, camera_views, crop_to_aspect, pad_to_aspect, preview_view
from pipeline import align_full_views, wigglegram_anchors


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

    def test_16_by_9_padding_retains_every_native_pixel(self):
        pixels = np.random.default_rng(42).integers(0, 256, (128, 96, 3), dtype=np.uint8)
        framed = pad_to_aspect(Image.fromarray(pixels))
        self.assertEqual(framed.size, (240, 135))
        np.testing.assert_array_equal(np.asarray(framed)[3:131, 72:168], pixels)
        self.assertEqual(framed.getpixel((0, 0)), (0, 0, 0))

    def test_full_resolution_padding_does_not_discard_sensor_rows(self):
        view = Image.new("RGB", (2328, 1748), "white")
        framed = pad_to_aspect(view)
        self.assertEqual(framed.size, (3120, 1755))
        self.assertEqual(framed.crop((396, 3, 2724, 1751)).tobytes(), view.tobytes())

    def test_preview_preserves_every_pixel_in_native_sensor_orientation(self):
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
                self.assertEqual(view.size, (128, 96))
                self.assertEqual(view.getpixel((0, 0)), (0, 0, camera * 50))
                self.assertEqual(view.getpixel((127, 95)), (127, 95, camera * 50))
                row, column = divmod(camera, 2)
                expected = np.asarray(raw)[row * 96:(row + 1) * 96, column * 128:(column + 1) * 128]
                np.testing.assert_array_equal(np.asarray(view), expected)
                self.assertEqual(camera_views(raw)[camera].tobytes(), view.tobytes())

    def test_alignment_expands_to_preserve_all_edges_at_original_pixel_scale(self):
        pixels = np.random.default_rng(15).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        views = [Image.fromarray(pixels), Image.fromarray(pixels[:70, :120])]
        aligned = align_full_views(views, [(-20, 5), (10, -8)])
        self.assertEqual(aligned[0].size, (160, 103))
        np.testing.assert_array_equal(np.asarray(aligned[0])[13:103, :160], pixels)
        np.testing.assert_array_equal(np.asarray(aligned[1])[:70, 30:150], pixels[:70, :120])

    def test_gif_alignment_preserves_every_view_edge_in_16_by_9_output(self):
        gray = (np.arange(160 * 120).reshape(160, 120) % 256).astype(np.uint8)
        gray[0, 0] = gray[-1, -1] = 255
        pixels = np.repeat(gray[:, :, None], 3, axis=2)
        images = [Image.fromarray(pixels) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            # Shifts +7, 0, -5, -10 need 137 columns to retain every view.
            # This fits in 288x162 with 75 columns and 1 row of extra padding.
            wigglegram_anchors(images, anchors=[(53, 80), (60, 80), (65, 80), (70, 80)],
                               target_xy=(60, 80), save_dir=directory, filename="aligned")
            with Image.open(Path(directory) / "aligned.gif") as gif:
                self.assertEqual(gif.size, (288, 162))
                self.assertEqual(gif.n_frames, 6)
                for frame in range(gif.n_frames):
                    gif.seek(frame)
                    self.assertEqual(gif.width * 9, gif.height * 16)
                    left = (92, 85, 80, 75, 80, 85)[frame]
                    decoded = gif.convert("RGB")
                    # GIF palette quantization can slightly change white, but
                    # both source corner markers must remain in every frame.
                    self.assertGreater(min(decoded.getpixel((left, 1))), 245)
                    self.assertGreater(min(decoded.getpixel((left + 119, 160))), 245)


if __name__ == "__main__":
    unittest.main()
