from pathlib import Path
import logging
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overlays import ViewfinderOverlay


class OverlayTests(unittest.TestCase):
    def render(self, source, viewport=(800, 480)):
        return ViewfinderOverlay().render(
            source, viewport, shutter_us=100002, fps=15,
            storage="12.3GB", clock="12:34:56",
        )

    def test_overlay_sizes_do_not_depend_on_camera_stream_resolution(self):
        low = Image.new("RGB", (437, 582), (64, 80, 96))
        high = Image.new("RGB", (1748, 2328), (64, 80, 96))
        rendered_low, rendered_high = self.render(low), self.render(high)
        self.assertEqual(rendered_low.size, (360, 480))
        np.testing.assert_array_equal(np.asarray(rendered_low), np.asarray(rendered_high))
        self.assertEqual(low.getpixel((218, 291)), (64, 80, 96))

    def test_full_view_and_all_four_corners_remain_visible(self):
        source = Image.new("RGB", (240, 320))
        colors = ("red", "green", "blue", "yellow")
        for index, color in enumerate(colors):
            source.paste(Image.new("RGB", (120, 160), color),
                         ((index % 2) * 120, (index // 2) * 160))
        result = self.render(source)
        for point, color in zip(((2, 2), (357, 2), (2, 477), (357, 477)), colors):
            self.assertEqual(result.getpixel(point), Image.new("RGB", (1, 1), color).getpixel((0, 0)))

    def test_header_stays_compact_on_small_and_large_displays(self):
        source = Image.new("RGB", (437, 582), (64, 80, 96))
        for viewport in ((320, 240), (800, 480), (1920, 1080)):
            with self.subTest(viewport=viewport):
                rendered = self.render(source, viewport)
                pixels = np.asarray(rendered).astype(int)
                yellow = (pixels[:, :, 0] > pixels[:, :, 1] + 10) & (pixels[:, :, 1] > pixels[:, :, 2] + 10)
                rows, _ = np.where(yellow)
                self.assertGreater(len(rows), 0)
                self.assertLessEqual(int(rows.max() - rows.min()), 18)
                self.assertLessEqual(rendered.width, viewport[0])
                self.assertLessEqual(rendered.height, viewport[1])

    def test_app_does_not_enlarge_the_image_after_drawing_the_ui(self):
        # Exercise the app's real display path without a Tk window or GPIO access.
        with patch.dict(sys.modules, {"RPi": None}), \
                patch("logging.FileHandler", return_value=logging.NullHandler()):
            import app as app_module
        app = app_module.WiggleApp.__new__(app_module.WiggleApp)
        app.screen_w, app.screen_h = 800, 480
        app._overlay = ViewfinderOverlay()
        app._disk_usage_counter, app._disk_usage_cache = 0, "12.3GB"
        app.canvas, app._canvas_img_id = Mock(), 1
        source = Image.new("RGB", (437, 582), (64, 80, 96))
        with patch.object(app_module, "datetime") as clock, \
                patch.object(app_module.ImageTk, "PhotoImage", side_effect=lambda image: image):
            clock.now.return_value.strftime.return_value = "12:34:56"
            app._show_pil(source, overlays=True)
        np.testing.assert_array_equal(np.asarray(app._current_tk_img), np.asarray(self.render(source)))


if __name__ == "__main__":
    unittest.main()
