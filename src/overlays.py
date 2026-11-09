"""Compact viewfinder graphics drawn at the display's pixel size."""

from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

def ui_scale(viewport: Tuple[int, int]) -> float:
    return max(0.6, min(1.25, viewport[0] / 800, viewport[1] / 480))


class ViewfinderOverlay:
    def __init__(self):
        self._fonts = {}

    def _font(self, size: int):
        if size not in self._fonts:
            for path in (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "DejaVuSans.ttf",
            ):
                try:
                    self._fonts[size] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            else:
                self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]

    def render(self, source: Image.Image, viewport: Tuple[int, int], *,
               shutter_us: int, fps: int, storage: str, clock: str) -> Image.Image:
        """Fit the complete view first, then draw UI at stable screen sizes."""
        factor = min(viewport[0] / source.width, viewport[1] / source.height)
        size = (max(1, int(source.width * factor)), max(1, int(source.height * factor)))
        image = source.resize(size, Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        scale = ui_scale(viewport)
        margin = max(5, round(10 * scale))
        title_font = self._font(max(9, round(14 * scale)))
        info_font = self._font(max(8, round(12 * scale)))
        line_height = max(12, round(17 * scale))

        def text_width(text, font):
            bounds = draw.textbbox((0, 0), text, font=font)
            return bounds[2] - bounds[0]

        def label(x, y, text, font=info_font, color="white"):
            draw.text((x + 1, y + 1), text, font=font, fill="black")
            draw.text((x, y), text, font=font, fill=color)

        cx, cy = width // 2, height // 2
        length, gap = max(6, round(11 * scale)), max(2, round(3 * scale))
        for start, end in (
            ((cx - length, cy), (cx - gap, cy)), ((cx + gap, cy), (cx + length, cy)),
            ((cx, cy - length), (cx, cy - gap)), ((cx, cy + gap), (cx, cy + length)),
        ):
            draw.line((start, end), fill="black", width=3)
            draw.line((start, end), fill="white", width=1)

        title = "WIGGLEGRAM"
        time_text = f"{clock}  {storage}"
        if text_width(title, title_font) + text_width(time_text, info_font) + margin * 3 > width:
            time_text = clock
        if text_width(title, title_font) + text_width(time_text, info_font) + margin * 3 > width:
            title = "WG"
        label(margin, margin, title, title_font, (255, 200, 0))
        label(width - margin - text_width(time_text, info_font), margin, time_text)

        exposure = shutter_us / 1_000_000
        shutter = f"{exposure:.1f}s" if exposure >= 1 else f"1/{round(1 / exposure)}"
        for index, text in enumerate((f"SS: {shutter}", "EV: +0.5", "AWB: Auto")):
            label(margin, height - margin - line_height * (3 - index), text)
        for index, text in enumerate(("Full view", f"{fps} FPS")):
            label(width - margin - text_width(text, info_font),
                  height - margin - line_height * (2 - index), text)
        return image
