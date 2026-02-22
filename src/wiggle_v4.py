import os
import logging
from typing import List, Tuple, Optional, Literal

from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

AxisType = Literal["x", "xy"]


def main(raw_path: str, save_path: str):
    pieces = load_images(raw_path)

    # Pick an anchor point on EVERY frame (you click the same physical point, e.g. eye/logo)
    anchors = pick_anchors_all_frames(pieces)

    w, h = pieces[0].size
    target_xy = (w // 2, h // 2)

    wigglegram_manual_anchors(
        pieces,
        anchors=anchors,
        save_dir=save_path,
        target_xy=target_xy,
        axis="x",
        duration_ms=100,
        crop_common=True,
    )


def load_images(input_path: str) -> List[Image.Image]:
    files = sorted(os.listdir(input_path))
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise ValueError(f"No images found in: {input_path}")

    images: List[Image.Image] = []
    for f in image_files:
        p = os.path.join(input_path, f)
        im = Image.open(p)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.load()
        images.append(im)

    return images


def wigglegram_manual_anchors(
    images: List[Image.Image],
    *,
    anchors: List[Tuple[int, int]],
    save_dir: str,
    target_xy: Optional[Tuple[int, int]] = None,
    axis: AxisType = "x",
    duration_ms: int = 100,
    crop_common: bool = True,
):
    """
    Manual anchor workflow:
      - You click an anchor point on every frame (anchors[i])
      - For each frame i, we translate by (dx,dy) so that anchors[i] lands on target_xy
      - This guarantees the selected point is synchronized across frames.
    """
    if not images:
        raise ValueError("No images provided")
    if len(anchors) != len(images):
        raise ValueError(f"anchors length ({len(anchors)}) must match images length ({len(images)})")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "wiggle.gif")

    base_w, base_h = images[0].size
    if any(im.size != (base_w, base_h) for im in images):
        images = [im.resize((base_w, base_h), Image.Resampling.BILINEAR) for im in images]

    if target_xy is None:
        target_xy = (base_w // 2, base_h // 2)

    tx, ty = target_xy

    # Compute per-frame translations from manual anchors
    shifts: List[Tuple[int, int]] = []
    for (ax, ay) in anchors:
        dx = int(round(tx - ax))
        dy = 0 if axis == "x" else int(round(ty - ay))
        shifts.append((dx, dy))

    shifted = [translate_image(images[i], shifts[i]) for i in range(len(images))]

    if crop_common:
        shifted = crop_to_common_valid_area(shifted, shifts)

    snake = shifted + shifted[-2:0:-1]

    pal0 = snake[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    out_frames = [pal0]
    for fr in snake[1:]:
        out_frames.append(fr.quantize(palette=pal0))

    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
        format="GIF",
    )

    logger.info("Saved wiggle GIF to %s (manual anchors, axis=%s)", out_path, axis)


def translate_image(im: Image.Image, shift_xy: Tuple[int, int]) -> Image.Image:
    # Apply translation via affine transform; negative sign because Pillow uses inverse mapping
    dx, dy = shift_xy
    w, h = im.size
    return im.transform(
        (w, h),
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=None,
    )


def crop_to_common_valid_area(frames: List[Image.Image], shifts: List[Tuple[int, int]]) -> List[Image.Image]:
    """
    After translating, edges may contain empty/invalid regions (depending on fill).
    This crops all frames to the intersection region that is guaranteed to be valid
    for all shifts, which reduces edge flicker/breathing.

    For pure translations:
      - If dx > 0: frame content shifted right => left edge is "new"
      - If dx < 0: content shifted left  => right edge is "new"
    We crop based on max positive and negative shifts.
    """
    if not frames:
        return frames

    w, h = frames[0].size
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]

    max_right = max(0, max(dxs))    # content moved right, invalid left margin width = max_right
    max_left = max(0, -min(dxs))    # content moved left, invalid right margin width = max_left
    max_down = max(0, max(dys))     # content moved down, invalid top margin = max_down
    max_up = max(0, -min(dys))      # content moved up, invalid bottom margin = max_up

    x0 = max_right
    x1 = w - max_left
    y0 = max_down
    y1 = h - max_up

    # Clamp to at least 1x1
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    return [fr.crop((x0, y0, x1, y1)) for fr in frames]


def pick_anchors_all_frames(images: List[Image.Image]) -> List[Tuple[int, int]]:
    """
    Shows each frame, asks you to click the anchor point, and stores (x,y).
    Tip: click the exact same physical point on the subject each time (eye highlight, logo corner).
    """
    import matplotlib.pyplot as plt

    anchors: List[Tuple[int, int]] = []

    for i, im in enumerate(images):
        fig, ax = plt.subplots()
        ax.imshow(im)
        ax.set_title(f"Frame {i+1}/{len(images)}: click the SAME point on the subject, then close window")
        pt = plt.ginput(1)[0]
        plt.close(fig)
        anchors.append((int(pt[0]), int(pt[1])))

    return anchors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # raw_path should be a directory
    main("input/ben", "output/ben")