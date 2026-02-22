import os
import logging
from typing import List, Tuple, Optional, Literal

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

AxisType = Literal["x", "xy"]


# -------------------------
# Main pipeline
# -------------------------
def main(raw_path: str, save_path: str):
    pieces = load_images(raw_path)

    # AUTO: compute nose anchors for each frame
    anchors = pick_nose_anchors_all_frames(pieces, model_path="face_landmarker.task")

    w, h = pieces[0].size
    target_xy = (w // 2, h // 2)

    wigglegram_anchors(
        pieces,
        anchors=anchors,
        save_dir=save_path,
        target_xy=target_xy,
        axis="x",
        duration_ms=100,
        crop_common=True,
    )


# -------------------------
# Image loading
# -------------------------
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


# -------------------------
# Nose-anchor detection (MediaPipe Tasks API)
# -------------------------
def pick_nose_anchors_all_frames(
    images: List[Image.Image],
    *,
    model_path: str = "face_landmarker.task",
    max_num_faces: int = 1,
    fail_mode: Literal["raise", "previous", "center"] = "raise",
) -> List[Tuple[int, int]]:
    """
    Returns (x,y) nose-tip anchors for each frame using MediaPipe FaceLandmarker.

    fail_mode:
      - "raise": error if a face isn't detected in any frame
      - "previous": reuse previous anchor if detection fails
      - "center": use image center if detection fails
    """
    if not images:
        raise ValueError("No images provided")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=max_num_faces,
    )

    anchors: List[Tuple[int, int]] = []
    prev_anchor: Optional[Tuple[int, int]] = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        for i, pil_im in enumerate(images):
            # PIL -> numpy RGB
            rgb = np.array(pil_im)  # shape (H,W,3), RGB already

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            w, h = pil_im.size

            if not result.face_landmarks:
                msg = f"No face detected in frame {i+1}/{len(images)}"
                if fail_mode == "raise":
                    raise RuntimeError(msg)
                elif fail_mode == "previous" and prev_anchor is not None:
                    anchors.append(prev_anchor)
                    continue
                elif fail_mode == "center":
                    center = (w // 2, h // 2)
                    anchors.append(center)
                    prev_anchor = center
                    continue
                else:
                    # previous requested but none exists
                    raise RuntimeError(msg + " (and no previous anchor available)")

            # Nose tip landmark index = 1
            nose = result.face_landmarks[0][1]

            x = int(round(nose.x * w))
            y = int(round(nose.y * h))

            # Clamp just in case
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            anchor = (x, y)
            anchors.append(anchor)
            prev_anchor = anchor

    return anchors


# -------------------------
# Wiggle creation using anchors (formerly manual anchors)
# -------------------------
def wigglegram_anchors(
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
    Anchor workflow:
      - anchors[i] is the alignment point for frame i (here: nose tip)
      - translate so anchors[i] lands on target_xy
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

    shifts: List[Tuple[int, int]] = []
    for (ax, ay) in anchors:
        dx = int(round(tx - ax))
        dy = 0 if axis == "x" else int(round(ty - ay))
        shifts.append((dx, dy))

    shifted = [translate_image(images[i], shifts[i]) for i in range(len(images))]

    if crop_common:
        shifted = crop_to_common_valid_area(shifted, shifts)

    # ping-pong loop (forward then backward excluding endpoints)
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

    logger.info("Saved wiggle GIF to %s (nose anchors, axis=%s)", out_path, axis)


def translate_image(im: Image.Image, shift_xy: Tuple[int, int]) -> Image.Image:
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
    if not frames:
        return frames

    w, h = frames[0].size
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]

    max_right = max(0, max(dxs))
    max_left = max(0, -min(dxs))
    max_down = max(0, max(dys))
    max_up = max(0, -min(dys))

    x0 = max_right
    x1 = w - max_left
    y0 = max_down
    y1 = h - max_up

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    return [fr.crop((x0, y0, x1, y1)) for fr in frames]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main("input/ben", "output/ben")