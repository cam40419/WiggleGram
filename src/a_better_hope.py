import os
import logging
from typing import List, Tuple, Optional, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFile

from pathlib import Path
from datetime import datetime
import subprocess

from flash import trigger as flash_trigger

REPO_ROOT = Path(__file__).resolve().parents[1]   # src/ -> repo root


def resolve_input_file() -> Path:
    captured_files = sorted((REPO_ROOT / "input").glob("photo_*.jpg"))
    if captured_files:
        return captured_files[-1]
    return REPO_ROOT / "input" / "ben" / "IMG_1638.JPG"

# Capture photo on RPi and process it
def capture_and_process():
    
    # Generate filename with current datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = REPO_ROOT / "input" / f"photo_{timestamp}.jpg"
    output_dir = REPO_ROOT / "output" / timestamp
    
    # Create input directory if it doesn't exist
    input_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Capture photo on RPi
    try:
        flash_trigger()
        subprocess.run(
            [
                "rpicam-still",
                "-o", str(input_file),
                "--nopreview",
                "--quality", "95",
                "--autofocus-mode", "auto",
            ],
            check=True
        )
        logger.info(f"Photo captured: {input_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to capture photo: {e}")
        return
    
    # Process the captured photo
    main(str(input_file), str(output_dir))

# INPUT_DIR = resolve_input_file()
# OUTPUT_DIR = REPO_ROOT / "output" / "ben"

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

AxisType = Literal["x", "xy"]

# Main pipeline
def main(raw_path: str, save_path: str):
    # pieces = load_images(raw_path)
    
    raw = Image.open(raw_path)
    pieces = split_grid(raw, 2, 2)

    pieces[0] = pieces[0].rotate(90, expand=True)
    pieces[1] = pieces[1].rotate(90, expand=True)
    pieces[2] = pieces[2].rotate(90, expand=True)
    pieces[3] = pieces[3].rotate(90, expand=True)

    #pieces = apply_npz_transforms(pieces, REPO_ROOT / "calibration_4cam.npz")

    pieces = apply_color_correction(pieces)

    w, h = pieces[0].size
    target_xy = (w // 2, h // 2)

    anchors = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)

    save_anchor_debug(pieces, anchors, save_path)

    wigglegram_anchors(
        pieces,
        anchors=anchors,
        save_dir=save_path,
        target_xy=target_xy,
        axis="x",
        duration_ms=100,
        crop_common=True,
    )


def pick_anchors_template(
    images: List[Image.Image],
    *,
    target_xy: Tuple[int, int],
    template_fraction: float = 0.25,
    min_score: float = 0.3,
    ref_idx: int = 1,
) -> List[Optional[Tuple[int, int]]]:
    """Align frames by template-matching the center patch of frame ref_idx into all other frames.

    This locks onto whatever foreground object is in the center of the reference frame
    (e.g. a finger, face, or any close-up subject) rather than background texture.

    ref_idx: which camera frame to use as the reference (default 1).
    template_fraction: fraction of image width/height used for the template patch.
    min_score: minimum normalised cross-correlation score to accept a match.
    """
    if not images:
        return []

    def to_gray(pil_im: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.asarray(pil_im.convert("RGB")), cv2.COLOR_RGB2GRAY)

    tx, ty = target_xy
    ref_gray = to_gray(images[ref_idx])
    h_ref, w_ref = ref_gray.shape

    # Template patch: central region of reference frame
    half_tw = int(w_ref * template_fraction / 2)
    half_th = int(h_ref * template_fraction / 2)
    tmpl = ref_gray[ty - half_th : ty + half_th, tx - half_tw : tx + half_tw]
    logger.info("Template: %dx%d patch from center of frame %d", tmpl.shape[1], tmpl.shape[0], ref_idx)

    anchors: List[Optional[Tuple[int, int]]] = []

    for idx in range(len(images)):
        if idx == ref_idx:
            anchors.append((tx, ty))  # reference frame needs no shift
            continue

        gray = to_gray(images[idx])
        result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < min_score:
            logger.warning("Template match score %.3f too low for frame %d (need %.2f), no shift",
                           max_val, idx, min_score)
            anchors.append(None)
            continue

        # max_loc is the top-left of the best match; centre it
        found_cx = max_loc[0] + half_tw
        found_cy = max_loc[1] + half_th
        anchor = (found_cx, found_cy)
        anchors.append(anchor)
        logger.info("Template: frame %d match score=%.3f found_center=(%d,%d) shift=(%d,%d)",
                    idx, max_val, found_cx, found_cy, tx - found_cx, ty - found_cy)

    return anchors


def apply_color_correction(images: List[Image.Image], profile_path: Path = COLOR_PROFILE_PATH) -> List[Image.Image]:
    """Apply per-camera 3×3 colour-correction matrix loaded from *profile_path*.

    The CCMs are produced by src/color_correct.py.  If the profile does not
    exist the images are returned unchanged so the pipeline never hard-fails.
    """
    if not profile_path.exists():
        logger.warning("Color profile not found: %s (skipping colour correction)", profile_path)
        return images

    data = np.load(str(profile_path))
    ccms = data["ccms"]  # (4, 3, 3) float32

    def srgb_to_linear(x: np.ndarray) -> np.ndarray:
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(ccms):
            corrected.append(im)
            continue
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        lin = srgb_to_linear(arr)               # (H, W, 3)
        flat = lin.reshape(-1, 3)               # (N, 3)
        flat_cc = flat @ ccms[idx].T            # apply CCM
        flat_cc = np.clip(flat_cc, 0.0, 1.0)
        out = linear_to_srgb(flat_cc.reshape(arr.shape))
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        corrected.append(Image.fromarray(out))
        logger.info("Colour correction applied to camera %d", idx)

    return corrected


def apply_npz_transforms(images: List[Image.Image], npz_path: Path) -> List[Image.Image]:
    if not npz_path.exists():
        logger.warning("Calibration file not found: %s (skipping npz transforms)", npz_path)
        return images

    data = np.load(npz_path, allow_pickle=True)
    Ks = np.asarray(data["Ks"], dtype=np.float64)
    raw_dists = data["dists"]

    transformed: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(Ks) or idx >= len(raw_dists):
            transformed.append(im)
            continue

        dist = np.asarray(raw_dists[idx], dtype=np.float64).reshape(-1, 1)

        rgb = np.asarray(im)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        undistorted = cv2.undistort(bgr, Ks[idx], dist)
        undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        transformed.append(Image.fromarray(undistorted_rgb))

    return transformed


# Image loading
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


def split_grid(img, rows, cols):
    width, height = img.size
    
    piece_width = width // cols
    piece_height = height // rows

    # Create a list to hold the pieces
    pieces = []

    # Loop through the image and add each piece to the list
    for i in range(rows):
        for j in range(cols):
            # Calculate the position of the current piece
            left = j * piece_width
            upper = i * piece_height
            right = (j + 1) * piece_width
            lower = (i + 1) * piece_height
            # Crop the image and add it to the list
            pieces.append(img.crop((left, upper, right, lower)))

    return pieces


def save_anchor_debug(
    images: List[Image.Image],
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    cross_size: int = 40,
    thickness: int = 6,
):
    """Save each frame with a large red X drawn at the detected anchor point."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, (im, anchor) in enumerate(zip(images, anchors)):
        arr = np.asarray(im.convert("RGB")).copy()
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if anchor is not None:
            ax, ay = anchor
            s = cross_size
            cv2.line(arr_bgr, (ax - s, ay - s), (ax + s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr_bgr, (ax + s, ay - s), (ax - s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr_bgr, (ax - s, ay - s), (ax + s, ay + s), (0, 0, 255), thickness)
            cv2.line(arr_bgr, (ax + s, ay - s), (ax - s, ay + s), (0, 0, 255), thickness)
        else:
            logger.warning("No anchor for cam%d — no marker drawn", idx)
        out = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
        out.save(os.path.join(save_dir, f"debug_anchor_cam{idx}.jpg"), quality=90)
    logger.info("Saved anchor debug images to %s", save_dir)


# Wiggle creation using anchors
def wigglegram_anchors(
    images: List[Image.Image],
    *,
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    target_xy: Optional[Tuple[int, int]] = None,
    axis: AxisType = "x",
    duration_ms: int = 100,
    crop_common: bool = True,
):
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
    for anchor in anchors:
        if anchor is None:
            shifts.append((0, 0))  # no transform
            continue

        ax, ay = anchor
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
    capture_and_process()
    #main(str(INPUT_DIR), str(OUTPUT_DIR))