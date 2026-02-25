import os
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config as cfg
from manual_align import get_manual_anchors

logger = logging.getLogger(__name__)

AxisType = Literal["x", "xy"]


def split_grid(img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    """Split an image into a grid of pieces."""
    width, height = img.size
    pw, ph = width // cols, height // rows
    pieces = []
    for r in range(rows):
        for c in range(cols):
            pieces.append(img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)))
    return pieces


def apply_rotation_calibration(images: List[Image.Image]) -> List[Image.Image]:
    """Rotate each camera image by the CCW angle stored in rotation_calibration.npz."""
    
    calib_path = cfg.ROTATION_CALIBRATION_PATH
    
    if not calib_path.exists():
        logger.debug("Rotation calibration not found: %s (skipping)", calib_path)
        return images
    data   = np.load(str(calib_path))
    angles = data["angles"].astype(float)
    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(angles) or abs(angles[idx]) < 1e-4:
            corrected.append(im)
        else:
            corrected.append(im.rotate(
                float(angles[idx]),
                resample=Image.Resampling.BICUBIC,
                expand=False,
            ))
            logger.info("Rotation calibration applied to camera %d: %.4f°", idx, angles[idx])
    return corrected


def apply_color_correction(images: List[Image.Image]) -> List[Image.Image]:
    """Apply color correction using color calibration profile."""
    profile_path = cfg.COLOR_PROFILE_PATH
    if not profile_path.exists():
        logger.warning("Color profile not found: %s (skipping)", profile_path)
        return images
    data = np.load(str(profile_path))
    ccms = data["ccms"]   # (4, 3, 3) float32

    def srgb_to_linear(x):
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(ccms):
            corrected.append(im)
            continue
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        lin = srgb_to_linear(arr)
        flat_cc = np.clip(lin.reshape(-1, 3) @ ccms[idx].T, 0.0, 1.0)
        out = np.clip(linear_to_srgb(flat_cc.reshape(arr.shape)) * 255.0, 0, 255).astype(np.uint8)
        corrected.append(Image.fromarray(out))
        logger.info("Colour correction applied to camera %d", idx)
    return corrected


def apply_white_balance(images: List[Image.Image]) -> List[Image.Image]:
    """Apply white balance using gain maps from calibration profile."""
    profile_path = cfg.WB_PROFILE_PATH
    if not profile_path.exists():
        logger.warning("White balance profile not found: %s (skipping)", profile_path)
        return images
    data = np.load(str(profile_path))
    if "gain_maps" not in data:
        logger.warning("White balance profile missing 'gain_maps' — re-run wb_calibrate.py (skipping)")
        return images
    gain_maps = data["gain_maps"]  # (4, ds_h, ds_w, 3)

    def srgb_to_linear(x):
        x = np.clip(x / 255.0, 0.0, 1.0)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(gain_maps):
            corrected.append(im)
            continue
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        h, w = arr.shape[:2]
        gm  = cv2.resize(gain_maps[idx], (w, h), interpolation=cv2.INTER_LINEAR)
        lin = srgb_to_linear(arr)
        out = np.clip(linear_to_srgb(np.clip(lin * gm, 0.0, 1.0)) * 255.0, 0, 255).astype(np.uint8)
        corrected.append(Image.fromarray(out))
        logger.info("White balance applied to camera %d", idx)
    return corrected


def pick_anchors_template(
    images: List[Image.Image],
    *,
    target_xy: Tuple[int, int],
    template_fraction: float = 0.25,
    min_score: float = 0.3,
    ref_idx: int = 1,
) -> List[Optional[Tuple[int, int]]]:
    """Align frames by template-matching the centre patch of the reference frame."""
    if not images:
        return []

    def to_gray(pil_im):
        return cv2.cvtColor(np.asarray(pil_im.convert("RGB")), cv2.COLOR_RGB2GRAY)

    tx, ty = target_xy
    ref_gray = to_gray(images[ref_idx])
    h_ref, w_ref = ref_gray.shape
    half_tw = int(w_ref * template_fraction / 2)
    half_th = int(h_ref * template_fraction / 2)
    tmpl = ref_gray[ty - half_th:ty + half_th, tx - half_tw:tx + half_tw]
    logger.info("Template: %dx%d patch from frame %d", tmpl.shape[1], tmpl.shape[0], ref_idx)

    anchors: List[Optional[Tuple[int, int]]] = []
    for idx in range(len(images)):
        if idx == ref_idx:
            anchors.append((tx, ty))
            continue
        gray = to_gray(images[idx])
        result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < min_score:
            logger.warning("Template match %.3f too low for frame %d (need %.2f), no shift",
                           max_val, idx, min_score)
            anchors.append(None)
            continue
        found_cx = max_loc[0] + half_tw
        found_cy = max_loc[1] + half_th
        anchors.append((found_cx, found_cy))
        logger.info("Frame %d match=%.3f anchor=(%d,%d) shift=(%d,%d)",
                    idx, max_val, found_cx, found_cy, tx - found_cx, ty - found_cy)
    return anchors


def save_anchor_debug(
    images: List[Image.Image],
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    cross_size: int = 40,
    thickness: int = 6,
):
    """Save debug images with anchor points marked."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, (im, anchor) in enumerate(zip(images, anchors)):
        arr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR).copy()
        if anchor is not None:
            ax, ay = anchor
            s = cross_size
            cv2.line(arr, (ax - s, ay - s), (ax + s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr, (ax + s, ay - s), (ax - s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr, (ax - s, ay - s), (ax + s, ay + s), (0, 0, 255), thickness)
            cv2.line(arr, (ax + s, ay - s), (ax - s, ay + s), (0, 0, 255), thickness)
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(
            os.path.join(save_dir, f"debug_anchor_cam{idx}.jpg"), quality=90
        )
    logger.info("Saved anchor debug images to %s", save_dir)


def translate_image(im: Image.Image, shift_xy: Tuple[int, int]) -> Image.Image:
    """Translate an image by the given x,y shift."""
    dx, dy = shift_xy
    w, h = im.size
    return im.transform(
        (w, h), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR, fillcolor=None,
    )


def crop_to_common_valid_area(
    frames: List[Image.Image], shifts: List[Tuple[int, int]]
) -> List[Image.Image]:
    """Crop all frames to the common valid area after translation."""
    if not frames:
        return frames
    w, h = frames[0].size
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]
    x0 = max(0, min(max(0, max(dxs)), w - 1))
    x1 = max(x0 + 1, min(w - max(0, -min(dxs)), w))
    y0 = max(0, min(max(0, max(dys)), h - 1))
    y1 = max(y0 + 1, min(h - max(0, -min(dys)), h))
    return [fr.crop((x0, y0, x1, y1)) for fr in frames]


def wigglegram_anchors(
    images: List[Image.Image],
    *,
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    filename: str,
    target_xy: Optional[Tuple[int, int]] = None,
    axis: AxisType = "x",
    duration_ms: int = 100,
    crop_common: bool = True,
):
    """Generate a wigglegram GIF from aligned images using anchor points."""
    if not images:
        raise ValueError("No images provided")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{filename}.gif")

    base_w, base_h = images[0].size
    if any(im.size != (base_w, base_h) for im in images):
        images = [im.resize((base_w, base_h), Image.Resampling.BILINEAR) for im in images]

    tx, ty = target_xy if target_xy else (base_w // 2, base_h // 2)
    shifts: List[Tuple[int, int]] = []
    for anchor in anchors:
        if anchor is None:
            shifts.append((0, 0))
        else:
            ax, ay = anchor
            shifts.append((
                int(round(tx - ax)),
                0 if axis == "x" else int(round(ty - ay)),
            ))

    shifted = [translate_image(images[i], shifts[i]) for i in range(len(images))]
    if crop_common:
        shifted = crop_to_common_valid_area(shifted, shifts)

    snake = shifted + shifted[-2:0:-1]
    pal0 = snake[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    out_frames = [pal0] + [fr.quantize(palette=pal0) for fr in snake[1:]]
    out_frames[0].save(
        out_path, save_all=True, append_images=out_frames[1:],
        duration=duration_ms, loop=0, optimize=False, disposal=2, format="GIF",
    )
    logger.info("Saved wiggle GIF to %s (axis=%s)", out_path, axis)


def run_pipeline(raw_path: str, save_path: str, timestamp: str):
    """Full wigglegram processing pipeline."""
    raw    = Image.open(raw_path)
    pieces = split_grid(raw, 2, 2)
    pieces = [p.transpose(Image.Transpose.ROTATE_90) for p in pieces]
    pieces = apply_rotation_calibration(pieces)
    pieces = apply_white_balance(pieces)
    # pieces = apply_color_correction(pieces)

    w, h      = pieces[0].size
    target_xy = (w // 2, h // 2)
    
    # Check if manual alignment mode is enabled
    if cfg.get_runtime("manual_alignment", False):
        logger.info("Manual alignment mode enabled - launching GUI")
        anchors = get_manual_anchors(pieces)
        if anchors is None:
            logger.warning("Manual alignment cancelled - falling back to automatic")
            anchors = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)
    else:
        anchors = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)
    
    # save_anchor_debug(pieces, anchors, save_path)
    wigglegram_anchors(
        pieces, anchors=anchors, save_dir=save_path, filename=timestamp,
        target_xy=target_xy, axis="x", duration_ms=100, crop_common=True,
    )
