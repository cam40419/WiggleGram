"""
White Balance / Flat-Field Calibration Script
==============================================
Point all four cameras at a plain **white piece of paper** filling the entire
frame, then run this script.

Instead of a single per-channel scalar, this script computes a spatially-
varying per-pixel gain map for every camera.  This corrects:
  - Global colour balance (pink / blue overall cast)
  - Spatial vignetting (colour shifts that vary across the frame)

How it works
------------
1. The white-paper image is converted to linear light.
2. Each channel is smoothed with a large Gaussian — this models the sensor's
   underlying spatial response surface, ignoring noise.
3. Per-pixel gain  =  global_target / smoothed_response
   where global_target = mean luminance across all pixels and channels.
4. The gain map is stored downsampled at 1/8 resolution.  It is upsampled
   back to full resolution (bilinear) when applied in a_better_hope.py.

Profile saved to:

    src/white_balance/wb_profile.npz  (key: "gain_maps", shape 4×H×W×3)

Usage
-----
    python3 src/wb_calibrate.py                   # capture via rpicam-still
    python3 src/wb_calibrate.py path/to/img.jpg   # use an existing combined 2×2 image
"""

from __future__ import annotations

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = Path(__file__).resolve().parent / "white_balance"
PROFILE_PATH = PROFILE_DIR / "wb_profile.npz"

# Store gain maps at 1/8 of camera resolution — enough to capture vignetting,
# tiny enough to keep the .npz small.
GAIN_MAP_DOWNSAMPLE = 8

# Clamp per-pixel gains to this range to prevent noise amplification in dark corners.
GAIN_MIN = 0.25
GAIN_MAX = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


# ---------------------------------------------------------------------------
# Capture / split
# ---------------------------------------------------------------------------

def capture_image() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "input" / f"wb_calibration_{timestamp}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Capturing image with rpicam-still → %s", out)
    subprocess.run(
        [
            "rpicam-still",
            "-o", str(out),
            "--nopreview",
            "--quality", "95",
            "--autofocus-mode", "manual",
            "--awb", "auto",
            "--immediate",
            "-t", "500",
            "--tuning-file", "/usr/share/libcamera/ipa/rpi/vc4/imx219_noir.json",
        ],
        check=True,
    )
    return out


def split_and_rotate(raw: Image.Image) -> list[Image.Image]:
    """Split 2×2 grid and rotate 90° CW — matches a_better_hope.py."""
    w, h = raw.size
    pw, ph = w // 2, h // 2
    pieces = []
    for r in range(2):
        for c in range(2):
            crop = raw.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            pieces.append(crop.rotate(90, expand=True))
    return pieces


# ---------------------------------------------------------------------------
# Spatial gain map computation
# ---------------------------------------------------------------------------

def compute_gain_map(image_rgb: np.ndarray) -> np.ndarray:
    """Compute a spatially-varying per-channel gain map from a white-paper frame.

    Steps:
      1. Convert to linear light.
      2. Gaussian-blur each channel (sigma ≈ 5 % of shorter dimension) to model
         the smooth sensor response surface and suppress paper texture / noise.
      3. global_target = mean linear luminance across all pixels & channels.
      4. gain[y,x,c] = global_target / smoothed[y,x,c]   (clamped to GAIN_MIN..GAIN_MAX)
      5. Downsample to 1/GAIN_MAP_DOWNSAMPLE for compact storage.

    Returns float32 array of shape (ds_h, ds_w, 3).
    """
    h, w = image_rgb.shape[:2]

    lin = srgb_to_linear(image_rgb.astype(np.float32))  # (H, W, 3) linear [0,1]

    # Sigma: ~5 % of the shorter image dimension, but at least 20 px.
    sigma = max(20, min(h, w) // 20)
    ksize = int(sigma * 6) | 1  # nearest odd integer

    smoothed = np.stack(
        [cv2.GaussianBlur(lin[:, :, c], (ksize, ksize), sigma) for c in range(3)],
        axis=-1,
    )  # (H, W, 3)

    # Global mean luminance across all channels → our neutral-grey target.
    target = float(lin.mean())

    gain_map = np.where(smoothed > 1e-6, target / smoothed, 1.0)
    gain_map = np.clip(gain_map, GAIN_MIN, GAIN_MAX).astype(np.float32)  # (H, W, 3)

    # Downsample for storage
    ds_h = max(1, h // GAIN_MAP_DOWNSAMPLE)
    ds_w = max(1, w // GAIN_MAP_DOWNSAMPLE)
    gain_small = cv2.resize(gain_map, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    for c, name in enumerate("RGB"):
        logger.info(
            "  cam gain_map %s: min=%.3f  max=%.3f  mean=%.3f",
            name, gain_small[:, :, c].min(), gain_small[:, :, c].max(),
            gain_small[:, :, c].mean(),
        )

    return gain_small  # (ds_h, ds_w, 3)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(image_path: Path | None = None):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    if image_path is None:
        image_path = capture_image()
    else:
        image_path = Path(image_path)

    logger.info("Loading %s", image_path)
    raw = Image.open(image_path)
    raw = ImageOps.exif_transpose(raw).convert("RGB")
    pieces = split_and_rotate(raw)

    all_maps: list[np.ndarray] = []
    for cam_idx, piece in enumerate(pieces):
        logger.info("=== Camera %d ===", cam_idx)
        gm = compute_gain_map(np.asarray(piece))
        all_maps.append(gm)

    gain_maps = np.stack(all_maps, axis=0)  # (4, ds_h, ds_w, 3) float32
    np.savez(str(PROFILE_PATH), gain_maps=gain_maps)
    logger.info("White balance profile saved → %s  shape=%s", PROFILE_PATH, gain_maps.shape)
    print(f"\nProfile saved to: {PROFILE_PATH}")
    print("White balance will now be applied automatically by a_better_hope.py.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    calibrate(img)
