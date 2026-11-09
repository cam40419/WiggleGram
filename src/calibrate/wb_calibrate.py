from __future__ import annotations

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from camera_settings import still_command
from framing import camera_views

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROFILE_PATH = cfg.WB_PROFILE_PATH
PROFILE_DIR = PROFILE_PATH.parent

# Store gain maps at 1/8 of camera resolution — enough to capture vignetting,
# tiny enough to keep the .npz small.
GAIN_MAP_DOWNSAMPLE = 8

# Clamp per-pixel gains to this range to prevent noise amplification in dark corners.
GAIN_MIN = 0.25
GAIN_MAX = 4.0


# Helpers
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


# Capture / split
def capture_image() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cfg.INPUT_DIR / f"wb_calibration_{timestamp}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Capturing image with rpicam-still → %s", out)
    subprocess.run(still_command(out), check=True)
    return out


def split_and_rotate(raw: Image.Image) -> list[Image.Image]:
    """Use the pipeline's full views with exactly 90 degrees CCW rotation."""
    return camera_views(raw)


# Spatial gain map computation
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


# Main calibration routine
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
    print("White balance will now be applied automatically by the wigglegram pipeline.")

if __name__ == "__main__":
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    calibrate(img)
