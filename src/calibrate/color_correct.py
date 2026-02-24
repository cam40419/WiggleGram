from __future__ import annotations

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path(__file__).resolve().parent / "color_profile.npz"

# ---------------------------------------------------------------------------
# X-Rite ColorChecker Classic 24-patch sRGB reference values (D65, 8-bit)
# Order: row-major, top-left to bottom-right
# Patch layout (6 columns × 4 rows):
#   Row 1: Dark Skin, Light Skin, Blue Sky, Foliage, Blue Flower, Bluish Green
#   Row 2: Orange, Purplish Blue, Moderate Red, Purple, Yellow Green, Orange Yellow
#   Row 3: Blue, Green, Red, Yellow, Magenta, Cyan
#   Row 4: White 9.5, Neutral 8, Neutral 6.5, Neutral 5, Neutral 3.5, Black 2
# ---------------------------------------------------------------------------
COLORCHECKER_REFERENCE_SRGB = np.array([
    # Row 1
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67],
    [133, 128, 177], [ 94, 190, 172],
    # Row 2
    [220, 123,  44], [ 72,  91, 166], [194,  84,  97], [ 91,  59, 106],
    [157, 188,  64], [224, 163,  46],
    # Row 3
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31],
    [187,  86, 149], [  8, 133, 161],
    # Row 4
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121],
    [ 85,  85,  85], [ 52,  52,  52],
], dtype=np.float32)

N_ROWS = 4
N_COLS = 6
N_PATCHES = N_ROWS * N_COLS

# Image helpers (mirror the pipeline in a_better_hope.py)
def capture_image() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "input" / f"calibration_{timestamp}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Capturing image with rpicam-still → %s", out)
    subprocess.run(
        ["rpicam-still", "-o", str(out), "--nopreview", "--quality", "95",
         "--autofocus-mode", "auto"],
        check=True,
    )
    return out


def split_and_rotate(raw: Image.Image) -> list[Image.Image]:
    """Split 2x2 grid and rotate 90° CW — same as a_better_hope.py."""
    w, h = raw.size
    pw, ph = w // 2, h // 2
    pieces = []
    for r in range(2):
        for c in range(2):
            crop = raw.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            pieces.append(crop.rotate(90, expand=True))
    return pieces

# Interactive corner selection
def pick_corners(image_rgb: np.ndarray, cam_idx: int) -> np.ndarray:
    """Show the image and ask the user to click 4 corners of the chart.

    Returns an (4, 2) float32 array: TL, TR, BR, BL.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_rgb)
    ax.set_title(
        f"Camera {cam_idx}  –  Click the 4 OUTER CORNERS of the colour chart\n"
        "Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left\n"
        "(close window when done)",
        fontsize=10,
    )

    corners: list[tuple[float, float]] = []
    markers = []

    labels = ["TL", "TR", "BR", "BL"]
    colours = ["red", "green", "blue", "orange"]

    def on_click(event):
        if event.inaxes != ax:
            return
        if len(corners) >= 4:
            return
        x, y = event.xdata, event.ydata
        corners.append((x, y))
        idx = len(corners) - 1
        m = ax.plot(x, y, "x", color=colours[idx], markersize=14, markeredgewidth=3)[0]
        ax.annotate(labels[idx], (x, y), color=colours[idx], fontsize=12,
                    textcoords="offset points", xytext=(8, 8))
        markers.append(m)
        if len(corners) == 4:
            xs = [c[0] for c in corners] + [corners[0][0]]
            ys = [c[1] for c in corners] + [corners[0][1]]
            ax.plot(xs, ys, "--", color="yellow", linewidth=1.5)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(corners) < 4:
        raise RuntimeError(f"Camera {cam_idx}: only {len(corners)} corners selected – need 4.")

    return np.array(corners, dtype=np.float32)


# Patch extraction via perspective warp
def extract_patches(image_rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the chart region to a canonical rectangle and sample patch centres.

    Returns (24, 3) float32 array of mean sRGB values [0, 255].
    """
    dst_w, dst_h = 600, 400
    dst_corners = np.array(
        [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(image_rgb, M, (dst_w, dst_h))

    # Each patch occupies (dst_w/6) × (dst_h/4) pixels.
    # Sample the central 50 % of each patch to avoid border bleed.
    cell_w = dst_w / N_COLS
    cell_h = dst_h / N_ROWS
    margin_x = cell_w * 0.25
    margin_y = cell_h * 0.25

    patches = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x0 = int(c * cell_w + margin_x)
            x1 = int((c + 1) * cell_w - margin_x)
            y0 = int(r * cell_h + margin_y)
            y1 = int((r + 1) * cell_h - margin_y)
            region = warped[y0:y1, x0:x1].astype(np.float32)
            patches.append(region.reshape(-1, 3).mean(axis=0))

    return np.array(patches, dtype=np.float32)  # (24, 3)


def show_patch_comparison(measured: np.ndarray, reference: np.ndarray, cam_idx: int):
    """Display measured vs reference patches for visual verification."""
    fig, axes = plt.subplots(2, N_PATCHES, figsize=(18, 3))
    for i in range(N_PATCHES):
        axes[0, i].imshow(
            np.full((30, 30, 3), np.clip(measured[i], 0, 255).astype(np.uint8))
        )
        axes[0, i].axis("off")
        axes[1, i].imshow(
            np.full((30, 30, 3), reference[i].astype(np.uint8))
        )
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Measured", fontsize=8)
    axes[1, 0].set_ylabel("Reference", fontsize=8)
    fig.suptitle(f"Camera {cam_idx} – top: measured  /  bottom: reference", fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CCM fitting
# ---------------------------------------------------------------------------

def fit_ccm(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fit a 3×3 colour-correction matrix via least squares.

    Solves:  measured_linear @ CCM.T ≈ reference_linear

    Both inputs are in [0, 255] sRGB.  We convert to linear light before
    fitting so the matrix operates in a perceptually uniform space.

    Returns CCM shape (3, 3).
    """
    def srgb_to_linear(x: np.ndarray) -> np.ndarray:
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    src = srgb_to_linear(measured)   # (24, 3)
    dst = srgb_to_linear(reference)  # (24, 3)

    # Least squares: src @ CCM.T = dst  →  CCM.T = pinv(src) @ dst
    ccm_t, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
    ccm = ccm_t.T  # (3, 3)
    logger.info("CCM:\n%s", np.round(ccm, 4))
    return ccm.astype(np.float32)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(image_path: Path | None = None):
    if image_path is None:
        image_path = capture_image()
    else:
        image_path = Path(image_path)

    logger.info("Loading %s", image_path)
    raw = Image.open(image_path)
    raw = ImageOps.exif_transpose(raw).convert("RGB")
    pieces = split_and_rotate(raw)

    ccms = []
    for cam_idx, piece in enumerate(pieces):
        logger.info("=== Camera %d ===", cam_idx)
        arr = np.asarray(piece)

        corners = pick_corners(arr, cam_idx)
        measured = extract_patches(arr, corners)
        show_patch_comparison(measured, COLORCHECKER_REFERENCE_SRGB, cam_idx)
        ccm = fit_ccm(measured, COLORCHECKER_REFERENCE_SRGB)
        ccms.append(ccm)

    ccms_array = np.stack(ccms, axis=0)  # (4, 3, 3)
    np.savez(str(PROFILE_PATH), ccms=ccms_array)
    logger.info("Color profile saved → %s", PROFILE_PATH)
    print(f"\nProfile saved to: {PROFILE_PATH}")
    print("You can now use apply_color_correction() in a_better_hope.py.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    calibrate(img)
