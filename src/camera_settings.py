"""Shared rpicam/libcamera settings for UC-873 (IMX519) CamArray modules.

The HAT appears as one camera with four views in a 2x2 composite. At the
maximum composite size each view is 2328x1748, not the sensor's full 16MP.
Keep that native composite intact; 16:9 framing is applied to individual views.
"""

import shutil
from pathlib import Path
from typing import Optional

import config as cfg


def _positive_int(key: str, default: int) -> int:
    value = int(cfg.get_env_var(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


CAMERA_INDEX = int(cfg.get_env_var("WIGGLEGRAM_CAMERA_INDEX", "0"))
CAPTURE_WIDTH = _positive_int("WIGGLEGRAM_CAPTURE_WIDTH", 4656)
CAPTURE_HEIGHT = _positive_int("WIGGLEGRAM_CAPTURE_HEIGHT", 3496)
VF_WIDTH = _positive_int("WIGGLEGRAM_VF_WIDTH", 1164)
VF_HEIGHT = _positive_int("WIGGLEGRAM_VF_HEIGHT", 874)
VF_FPS = _positive_int("WIGGLEGRAM_VF_FPS", 15)
SETTLE_MS = _positive_int("WIGGLEGRAM_SETTLE_MS", 2000)
SHUTTER_US = _positive_int("WIGGLEGRAM_SHUTTER_US", 16667 * 6)
CAPTURE_TIMEOUT_S = float(cfg.get_env_var("WIGGLEGRAM_CAPTURE_TIMEOUT_S", "15"))
FLASH_OFFSET_S = float(cfg.get_env_var("WIGGLEGRAM_FLASH_OFFSET_S", "0.30"))
FOCUS_MODE = cfg.get_env_var("WIGGLEGRAM_FOCUS_MODE", "auto")
LENS_POSITION = cfg.get_env_var("WIGGLEGRAM_LENS_POSITION")
TUNING_FILE = cfg.get_env_var("WIGGLEGRAM_TUNING_FILE")

if CAMERA_INDEX < 0:
    raise ValueError("WIGGLEGRAM_CAMERA_INDEX must be non-negative")
if any(size % 2 for size in (CAPTURE_WIDTH, CAPTURE_HEIGHT, VF_WIDTH, VF_HEIGHT)):
    raise ValueError("Composite capture and viewfinder dimensions must be even")
if CAPTURE_TIMEOUT_S <= 0 or FLASH_OFFSET_S < 0:
    raise ValueError("Capture timeout must be positive and flash offset non-negative")
if FOCUS_MODE not in ("auto", "manual", "continuous"):
    raise ValueError("WIGGLEGRAM_FOCUS_MODE must be auto, manual, or continuous")
if LENS_POSITION and FOCUS_MODE != "manual":
    raise ValueError("Set WIGGLEGRAM_FOCUS_MODE=manual to use WIGGLEGRAM_LENS_POSITION")


def camera_executable(kind: str) -> str:
    """Prefer Bookworm's command names, with a Bullseye fallback."""
    for name in (f"rpicam-{kind}", f"libcamera-{kind}"):
        if shutil.which(name):
            return name
    raise FileNotFoundError(
        f"Neither rpicam-{kind} nor libcamera-{kind} is installed. "
        "Install the IMX519-compatible camera stack; see README.md."
    )


def _camera_arguments(*, preview: bool = False, camera: Optional[int] = None) -> list:
    # Preview tracks focus; still capture sweeps at startup and then holds focus
    # before SIGUSR1. Capture-time AF would change the existing flash timing.
    focus = "continuous" if preview and FOCUS_MODE == "auto" else FOCUS_MODE
    args = ["--camera", str(CAMERA_INDEX if camera is None else camera),
            "--awb", "auto", "--autofocus-mode", focus]
    if LENS_POSITION:
        args += ["--lens-position", LENS_POSITION]
    # Let libcamera select its IMX519 tuning by default (vc4 / pisp as appropriate).
    if TUNING_FILE:
        args += ["--tuning-file", TUNING_FILE]
    return args


def still_command(output: Path, *, signal_capture: bool = False,
                  shutter_us: Optional[int] = None,
                  executable: Optional[str] = None) -> list:
    args = [executable or camera_executable("still"),
            "-o", str(output), "--nopreview", "--quality", "100",
            "--width", str(CAPTURE_WIDTH), "--height", str(CAPTURE_HEIGHT),
            "--denoise", "cdn_hq", "--sharpness", "1.5", "--ev", "0.5"]
    args += _camera_arguments()
    args += ["-t", "0" if signal_capture else str(SETTLE_MS)]
    if signal_capture:
        args += ["--signal"]
    if shutter_us is not None:
        args += ["--shutter", str(shutter_us)]
    return args


def viewfinder_command(*, camera: Optional[int] = None, width: int = VF_WIDTH,
                       height: int = VF_HEIGHT, fps: int = VF_FPS) -> list:
    return [camera_executable("vid"), "-t", "0", "--codec", "mjpeg", "-o", "-",
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps), "--nopreview"] + _camera_arguments(
                preview=True, camera=camera)
