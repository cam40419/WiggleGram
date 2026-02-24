import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Load all environment variables into a dictionary
ENV_VARS = dict(os.environ)

# Static config variables
ROOT_DIR = Path("/home/admin/wigglegram/src")

NPZ_DIR = ROOT_DIR / "calibrate" / "out"
COLOR_PROFILE_PATH = NPZ_DIR / "color_profile.npz"
WB_PROFILE_PATH = NPZ_DIR / "white_balance_profile.npz"
ROTATION_CALIBRATION_PATH = NPZ_DIR / "rotation_calibration.npz"
COMBINED_CALIBRATION_PATH = NPZ_DIR / "combined_calibration.npz"

MEDIA_DIR = Path("/home/admin/wigglegram/camera")
INPUT_DIR = MEDIA_DIR / "input"
OUTPUT_DIR = MEDIA_DIR / "output"

# Runtime config (can be updated during runtime)
RUNTIME_CONFIG = {}

def get_env_var(key, default=None):
    """Get an environment variable with an optional default."""
    return ENV_VARS.get(key, default)

def get_runtime(key, default=None):
    return RUNTIME_CONFIG.get(key, default)

def set_runtime(key, value):
    RUNTIME_CONFIG[key] = value

