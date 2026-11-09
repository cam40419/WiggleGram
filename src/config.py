import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Load all environment variables into a dictionary
ENV_VARS = dict(os.environ)

# Static config variables
ROOT_DIR = Path(ENV_VARS.get("WIGGLEGRAM_ROOT_DIR", Path(__file__).resolve().parent))

# Retain native sensor orientation. Ignore old rotation values in existing .env files.
VIEW_ROTATION = 0

# The previous 90-degree spatial maps must not be applied to unrotated views.
_profile_dir = ROOT_DIR / "calibrate" / "out" / "imx519" / "rotation_0"
NPZ_DIR = Path(ENV_VARS.get("WIGGLEGRAM_CALIBRATION_DIR", _profile_dir))
COLOR_PROFILE_PATH = NPZ_DIR / "color_profile.npz"
WB_PROFILE_PATH = NPZ_DIR / "white_balance_profile.npz"
ROTATION_CALIBRATION_PATH = NPZ_DIR / "rotation_calibration.npz"
COMBINED_CALIBRATION_PATH = NPZ_DIR / "combined_calibration.npz"

MEDIA_DIR = Path(ENV_VARS.get("WIGGLEGRAM_MEDIA_DIR", ROOT_DIR.parent / "camera"))
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

