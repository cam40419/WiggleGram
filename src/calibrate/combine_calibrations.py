import sys
import logging
from pathlib import Path
import numpy as np

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INDIVIDUAL_DIR = SCRIPT_DIR / "individual"


def load_calibration_file(path: Path):
    """Load a calibration file and return its contents as a dictionary."""
    if not path.exists():
        logger.warning(f"Calibration file not found: {path}")
        return {}
    try:
        data = np.load(str(path), allow_pickle=True)
        result = {key: data[key] for key in data.keys()}
        logger.info(f"Loaded calibration from {path.name}")
        for key in result.keys():
            logger.info(f"  {key}: shape={result[key].shape}, dtype={result[key].dtype}")
        return result
    except Exception as e:
        logger.error(f"Failed to load calibration {path.name}: {e}")
        return {}


def combine_all_calibrations(output_dir: Path) -> bool:
    """
    Combine all calibration files in the individual/ folder into a single .npz file in output_dir.
    """
    output_path = output_dir / "combined_calibration.npz"
    combined_data = {}
    found_files = list(INDIVIDUAL_DIR.glob("*.npz"))
    if not found_files:
        logger.error(f"No calibration files found in {INDIVIDUAL_DIR}")
        return False
    for calib_file in found_files:
        data = load_calibration_file(calib_file)
        prefix = calib_file.stem
        for key, value in data.items():
            combined_data[f"{prefix}_{key}"] = value
    if not combined_data:
        logger.error("No calibration data to combine. Check that calibration files exist.")
        return False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), **combined_data)
        logger.info(f"\n{'='*70}")
        logger.info(f"Successfully created combined calibration file:")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Total keys: {len(combined_data)}")
        logger.info(f"{'='*70}\n")
        logger.info("Combined file contents:")
        for key, value in combined_data.items():
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        return True
    except Exception as e:
        logger.error(f"Failed to save combined calibration: {e}")
        return False


def main():
    output_dir = cfg.NPZ_DIR
    success = combine_all_calibrations(output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
