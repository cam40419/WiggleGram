import os
import signal
import subprocess
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from flash import trigger as flash_trigger
import config as cfg

logger = logging.getLogger(__name__)

# Capture constants
AWB_SETTLE_MS    = 1000
AWB_SETTLE_S     = AWB_SETTLE_MS / 1000.0
SHUTTER_US       = 16667 * 6       # ≈1/60 s * 6
CAPTURE_SETTLE_S = 2.0              # wait after SIGUSR1 for rpicam-still to expose & write
FLASH_OFFSET_S   = 0.30             # flash delay relative to SIGINT (seconds)


def do_capture() -> Tuple[Optional[Path], Optional[Path]]:
    """Run rpicam-still + flash and return (input_file, output_dir) on success."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = cfg.INPUT_DIR / f"photo_{timestamp}.jpg"
    output_dir = cfg.OUTPUT_DIR
    input_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen([
            "rpicam-still",
            "-o", str(input_file),
            "--nopreview",
            "--quality", "100",
            "--awb", "auto",
            "--denoise", "cdn_hq",
            "--sharpness", "1.5",
            "--ev", "0.5",
            "-t", "0",
            "--signal",
            "--shutter", str(SHUTTER_US),
        ])

        time.sleep(AWB_SETTLE_S)
        # Queue a capture request, then wait for rpicam-still to expose and write
        os.kill(proc.pid, signal.SIGUSR1)
        
        # Fire flash after FLASH_OFFSET_S, send SIGINT immediately — both on separate threads
        def delayed_flash():
            time.sleep(FLASH_OFFSET_S)
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()

        flash_thread = threading.Thread(target=delayed_flash, daemon=True)
        flash_thread.start()
        time.sleep(CAPTURE_SETTLE_S)
        proc.send_signal(signal.SIGINT)
        flash_thread.join()
        proc.wait()

        # ensure the captured file shows up before returning
        for _ in range(20):
            if input_file.exists():
                break
            time.sleep(0.1)
        if not input_file.exists():
            logger.error("Capture succeeded but file missing: %s", input_file)
            return None, None

        logger.info("Captured: %s", input_file)
        return input_file, output_dir

    except Exception as exc:
        logger.error("Capture failed: %s", exc)
        return None, None
