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
from camera_settings import (
    SETTLE_MS, SHUTTER_US, CAPTURE_TIMEOUT_S, FLASH_OFFSET_S, still_command,
)

logger = logging.getLogger(__name__)

# Capture constants
AWB_SETTLE_MS    = SETTLE_MS
AWB_SETTLE_S     = AWB_SETTLE_MS / 1000.0
CAPTURE_SETTLE_S = 2.0              # wait after SIGUSR1 for rpicam-still to expose & write


def _jpeg_complete(path: Path) -> bool:
    """Wait for the JPEG end marker rather than accepting a partially written file."""
    try:
        with path.open("rb") as capture:
            capture.seek(-2, os.SEEK_END)
            return capture.read(2) == b"\xff\xd9"
    except OSError:
        return False


def do_capture() -> Tuple[Optional[Path], Optional[Path]]:
    """Run rpicam-still + flash and return (input_file, output_dir) on success."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    input_file = cfg.INPUT_DIR / f"photo_{timestamp}.jpg"
    output_dir = cfg.OUTPUT_DIR
    proc = None
    flash_thread = None
    cancel_flash = threading.Event()

    try:
        input_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(still_command(
            input_file, signal_capture=True, shutter_us=SHUTTER_US,
        ))

        time.sleep(AWB_SETTLE_S)
        if proc.poll() is not None:
            raise RuntimeError(f"Camera exited during startup (status {proc.returncode})")
        # Queue a capture request, then wait for rpicam-still to expose and write
        os.kill(proc.pid, signal.SIGUSR1)
        deadline = time.monotonic() + CAPTURE_TIMEOUT_S
        
        # Keep autofocus at startup so it does not delay this flash schedule.
        def delayed_flash():
            if cancel_flash.wait(FLASH_OFFSET_S):
                return
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()

        flash_thread = threading.Thread(target=delayed_flash, daemon=True)
        flash_thread.start()
        time.sleep(CAPTURE_SETTLE_S)
        while not _jpeg_complete(input_file):
            if proc.poll() is not None:
                raise RuntimeError(f"Camera exited before writing the image (status {proc.returncode})")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Camera did not finish writing {input_file}")
            time.sleep(0.1)

        logger.info("Captured: %s", input_file)
        return input_file, output_dir

    except Exception as exc:
        logger.error("Capture failed: %s", exc)
        return None, None
    finally:
        cancel_flash.set()
        if flash_thread is not None:
            flash_thread.join(timeout=1)
        # Always release the camera so the viewfinder can restart after failures.
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.error("Could not stop capture process: %s", exc)
