import os, signal, time, logging, threading, subprocess
from pathlib import Path
from datetime import datetime

from flash import trigger as flash_trigger

import config as cfg

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]   # src/ -> repo root

# ── Flash timing ─────────────────────────────────────────────────────────────
# rpicam-still is started with "-t 0 --signal" so it waits indefinitely.
# Python sleeps for AWB_SETTLE_S to let AWB/AE converge, then fires the flash
# and sends SIGUSR1 to rpicam-still at the same instant, guaranteeing that the
# shutter and flash are synchronised regardless of subprocess startup jitter.
AWB_SETTLE_MS  = 1000                     # AWB/AE settle time before capture (ms)
AWB_SETTLE_S   = AWB_SETTLE_MS / 1000.0  # same value in seconds
SHUTTER_US     = 16667                   # shutter speed in µs (16667 ≈ 1/60 s)
# Measured offset between SIGUSR1 and actual shutter open.
# Flash fires this many seconds AFTER SIGUSR1 so it coincides with shutter open.
FLASH_OFFSET_S = 0.130               # seconds (36 ms measured offset)


def capture_photo(filename: str) -> bool:    
    try:
        # Start rpicam-still in signal-triggered mode.  It will begin AWB/AE
        # immediately but NOT capture until it receives SIGUSR1.  This lets us
        # fire the flash and trigger the shutter at exactly the same instant,
        # with no dependency on subprocess startup timing.
        proc = subprocess.Popen(
            [
                "rpicam-still",
                "-o", str(cfg.OUTPUT_DIR / filename),
                "--nopreview",
                "--quality", "100",             # 95 is perceptually lossless
                #"--autofocus-mode", "manual",  # Fix focus if camera doesn't move
                "--awb", "auto",               # Let AWB converge during the settle window
                "--denoise", "cdn_hq",         # Clean up grain in ceiling shadows
                "--sharpness", "1.5",          # Bump sharpness for pipe edges
                "--ev", "0.8",
                "-t", "0",                     # Wait indefinitely (until SIGUSR1)
                "--signal",                    # Capture on SIGUSR1
                "--shutter", str(SHUTTER_US),  # Fixed shutter speed (µs)
            ],
        )

        # Wait for AWB/AE to converge, then fire flash + shutter simultaneously.
        time.sleep(AWB_SETTLE_S)
        # Trigger shutter first, then fire flash after the measured offset so
        # the flash pulse coincides with the shutter being open.
        def _delayed_flash():
            time.sleep(FLASH_OFFSET_S)
            flash_trigger()
        flash_thread = threading.Thread(target=_delayed_flash, daemon=True)
        flash_thread.start()
        os.kill(proc.pid, signal.SIGUSR1)      # trigger capture immediately

        # rpicam-still --signal stays alive after capturing; wait for the
        # file to be written then send SIGINT to make it exit cleanly.
        flash_thread.join()                    # GPIO cleanup first
        time.sleep(1.5)                        # let rpicam-still finish writing
        proc.send_signal(signal.SIGINT)
        proc.wait()
        logger.info(f"Photo captured: {cfg.OUTPUT_DIR / filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to capture photo: {e}")
        return False


def capture_and_process():
    from a_better_hope import main
    
    # Generate filename with current datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = REPO_ROOT / "input" / f"photo_{timestamp}.jpg"
    output_dir = REPO_ROOT / "output" / timestamp
    
    # Capture photo
    if not capture_photo(input_file):
        return
    
    # Process the captured photo
    main(str(input_file), str(output_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    capture_and_process()
