import time
import threading
import logging

logger = logging.getLogger(__name__)

FLASH_PIN = 26


def trigger(delay_s: float = 0.0):
    logger.info(f"Triggering flash on GPIO 26 (delay={delay_s:.3f}s)...")
    if delay_s > 0:
        time.sleep(delay_s)
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(FLASH_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(FLASH_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(FLASH_PIN, GPIO.LOW)
        GPIO.cleanup(FLASH_PIN)
        logger.info("Flash triggered on GPIO %d (delay=%.3fs)", FLASH_PIN, delay_s)
    except ImportError:
        logger.warning("RPi.GPIO not available – flash skipped")


def schedule_trigger(delay_s: float):
    """Fire the flash in a background thread after *delay_s* seconds.

    Returns the Thread so the caller can join() it if needed.
    """
    t = threading.Thread(target=trigger, args=(delay_s,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    trigger()