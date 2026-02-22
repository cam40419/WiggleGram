"""
flash.py  –  fire the transistor flash on BCM GPIO 26 once.
"""

import time
import logging

logger = logging.getLogger(__name__)

FLASH_PIN = 26  # BCM


def trigger():
    print("Triggering flash on GPIO 26...")
    """Drive GPIO 26 HIGH for 100 ms, then LOW."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(FLASH_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(FLASH_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(FLASH_PIN, GPIO.LOW)
        GPIO.cleanup(FLASH_PIN)
        logger.info("Flash triggered on GPIO %d", FLASH_PIN)
    except ImportError:
        logger.warning("RPi.GPIO not available – flash skipped")

if __name__ == "__main__":
    trigger()