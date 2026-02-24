"""
GPIO 26 transistor test
=======================
Drives BCM GPIO 26 to switch a transistor bridging a 5V signal.

Usage:
    python3 src/test_gpio26.py           # interactive toggle loop
    python3 src/test_gpio26.py on        # set HIGH and exit
    python3 src/test_gpio26.py off       # set LOW and exit
    python3 src/test_gpio26.py blink 5  # blink 5 times then exit
"""

import sys
import time
import signal

PIN = 26  # BCM

try:
    import RPi.GPIO as GPIO
except ImportError:
    sys.exit("RPi.GPIO not found.  Install with:  pip3 install RPi.GPIO")


def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.OUT, initial=GPIO.LOW)
    print(f"GPIO {PIN} configured as output (LOW)")


def cleanup(sig=None, frame=None):
    GPIO.output(PIN, GPIO.LOW)
    GPIO.cleanup()
    print("\nGPIO cleaned up, pin LOW.")
    sys.exit(0)


def set_pin(state: bool):
    GPIO.output(PIN, GPIO.HIGH if state else GPIO.LOW)
    print(f"GPIO {PIN} → {'HIGH (transistor ON)' if state else 'LOW  (transistor OFF)'}")


def blink(count: int = 5, on_ms: int = 500, off_ms: int = 500):
    for i in range(count):
        set_pin(True)
        time.sleep(on_ms / 1000)
        set_pin(False)
        time.sleep(off_ms / 1000)
        print(f"  blink {i + 1}/{count}")


def interactive():
    print("\nInteractive mode – commands: on / off / blink [n] / quit")
    while True:
        try:
            cmd = input("gpio26> ").strip().lower()
        except EOFError:
            break

        if cmd in ("on", "1"):
            set_pin(True)
        elif cmd in ("off", "0"):
            set_pin(False)
        elif cmd.startswith("blink"):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 5
            blink(n)
        elif cmd in ("quit", "q", "exit"):
            break
        elif cmd:
            print("Unknown command.  Use: on / off / blink [n] / quit")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    setup()

    args = sys.argv[1:]
    if not args:
        interactive()
    elif args[0] == "on":
        set_pin(True)
        print("Pin held HIGH – Ctrl+C to release and clean up.")
        signal.pause()
    elif args[0] == "off":
        set_pin(False)
    elif args[0] == "blink":
        n = int(args[1]) if len(args) > 1 else 5
        blink(n)
    else:
        print(__doc__)

    cleanup()
