#!/usr/bin/env python3
"""Capture one native IMX519 CamArray JPEG, independently of WiggleGram.

Run on the Pi after manual camera setup:
    python3 scripts/capture_quad.py -o quad-test.jpg

The HAT must already be in four-in-one mode (its default). This script does
not install drivers, change I2C registers, load .env, rotate, or crop images.
Only Python's standard library and rpicam-still/libcamera-still are needed.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(f"quad-test-{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"))
    parser.add_argument("--camera", type=int, default=0,
                        help="The single IMX519 composite device's index (default: 0).")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be non-negative")

    try:
        executable = shutil.which("rpicam-still") or shutil.which("libcamera-still")
        if not executable:
            raise RuntimeError("Install the Arducam camera stack first; see docs/camera-setup.md.")

        # A camera listing can return exit status 0 even with no sensors found.
        listing = subprocess.run([executable, "--list-cameras"], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        print(listing.stdout, end="" if listing.stdout.endswith("\n") else "\n", flush=True)
        if listing.returncode or not re.search(
                rf"(?im)^\s*{args.camera}\s*:\s*imx519\b", listing.stdout):
            raise RuntimeError(
                f"No IMX519 at camera index {args.camera}. Check dtoverlay=imx519, "
                "reboot, and inspect the driver log. See docs/camera-setup.md."
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [executable, "--camera", str(args.camera), "--nopreview",
                   "--mode", "4656:3496:10", "--width", "4656", "--height", "3496",
                   "--viewfinder-mode", "2328:1748:10",
                   "--viewfinder-width", "1164", "--viewfinder-height", "874",
                   "--roi", "0,0,1,1", "--rotation", "0", "--quality", "100",
                   "--awb", "auto", "--autofocus-mode", "auto",
                   "--timeout", "3000", "--output", str(args.output)]
        print("+ " + shlex.join(command), flush=True)
        # Inherit diagnostics so failures are visible in a terminal or SSH session.
        subprocess.run(command, check=True, timeout=60)
        with args.output.open("rb") as image:
            start = image.read(2)
            image.seek(-2, 2)
            if start != b"\xff\xd8" or image.read(2) != b"\xff\xd9":
                raise RuntimeError(f"Capture did not produce a complete JPEG: {args.output}")
        print(f"Saved: {args.output.resolve()}")
        print("Expected: one 4656x3496 JPEG with four 2328x1748 views in a 2x2 grid.")
        print("Open the image and confirm all four distinct views; layout cannot be verified automatically.")
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Capture cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
