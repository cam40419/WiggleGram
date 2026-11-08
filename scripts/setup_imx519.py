#!/usr/bin/env python3
"""Install the IMX519 CamArray stack and configure a Raspberry Pi 4B.

Run on the Pi: sudo python3 scripts/setup_imx519.py
Guide: https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#imx519-multi-camera-kit
Uses only Python's standard library, independently of the app environment.
"""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from urllib.request import urlopen


INSTALLER_URL = (
    "https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/"
    "releases/download/install_script/install_pivariety_pkgs.sh"
)
BEGIN = "# BEGIN WiggleGram IMX519 CamArray"
END = "# END WiggleGram IMX519 CamArray"
BLOCK = f"{BEGIN}\ncamera_auto_detect=0\ndtoverlay=imx519\n{END}\n"
CAMERA_SETTING = re.compile(
    r"^\s*(?:camera_auto_detect\s*=|dtoverlay\s*=\s*"
    r"(?:imx\d+|ov\d+|arducam(?:[-_][\w-]+)?|camera-mux-\d+port)"
    r"(?=\s*(?:,|#|$)))", re.IGNORECASE,
)


class SetupError(Exception):
    pass


def read_os_release(path: Path = Path("/etc/os-release")) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.startswith("#"):
            parts = shlex.split(value, comments=True)
            values[key] = " ".join(parts)
    return values


def check_platform() -> str:
    if sys.platform != "linux":
        raise SetupError("Run this script on the Raspberry Pi 4B using Raspberry Pi OS.")
    model = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    if not model.startswith("Raspberry Pi 4 Model B"):
        raise SetupError(f"This setup targets the Raspberry Pi 4B; detected: {model}")
    release = read_os_release()
    codename = release.get("VERSION_CODENAME", "")
    if codename not in ("bullseye", "bookworm", "trixie"):
        raise SetupError(f"Unsupported OS: {release.get('PRETTY_NAME', codename)}")
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", platform.release())
    if not match or tuple(map(int, match.groups())) < (6, 1, 21):
        raise SetupError(
            "This setup requires kernel 6.1.21 or newer. Update Raspberry Pi OS "
            "before running it; older kernels need Arducam's legacy driver procedure."
        )
    print(f"Detected {model}, {codename}, kernel {platform.release()}", flush=True)
    return codename


def boot_config_path(codename: str) -> Path:
    if codename == "bullseye":
        return Path("/boot/config.txt")
    return Path("/boot/firmware/config.txt")


def read_boot_configs(config: Path) -> dict:
    """Read the root config and its includes before changing any files."""
    root = config.resolve(strict=True)
    boot_dir = root.parent
    contents = {}

    def visit(path):
        path = path.resolve(strict=True)
        if not path.is_relative_to(boot_dir):
            raise SetupError(f"Boot config include is outside {boot_dir}: {path}")
        if path in contents:
            return
        contents[path] = path.read_text(encoding="utf-8")
        for line in contents[path].splitlines():
            match = re.match(r"^\s*include\s+([^#]+)", line)
            if match:
                # Firmware resolves include names from the boot partition root.
                visit(boot_dir / match.group(1).strip())

    visit(root)
    return contents


def configure_text(text: str, *, root: bool) -> str:
    """Disable camera directives and put one managed block in global scope."""
    if text.count(BEGIN) != text.count(END) or text.count(BEGIN) > 1:
        raise SetupError("Malformed WiggleGram config block; restore its backup first.")
    text = re.sub(r"(?ms)^" + re.escape(BEGIN) + r"\n.*?^" + re.escape(END) + r"\n?", "", text)
    lines = []
    for line in text.splitlines():
        if CAMERA_SETTING.match(line):
            line = "# Disabled by WiggleGram IMX519 setup: " + line
        lines.append(line)
    # Retain unrelated file contents (including their trailing blank lines).
    updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    if root:
        updated = updated.rstrip()
        if updated.splitlines()[-1:] != ["[all]"]:
            updated += "\n\n[all]" if updated else "[all]"
        updated += "\n" + BLOCK
    return updated


def config_updates(config: Path) -> dict:
    root = config.resolve(strict=True)
    return {path: (text, configure_text(text, root=path == root))
            for path, text in read_boot_configs(config).items()}


def backup_configs(paths) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for path in paths:
        backup = path.with_name(f"{path.name}.wigglegram-{stamp}.bak")
        # Exclusive creation protects earlier backups, even on repeated runs.
        with path.open("rb") as source, backup.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        shutil.copymode(path, backup)
        print(f"Backup: {backup}", flush=True)


def write_config(path: Path, text: str) -> None:
    """Replace each config atomically, retaining permissions and ownership."""
    original = path.stat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".wigglegram-", delete=False) as output:
            temporary = Path(output.name)
            output.write(text.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(original.st_mode))
        if hasattr(os, "chown"):
            os.chown(temporary, original.st_uid, original.st_gid)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(command, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def install_camera_stack() -> None:
    # The vendor script uses wget, unzip, sudo, and Debian's packaging tools.
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "ca-certificates", "wget", "unzip", "sudo"])
    with tempfile.TemporaryDirectory(prefix="wigglegram-imx519-") as directory:
        installer = Path(directory) / "install_pivariety_pkgs.sh"
        print(f"Downloading Arducam installer: {INSTALLER_URL}", flush=True)
        with urlopen(INSTALLER_URL, timeout=60) as response:
            script = response.read()
        if not script.startswith(b"#!/bin/bash"):
            raise SetupError("The downloaded Arducam installer is not a Bash script.")
        installer.write_bytes(script)
        # Run the guide's two explicit package installs, without auto-detect (-d).
        # The installer tests nonzero statuses internally, so do not add bash -e.
        for package in ("libcamera", "libcamera_apps"):
            run(["bash", str(installer), "-p", package], cwd=directory)
    # Some vendor versions can mask a package-manager failure; check its state.
    audit = subprocess.run(["dpkg", "--audit"], check=True, text=True, capture_output=True)
    if audit.stdout.strip():
        raise SetupError("Package installation is incomplete:\n" + audit.stdout)
    for kind in ("still", "vid"):
        executable = camera_executable(kind)
        run([executable, "--version"], timeout=30)


def camera_executable(kind: str) -> str:
    for name in (f"rpicam-{kind}", f"libcamera-{kind}"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise SetupError(f"No rpicam-{kind} or libcamera-{kind} command found after installation.")


def check_cameras() -> None:
    executable = camera_executable("still")
    result = subprocess.run([executable, "--list-cameras"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode or not re.search(r"(?im)^\s*\d+\s*:\s*imx519\b", result.stdout):
        raise SetupError(
            "IMX519 is not available. Reboot after setup, stop the camera app, "
            "and check the HAT/ribbons with power disconnected. Diagnostics:\n"
            "  sudo dmesg | grep -Ei 'imx519|arducam|unicam|ak7375'"
        )
    print("IMX519 detected. The UC-512's four views appear as one camera device.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and config diff without changes or downloads.")
    parser.add_argument("--config-only", action="store_true", help="Apply boot settings using an already installed camera stack.")
    parser.add_argument("--check", action="store_true", help="Check IMX519 detection after reboot; makes no changes.")
    args = parser.parse_args(argv)
    if args.check and (args.config_only or args.dry_run):
        parser.error("--check cannot be combined with --config-only or --dry-run")
    try:
        codename = check_platform()
        if args.check:
            check_cameras()
            return 0
        config = boot_config_path(codename)
        updates = config_updates(config)
        print(f"Boot config: {config}", flush=True)
        for path, (before, after) in updates.items():
            if before != after:
                print("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                 fromfile=str(path), tofile=str(path) + " (IMX519)")), end="")
        if args.dry_run:
            if not args.config_only:
                print(f"Would download {INSTALLER_URL} and install libcamera, then libcamera_apps.")
            print("Would back up boot config files and apply the changes above. Reboot is manual.")
            return 0
        if os.geteuid() != 0:
            raise SetupError("Use sudo python3 scripts/setup_imx519.py (or use --dry-run).")
        # Back up before invoking the vendor installer, which may also edit config.
        backup_configs(updates)
        if not args.config_only:
            install_camera_stack()
        overlay = config.resolve().parent / "overlays" / "imx519.dtbo"
        if not overlay.is_file():
            raise SetupError(f"Missing {overlay}. Install a current Raspberry Pi OS kernel/firmware and rerun setup.")
        # Reread in case the vendor packages changed the boot config.
        updated_configs = config_updates(config)
        backup_configs(path for path in updated_configs if path not in updates)
        for path, (before, after) in updated_configs.items():
            if before != after:
                write_config(path, after)
                print(f"Configured: {path}", flush=True)
        print("\nCamera setup complete. Reboot to activate the driver:\n  sudo reboot")
        print("After reboot, verify detection:\n  python3 scripts/setup_imx519.py --check")
        return 0
    except (SetupError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        print("Fix the reported error and rerun. Boot-config backups have a .wigglegram-*.bak suffix.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
