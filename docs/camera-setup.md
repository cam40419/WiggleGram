# Manual camera setup: Pi 4B + UC-512 + four UC-873 IMX519 modules

Set up and test the camera hardware before installing or starting WiggleGram.
These steps are manual; you do not need to run `scripts/setup_imx519.py`.

## 1. Connect the matching hardware

With the Pi shut down and power disconnected, connect the HAT to the Pi's
**CAMERA/CSI port** and attach **all four** camera modules to the HAT. Check
the connector labels and cable contact orientation against
[Arducam's connection diagrams](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#hardware-connection).
Keep all four cameras connected throughout testing; the quad HAT can fail to
initialize with a missing module. See
[Arducam's quad-camera troubleshooting](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/Troubleshooting/#1-streaming-errors-when-physically-disconnecting-cameras-quad-camera-kit).

Because this rig changed sensor types, check that the HAT's firmware and
replacement modules are compatible with the **IMX519 CamArray kit (B0388)**.
UC-512 identifies the board, not a guarantee that any sensor can be swapped
onto it. Arducam documents sensor-specific CamArray firmware and modified
kit modules in its [CamArray compatibility notes](https://www.arducam.com/blog/camarray-release-multiple-mipi-camera-solution/).
If the HAT came from a different sensor kit or the cameras were bought
individually, ask Arducam to verify the original kit SKU and replacement module
compatibility. Reflashing the Pi does not change the HAT's firmware.

## 2. Start with Raspberry Pi OS

For this bring-up, use **Raspberry Pi OS 64-bit based on Debian 12 Bookworm**,
either Desktop or Lite. This is the OS explicitly covered by Arducam's
[CamArray Pi 4 instructions and support table](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#supported-platforms-and-os).
Arducam's separate [IMX519 guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/)
also lists Trixie support; the Bookworm route keeps this procedure aligned
with the CamArray guide. Confirm the actual OS rather than relying on a
"latest" image label:

```sh
cat /etc/os-release
uname -r
```

For the Bookworm instructions below, expect `VERSION_CODENAME=bookworm` and
a kernel at least **6.1.21**. Connect to the internet, then update the base OS
before installing vendor camera packages:

```sh
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y python3 ca-certificates wget unzip git i2c-tools
sudo reboot
```

## 3. Get the camera software from Arducam

There are three separate parts: the Linux **IMX519 sensor driver**, the
**libcamera library/tuning**, and the **rpicam capture applications**. Current
Raspberry Pi kernels supply the sensor driver and `imx519` device-tree overlay;
the overlay enables it at boot. See the
[Raspberry Pi overlay reference](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/arch/arm/boot/dts/overlays/README).

Install the vendor userspace packages using the exact installer and package
names from [Arducam's IMX519 CamArray instructions](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#imx519-multi-camera-kit):

```sh
mkdir -p ~/arducam-imx519
cd ~/arducam-imx519
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera
./install_pivariety_pkgs.sh -p libcamera_apps
sudo dpkg --audit
```

Run the install commands one at a time and check for download or package
errors. `dpkg --audit` should print nothing. The installer selects packages
for the detected OS/architecture; do not substitute `.deb` files from another
OS release or a Jetson guide.

On the modern-kernel route above, the CamArray guide does **not** request
`kernel_driver` or `imx519_kernel_driver_low_speed`; those appear in legacy
instructions for older kernels. Likewise, use the **IMX519** overlay below;
the `arducam-pivariety` overlay is for a different camera family.

## 4. Enable IMX519 in the boot configuration

On Bookworm (also Trixie), back up and edit:

```sh
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.before-imx519
sudo nano /boot/firmware/config.txt
```

Change the existing `camera_auto_detect=1` to `camera_auto_detect=0`. Under
an `[all]` section at the end, ensure these settings are active:

```ini
[all]
camera_auto_detect=0
dtoverlay=imx519
```

Keep just one active `dtoverlay=imx519`. Comment out conflicting camera
overlays such as `imx219`, `imx477`, `arducam-pivariety`, or
`camera-mux-4port`. Preserve display settings such as `dtoverlay=vc4-kms-v3d`.
This Pi 4B uses `dtoverlay=imx519` without the Pi 5's `,cam0` suffix.
Use the modern libcamera stack; leave the legacy camera stack disabled.

Save in nano with Ctrl+O, Enter, Ctrl+X, then check that the overlay exists
and reboot:

```sh
ls -l /boot/firmware/overlays/imx519.dtbo
sudo reboot
```

If the overlay is missing, stop and check the Raspberry Pi OS kernel/firmware
installation. Do not download an arbitrary overlay from another kernel.
On Bullseye only, the configuration and overlay paths are `/boot/config.txt`
and `/boot/overlays/imx519.dtbo` instead.

## 5. Verify camera detection

After reboot, with no camera apps running:

```sh
rpicam-still --version
rpicam-still --list-cameras
```

If the commands are named `libcamera-still` on your installation, use that
name for both commands instead. The
[Raspberry Pi camera software reference](https://www.raspberrypi.com/documentation/accessories/camera-software.html)
documents the newer `rpicam-*` names and command options.

Expect **one camera device**, typically a line beginning with:

```text
0 : imx519 [4656x3496 ...]
```

The HAT combines the sensors into that single device. You do not select
four separate libcamera device indices. If this says "No cameras available",
resolve detection before continuing to capture; see the last section.

## 6. Ensure four-in-one mode

**Four-in-one is the default.** First try a capture in step 7. If you get
only one or two views, close any camera processes and use Arducam's
[composition-mode commands](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#switch-composition-mode)
on HAT firmware that supports mode switching:

```sh
sudo modprobe i2c-dev
i2cdetect -l
```

The Pi 4 example uses camera I2C bus **10**; confirm `i2c-10` is listed.
For that bus, the documented command to restore four-in-one mode is:

```sh
sudo i2cset -y 10 0x24 0x24 0x00
```

Use the documented bus for your hardware if it differs. If the command fails,
keep the error and check HAT firmware support; do not add force flags or
write to other addresses. A successful I2C write only selects the layout;
it does not prove that all four camera images are healthy.

## 7. Capture a test image

Once these dev-branch files are on the Pi, run from the repository root:

```sh
python3 scripts/capture_quad.py -o quad-test.jpg
```

This script uses only Python's standard library and the installed camera
command, so it works without a virtual environment, pip packages, `.env`,
GPIO, flash, calibration, or the WiggleGram UI. It checks IMX519 enumeration,
makes **one still-capture request**, and saves the native composite without
rotation or cropping. It does not change the HAT composition mode. Without
`-o`, it uses a timestamped filename; an explicitly named output is overwritten
by the camera application on repeat runs.

To test directly without any repository script, the equivalent command is:

```sh
rpicam-still --camera 0 --nopreview \
    --mode 4656:3496:10 --width 4656 --height 3496 \
    --viewfinder-mode 2328:1748:10 \
    --viewfinder-width 1164 --viewfinder-height 874 \
    --roi 0,0,1,1 --rotation 0 --quality 100 \
    --awb auto --autofocus-mode auto --timeout 3000 \
    --output quad-test.jpg
```

Use `libcamera-still` in its place if needed. No `sudo` or desktop is normally
needed to capture; `--nopreview` also works over SSH.

Open `quad-test.jpg`. Expect **4656x3496 total**, divided into a **2x2 grid**
with **2328x1748 per camera**. Put a different labeled object in front of each
lens to confirm all four views. This is the maximum combined image, not four
independent 16MP files. Arducam also notes that pair 2/3 lags pair 0/1 by one
frame, so a composite capture does not guarantee identical exposure times
across all four sensors. See the
[IMX519 quad-kit specifications and timing note](https://www.arducam.com/arducam-16mp-imx519-autofocus-synchronized-quad-camera-kit-for-raspberry-pi-nvidia-jetson.html).

## If detection or capture fails

Keep the full output of these commands:

```sh
cat /etc/os-release
uname -a
ls -l /boot/firmware/overlays/imx519.dtbo
modinfo imx519
rpicam-still --version
rpicam-still --list-cameras
sudo dmesg | grep -Ei 'imx519|arducam|unicam|ak7375'
```

Substitute `libcamera-still` if appropriate. You can also collect the
repository's read-only detection report:

```sh
sudo python3 scripts/setup_imx519.py --check 2>&1 | tee camera-check.log
```

- **No cameras available:** verify the correct boot file, active IMX519
  overlay, and reboot. Kernel sensor-probe/I2C errors point toward power,
  cabling, a missing module, or HAT/module compatibility; they are not caused
  by GIF rotation or aspect ratio.
- **IMX519 lists, capture fails:** keep the capture error and kernel log.
  Check for another process using the camera, library/package mismatch,
  missing image buffers, or a transport timeout according to the actual error.
- **Only one/two views:** check the HAT composition mode in step 6.
- **Missing/black quadrants:** power down before checking all four ribbon
  connections and matching module types. A successful JPEG alone cannot
  certify four functioning sensors.

Avoid repeated package swaps until the logs identify which layer failed.
