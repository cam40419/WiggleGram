# WiggleGram

Raspberry Pi camera application for a four-view wigglegram rig. The current
hardware is a Raspberry Pi 4B, an Arducam UC-512 CamArray HAT, and four Arducam
UC-873 (Sony IMX519, 16MP) modules compatible with the CamArray kit.

The application expects the HAT's **four-in-one, 2x2 composite**: camera 0 at
top-left, 1 at top-right, 2 at bottom-left, and 3 at bottom-right. Each quadrant
is rotated 90 degrees counterclockwise for the existing physical mounting.
The viewfinder displays camera 1. The physical shutter uses BCM GPIO 5 and the
flash uses BCM GPIO 26.

The maximum composite is **4656x3496**, with **2328x1748 per view**. Full 16MP
per camera requires sequential single-channel capture, which this application
does not implement. Arducam also documents a one-frame offset between camera
pair 0/1 and pair 2/3 in quad mode; check moving subjects and flash coverage on
the rig. See [Arducam's quad-camera specifications](https://www.arducam.com/arducam-16mp-imx519-autofocus-synchronized-quad-camera-kit-for-raspberry-pi-nvidia-jetson.html).

## Set up the Pi 4B

1. Install an IMX519-compatible kernel/driver and libcamera/rpicam stack using
   [Arducam's IMX519 CamArray guide](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#imx519-multi-camera-kit).
   Use its instructions for your OS version. Python dependencies alone cannot
   enable the camera driver.
2. Edit `/boot/firmware/config.txt` on Bookworm (or `/boot/config.txt` on
   Bullseye). Disable camera auto-detection and replace the old camera sensor
   overlay with the IMX519 overlay under `[all]`:

   ```ini
   [all]
   camera_auto_detect=0
   dtoverlay=imx519
   ```

   Reboot after changing the boot configuration. The IMX519 overlay setup is
   described in [Arducam's sensor guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/).
3. Stop any running WiggleGram service so the camera is free, then check it:

   ```sh
   rpicam-still --list-cameras
   rpicam-still --camera 0 --nopreview --width 4656 --height 3496 \
       --autofocus-mode auto -t 2000 -o quad-test.jpg
   ```

   Expect one IMX519 device and four distinct views in `quad-test.jpg`. On
   Bullseye, use `libcamera-still` if `rpicam-still` is unavailable. The app
   automatically selects the installed command family. Verify the HAT is in
   four-in-one mode using the CamArray guide if the image shows one or two views.

## Install and run

From the repository root on the Pi:

```sh
sudo apt install python3-venv python3-tk python3-rpi.gpio
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python src/app.py
```

Run from the Pi's graphical session. Captures go to `camera/input/` and GIFs
to `camera/output/` by default. Set `WIGGLEGRAM_MEDIA_DIR` in `.env` to retain
a different existing media location.

## Focus and capture settings

Edit `.env` and restart the app. Preview defaults to 1164x874 at a requested
15 FPS; actual FPS depends on the selected sensor mode and HAT. Still capture,
preview, and calibration share the settings in `src/camera_settings.py`.

By default the preview uses continuous autofocus; stills focus once at startup
before the capture signal. Increase `WIGGLEGRAM_SETTLE_MS` if focus or white
balance needs longer to settle. For consistent focus across shots, set
`WIGGLEGRAM_FOCUS_MODE=manual` and `WIGGLEGRAM_LENS_POSITION` (dioptres; `0` is
infinity, `1` is approximately 1 metre). The HAT applies controls to all modules
together. Focus behavior follows the [camera command documentation](https://www.raspberrypi.com/documentation/accessories/camera-software.html#autofocus-mode).

The existing approximately 0.1-second exposure and 0.30-second flash offset are
retained and configurable. Flash timing needs testing with the new sensors;
software signal timing does not guarantee that all rolling shutters see the
flash. No autofocus sweep is requested at capture time, to avoid adding another
variable delay.

Leave `WIGGLEGRAM_TUNING_FILE` unset for automatic IMX519 tuning selection. Do
not reuse the old IMX219 NoIR tuning file.

## Recalibrate the replacement modules

Profiles now live in `src/calibrate/out/imx519/`, so old sensor profiles are
not applied by default. Stop the app before running calibration captures:

```sh
# Fill all four views with evenly lit white paper.
.venv/bin/python src/calibrate/wb_calibrate.py

# Use a straight reference edge to correct individual camera tilt.
.venv/bin/python src/calibrate/rotation_calibrate.py
```

Both commands also accept an existing composite JPEG as a positional argument.
They save directly to the paths consumed by the pipeline. White balance and
rotation corrections take effect on the next shot. ColorChecker calibration
(`src/calibrate/color_correct.py`) uses the same camera settings and profile
directory; color correction remains optional in `src/pipeline.py`.

## Development checks

With Pillow, NumPy, OpenCV, python-dotenv, Matplotlib, and Tk installed:

```sh
python -X utf8 -m unittest discover -s tests -v
python -m compileall -q src tests
```

Tests use simulated camera processes and synthetic images; camera detection,
physical focus, HAT timing, and flash exposure must be checked on the Pi.
