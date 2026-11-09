# WiggleGram

Raspberry Pi camera application for a four-view wigglegram rig. The current
hardware is a Raspberry Pi 4B, an Arducam UC-512 CamArray HAT, and four Arducam
UC-873 (Sony IMX519, 16MP) modules compatible with the CamArray kit.

The application expects the HAT's **four-in-one, 2x2 composite**: camera 0 at
top-left, 1 at top-right, 2 at bottom-left, and 3 at bottom-right. Every quadrant
retains its **original sensor orientation, with no rotation**. Capture
processing, preview, batch processing, and calibration use that same orientation.
The viewfinder fits the entire view from camera 1 onto
the screen. The physical shutter uses BCM GPIO 5 and the flash uses BCM GPIO 26.

The maximum composite is **4656x3496**, with **2328x1748 per view**. Full 16MP
per camera requires sequential single-channel capture, which this application
does not implement. Arducam also documents a one-frame offset between camera
pair 0/1 and pair 2/3 in quad mode; check moving subjects and flash coverage on
the rig. See [Arducam's quad-camera specifications](https://www.arducam.com/arducam-16mp-imx519-autofocus-synchronized-quad-camera-kit-for-raspberry-pi-nvidia-jetson.html).

Captures retain the full native composite. Each camera contributes its complete
**2328x1748** unrotated view. The **16:9 GIF adds black borders instead of
cropping or enlarging that view**. With no alignment shift, the output canvas
is 3120x1755; all 2328x1748 camera pixels remain at their original scale.
Alignment expands the canvas to retain the edges, so output dimensions can grow.

Stills explicitly request `--mode 4656:3496:10`. Preview requests the
`2328:1748:10` binned mode; both cover the full active sensor area in the
[IMX519 driver](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/drivers/media/i2c/imx519.c).
Both commands use `--roi 0,0,1,1`. Cropped 16:9 sensor modes are not selected.
The preview stream is scaled down for display, while still captures retain
the maximum combined resolution. Preview text and crosshair are drawn after
display scaling, so changing stream resolution does not enlarge the UI.

## Set up the Pi 4B

Close the app and stop any running WiggleGram service, then run from the
repository root **on the Pi**:

```sh
sudo python3 scripts/setup_imx519.py
sudo reboot
```

The script follows [Arducam's IMX519 CamArray guide](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/#imx519-multi-camera-kit):
it downloads Arducam's installer and runs `-p libcamera`, then
`-p libcamera_apps`. It requires internet access and Raspberry Pi OS Bullseye,
Bookworm, or Trixie with kernel 6.1.21 or newer. Trixie support follows
[Arducam's current IMX519 instructions](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/).
The setup script uses the system Python and needs no pip packages.

It selects `/boot/firmware/config.txt` on Bookworm/Trixie or `/boot/config.txt`
on Bullseye, backs up that file and its includes as `.wigglegram-*.bak`, disables
old camera overlays, and applies this block under `[all]`:

```ini
camera_auto_detect=0
dtoverlay=imx519
```

Display and GPIO settings are preserved. Repeated runs keep one active IMX519
block; package installation can reinstall the vendor packages. Backup filenames
are printed before installation. The script leaves rebooting to you.

To preview the package plan and exact config diff, or apply only the config
change when the camera stack is already installed:

```sh
python3 scripts/setup_imx519.py --dry-run
sudo python3 scripts/setup_imx519.py --config-only
```

After reboot, check detection and take a test image:

```sh
python3 scripts/setup_imx519.py --check
rpicam-still --camera 0 --nopreview --mode 4656:3496:10 --roi 0,0,1,1 \
    --width 4656 --height 3496 \
    --autofocus-mode auto -t 2000 -o quad-test.jpg
```

Expect one IMX519 device and four distinct views in `quad-test.jpg`. On Bullseye,
use `libcamera-still` if `rpicam-still` is unavailable. The app automatically
selects the installed command family. Verify the HAT is in four-in-one mode
using the CamArray guide if the image shows one or two views. A missing
`imx519.dtbo` overlay means the Pi's kernel/firmware must be updated before this
setup can complete; the script reports that instead of claiming success.

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

All automatic view rotation is disabled, including saved rotation-calibration
corrections. Old `WIGGLEGRAM_VIEW_ROTATION` values in existing `.env` files are
ignored. The startup log confirms that the app retains native sensor orientation.
Unrotated spatial calibration profiles use `src/calibrate/out/imx519/rotation_0/`
so previous 90-degree gain maps are not reused. Recalibrate white balance for
this orientation, or set `WIGGLEGRAM_CALIBRATION_DIR` to matching profiles.

Keep these values for maximum still resolution and an uncropped preview:

```ini
WIGGLEGRAM_CAPTURE_WIDTH=4656
WIGGLEGRAM_CAPTURE_HEIGHT=3496
WIGGLEGRAM_VF_WIDTH=1164
WIGGLEGRAM_VF_HEIGHT=874
```

Composite dimensions must keep the sensor's native aspect ratio. Set the
output's 16:9 shape by adding borders after the views are split, rather than
by requesting a 16:9 composite or cropping an upright camera view.

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

Profiles now live in `src/calibrate/out/imx519/rotation_0/`, so old sensor profiles are
not applied by default. Stop the app before running calibration captures:

```sh
# Fill all four views with evenly lit white paper.
.venv/bin/python src/calibrate/wb_calibrate.py
```

The command also accepts an existing composite JPEG as a positional argument
and saves directly to the path consumed by the pipeline. White balance
corrections take effect on the next shot. Saved rotation profiles are ignored.
ColorChecker calibration
(`src/calibrate/color_correct.py`) uses the same camera settings and profile
directory; color correction remains optional in `src/pipeline.py`.

## Development checks

With Pillow, NumPy, OpenCV, python-dotenv, Matplotlib, and Tk installed:

```sh
python -X utf8 -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

Tests use simulated camera processes and synthetic images; camera detection,
physical focus, HAT timing, and flash exposure must be checked on the Pi.
