import os, io, signal, threading, time, logging, queue, subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk, ImageFile

import cv2
import numpy as np

from flash import trigger as flash_trigger
import config as cfg

# Logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Pipeline constants
REPO_ROOT          = Path(__file__).resolve().parents[1]

AWB_SETTLE_MS    = 1000
AWB_SETTLE_S     = AWB_SETTLE_MS / 1000.0
SHUTTER_US       = 16667 * 6       # ≈01/60 s
CAPTURE_SETTLE_S = 2.0               # wait after SIGUSR1 for rpicam-still to expose & write
FLASH_OFFSET_S   = 0.30               # flash delay relative to SIGINT (seconds)

AxisType = Literal["x", "xy"]

# Pipeline functions
def split_grid(img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    width, height = img.size
    pw, ph = width // cols, height // rows
    pieces = []
    for r in range(rows):
        for c in range(cols):
            pieces.append(img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)))
    return pieces


def apply_rotation_calibration(images: List[Image.Image]) -> List[Image.Image]:
    """Rotate each camera image by the CCW angle stored in rotation_calibration.npz."""
    
    calib_path = cfg.ROTATION_CALIBRATION_PATH
    
    if not calib_path.exists():
        logger.debug("Rotation calibration not found: %s (skipping)", calib_path)
        return images
    data   = np.load(str(calib_path))
    angles = data["angles"].astype(float)
    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(angles) or abs(angles[idx]) < 1e-4:
            corrected.append(im)
        else:
            corrected.append(im.rotate(
                float(angles[idx]),
                resample=Image.Resampling.BICUBIC,
                expand=False,
            ))
            logger.info("Rotation calibration applied to camera %d: %.4f°", idx, angles[idx])
    return corrected


def apply_color_correction(images: List[Image.Image]) -> List[Image.Image]:
    profile_path = cfg.COLOR_PROFILE_PATH
    if not profile_path.exists():
        logger.warning("Color profile not found: %s (skipping)", profile_path)
        return images
    data = np.load(str(profile_path))
    ccms = data["ccms"]   # (4, 3, 3) float32

    def srgb_to_linear(x):
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(ccms):
            corrected.append(im)
            continue
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        lin = srgb_to_linear(arr)
        flat_cc = np.clip(lin.reshape(-1, 3) @ ccms[idx].T, 0.0, 1.0)
        out = np.clip(linear_to_srgb(flat_cc.reshape(arr.shape)) * 255.0, 0, 255).astype(np.uint8)
        corrected.append(Image.fromarray(out))
        logger.info("Colour correction applied to camera %d", idx)
    return corrected


def apply_white_balance(images: List[Image.Image]) -> List[Image.Image]:
    profile_path = cfg.WB_PROFILE_PATH
    if not profile_path.exists():
        logger.warning("White balance profile not found: %s (skipping)", profile_path)
        return images
    data = np.load(str(profile_path))
    if "gain_maps" not in data:
        logger.warning("White balance profile missing 'gain_maps' — re-run wb_calibrate.py (skipping)")
        return images
    gain_maps = data["gain_maps"]  # (4, ds_h, ds_w, 3)

    def srgb_to_linear(x):
        x = np.clip(x / 255.0, 0.0, 1.0)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

    corrected: List[Image.Image] = []
    for idx, im in enumerate(images):
        if idx >= len(gain_maps):
            corrected.append(im)
            continue
        arr = np.asarray(im.convert("RGB")).astype(np.float32)
        h, w = arr.shape[:2]
        gm  = cv2.resize(gain_maps[idx], (w, h), interpolation=cv2.INTER_LINEAR)
        lin = srgb_to_linear(arr)
        out = np.clip(linear_to_srgb(np.clip(lin * gm, 0.0, 1.0)) * 255.0, 0, 255).astype(np.uint8)
        corrected.append(Image.fromarray(out))
        logger.info("White balance applied to camera %d", idx)
    return corrected


def pick_anchors_template(
    images: List[Image.Image],
    *,
    target_xy: Tuple[int, int],
    template_fraction: float = 0.25,
    min_score: float = 0.3,
    ref_idx: int = 1,
) -> List[Optional[Tuple[int, int]]]:
    """Align frames by template-matching the centre patch of the reference frame."""
    if not images:
        return []

    def to_gray(pil_im):
        return cv2.cvtColor(np.asarray(pil_im.convert("RGB")), cv2.COLOR_RGB2GRAY)

    tx, ty = target_xy
    ref_gray = to_gray(images[ref_idx])
    h_ref, w_ref = ref_gray.shape
    half_tw = int(w_ref * template_fraction / 2)
    half_th = int(h_ref * template_fraction / 2)
    tmpl = ref_gray[ty - half_th:ty + half_th, tx - half_tw:tx + half_tw]
    logger.info("Template: %dx%d patch from frame %d", tmpl.shape[1], tmpl.shape[0], ref_idx)

    anchors: List[Optional[Tuple[int, int]]] = []
    for idx in range(len(images)):
        if idx == ref_idx:
            anchors.append((tx, ty))
            continue
        gray = to_gray(images[idx])
        result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < min_score:
            logger.warning("Template match %.3f too low for frame %d (need %.2f), no shift",
                           max_val, idx, min_score)
            anchors.append(None)
            continue
        found_cx = max_loc[0] + half_tw
        found_cy = max_loc[1] + half_th
        anchors.append((found_cx, found_cy))
        logger.info("Frame %d match=%.3f anchor=(%d,%d) shift=(%d,%d)",
                    idx, max_val, found_cx, found_cy, tx - found_cx, ty - found_cy)
    return anchors


def save_anchor_debug(
    images: List[Image.Image],
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    cross_size: int = 40,
    thickness: int = 6,
):
    os.makedirs(save_dir, exist_ok=True)
    for idx, (im, anchor) in enumerate(zip(images, anchors)):
        arr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR).copy()
        if anchor is not None:
            ax, ay = anchor
            s = cross_size
            cv2.line(arr, (ax - s, ay - s), (ax + s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr, (ax + s, ay - s), (ax - s, ay + s), (255, 255, 255), thickness + 4)
            cv2.line(arr, (ax - s, ay - s), (ax + s, ay + s), (0, 0, 255), thickness)
            cv2.line(arr, (ax + s, ay - s), (ax - s, ay + s), (0, 0, 255), thickness)
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(
            os.path.join(save_dir, f"debug_anchor_cam{idx}.jpg"), quality=90
        )
    logger.info("Saved anchor debug images to %s", save_dir)


def translate_image(im: Image.Image, shift_xy: Tuple[int, int]) -> Image.Image:
    dx, dy = shift_xy
    w, h = im.size
    return im.transform(
        (w, h), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR, fillcolor=None,
    )


def crop_to_common_valid_area(
    frames: List[Image.Image], shifts: List[Tuple[int, int]]
) -> List[Image.Image]:
    if not frames:
        return frames
    w, h = frames[0].size
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]
    x0 = max(0, min(max(0, max(dxs)), w - 1))
    x1 = max(x0 + 1, min(w - max(0, -min(dxs)), w))
    y0 = max(0, min(max(0, max(dys)), h - 1))
    y1 = max(y0 + 1, min(h - max(0, -min(dys)), h))
    return [fr.crop((x0, y0, x1, y1)) for fr in frames]


def wigglegram_anchors(
    images: List[Image.Image],
    *,
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    filename:str,
    target_xy: Optional[Tuple[int, int]] = None,
    axis: AxisType = "x",
    duration_ms: int = 100,
    crop_common: bool = True,
):
    if not images:
        raise ValueError("No images provided")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{filename}.gif")

    base_w, base_h = images[0].size
    if any(im.size != (base_w, base_h) for im in images):
        images = [im.resize((base_w, base_h), Image.Resampling.BILINEAR) for im in images]

    tx, ty = target_xy if target_xy else (base_w // 2, base_h // 2)
    shifts: List[Tuple[int, int]] = []
    for anchor in anchors:
        if anchor is None:
            shifts.append((0, 0))
        else:
            ax, ay = anchor
            shifts.append((
                int(round(tx - ax)),
                0 if axis == "x" else int(round(ty - ay)),
            ))

    shifted = [translate_image(images[i], shifts[i]) for i in range(len(images))]
    if crop_common:
        shifted = crop_to_common_valid_area(shifted, shifts)

    snake = shifted + shifted[-2:0:-1]
    pal0 = snake[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    out_frames = [pal0] + [fr.quantize(palette=pal0) for fr in snake[1:]]
    out_frames[0].save(
        out_path, save_all=True, append_images=out_frames[1:],
        duration=duration_ms, loop=0, optimize=False, disposal=2, format="GIF",
    )
    logger.info("Saved wiggle GIF to %s (axis=%s)", out_path, axis)


def run_pipeline(raw_path: str, save_path: str, timestamp: str):
    """Full wigglegram processing pipeline."""
    raw    = Image.open(raw_path)
    pieces = split_grid(raw, 2, 2)
    pieces = [p.rotate(90, expand=True) for p in pieces]
    # pieces = apply_rotation_calibration(pieces)
    # pieces = apply_white_balance(pieces)
    pieces = apply_color_correction(pieces)

    w, h      = pieces[0].size
    target_xy = (w // 2, h // 2)
    anchors   = pick_anchors_template(pieces, target_xy=target_xy, ref_idx=1)
    # save_anchor_debug(pieces, anchors, save_path)
    wigglegram_anchors(
        pieces, anchors=anchors, save_dir=save_path, filename=timestamp,
        target_xy=target_xy, axis="x", duration_ms=100, crop_common=True,
    )


# Config
SHUTTER_PIN    = 5
CAMERA_INDEX   = 0
VF_WIDTH       = 2028
VF_HEIGHT      = 1520
VF_FPS         = 15
CROSSHAIR_LEN  = 30
CROSSHAIR_GAP  = 6
CROSSHAIR_CLR  = (255, 255, 255)
CROSSHAIR_SHD  = (0, 0, 0)

# GPIO setup
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SHUTTER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    _GPIO_AVAILABLE = True
    logger.info("GPIO ready on BCM%d", SHUTTER_PIN)
except ImportError:
    _GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available – keyboard spacebar will act as shutter")


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG viewfinder reader
# ─────────────────────────────────────────────────────────────────────────────

class MJPEGReader(threading.Thread):
    """Continuously reads MJPEG frames from rpicam-vid stdout into a queue."""

    # JPEG SOI / EOI markers
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, frame_queue: queue.Queue, camera: int = 0,
                 width: int = VF_WIDTH, height: int = VF_HEIGHT, fps: int = VF_FPS):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.camera      = camera
        self.width       = width
        self.height      = height
        self.fps         = fps
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()

    def run(self):
        cmd = [
            "rpicam-vid",
            "--camera", str(self.camera),
            "-t", "0",
            "--codec", "mjpeg",
            "-o", "-",
            "--width",  str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--nopreview",
        ]
        logger.info("Starting viewfinder: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        buf = b""
        while not self._stop_event.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk

            # Extract all complete JPEG frames from buffer
            while True:
                soi = buf.find(self._SOI)
                if soi == -1:
                    buf = b""
                    break
                eoi = buf.find(self._EOI, soi + 2)
                if eoi == -1:
                    buf = buf[soi:]   # keep partial frame
                    break
                jpeg_bytes = buf[soi : eoi + 2]
                buf = buf[eoi + 2:]
                try:
                    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    # Drop old frames, keep only latest
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(img)
                except Exception:
                    pass  # corrupt frame, skip

    def stop(self):
        self._stop_event.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.info("Viewfinder stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture
# ─────────────────────────────────────────────────────────────────────────────

def do_capture() -> Tuple[Optional[Path], Optional[Path]]:
    """Run rpicam-still + flash and return (input_file, output_dir) on success."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = cfg.INPUT_DIR / f"photo_{timestamp}.jpg"
    output_dir = cfg.OUTPUT_DIR
    input_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen([
            "rpicam-still",
            "-o", str(input_file),
            "--nopreview",
            "--quality", "100",
            "--awb", "auto",
            "--denoise", "cdn_hq",
            "--sharpness", "1.5",
            "--ev", "0.5",
            "-t", "0",
            "--signal",
            "--shutter", str(SHUTTER_US),
        ])

        time.sleep(AWB_SETTLE_S)
        # Queue a capture request, then wait for rpicam-still to expose and write
        os.kill(proc.pid, signal.SIGUSR1)
        
        # Fire flash after FLASH_OFFSET_S, send SIGINT immediately — both on separate threads
        def delayed_flash():
            time.sleep(FLASH_OFFSET_S)
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()
            time.sleep(0.001)
            flash_trigger()

        flash_thread = threading.Thread(target=delayed_flash, daemon=True)
        flash_thread.start()
        time.sleep(CAPTURE_SETTLE_S)
        proc.send_signal(signal.SIGINT)
        flash_thread.join()
        proc.wait()

        # ensure the captured file shows up before returning
        for _ in range(20):
            if input_file.exists():
                break
            time.sleep(0.1)
        if not input_file.exists():
            logger.error("Capture succeeded but file missing: %s", input_file)
            return None, None

        logger.info("Captured: %s", input_file)
        return input_file, output_dir

    except Exception as exc:
        logger.error("Capture failed: %s", exc)
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# GIF loader helper
# ─────────────────────────────────────────────────────────────────────────────

def load_gif_frames(path: Path) -> Tuple[List[ImageTk.PhotoImage], List[int]]:
    """Return (list_of_PhotoImages, list_of_durations_ms) for all GIF frames."""
    gif = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frame = gif.copy().convert("RGB")
            frames.append(ImageTk.PhotoImage(frame))
            durations.append(gif.info.get("duration", 100))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames, durations


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class WiggleApp:

    STATE_VIEWFINDER = "viewfinder"
    STATE_CAPTURING  = "capturing"
    STATE_PROCESSING = "processing"
    STATE_GIF        = "gif"

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.state = self.STATE_VIEWFINDER

        # ── Window setup ──────────────────────────────────────────────────────
        root.title("Wigglegram")
        root.configure(bg="black")
        root.attributes("-fullscreen", True)
        root.update_idletasks()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        # ── Canvas ────────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            root, width=self.screen_w, height=self.screen_h,
            bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # ── Bottom bar: status label + progress bar ─────────────────────────
        PROGRESS_H = 48
        self._bar_frame = tk.Frame(root, bg="black", height=PROGRESS_H)
        self._bar_frame.place(
            x=0, y=self.screen_h - PROGRESS_H,
            width=self.screen_w, height=PROGRESS_H
        )
        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = tk.Label(
            self._bar_frame, textvariable=self.status_var,
            font=("Helvetica", 16), fg="white", bg="black"
        )
        self.status_lbl.pack(side="top", pady=(4, 0))

        style = ttk.Style(root)
        style.theme_use("default")
        style.configure(
            "Wiggle.Horizontal.TProgressbar",
            troughcolor="#333333", background="#ffffff",
            thickness=10,
        )
        self._progress = ttk.Progressbar(
            self._bar_frame, style="Wiggle.Horizontal.TProgressbar",
            orient="horizontal", mode="indeterminate",
            length=self.screen_w - 40,
        )
        # hidden until processing starts
        self._progress.pack(side="top", padx=20, pady=(2, 4))
        self._progress.pack_forget()

        # Image item on canvas
        self._canvas_img_id = self.canvas.create_image(
            self.screen_w // 2, self.screen_h // 2, anchor="center"
        )
        self._current_tk_img: Optional[ImageTk.PhotoImage] = None

        # ── Viewfinder ────────────────────────────────────────────────────────
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._mjpeg = MJPEGReader(self._frame_queue)
        self._mjpeg.start()
        self._last_vf_frame: Optional[Image.Image] = None

        # ── GIF playback state ────────────────────────────────────────────────
        self._gif_frames:    List[ImageTk.PhotoImage] = []
        self._gif_durations: List[int] = []
        self._gif_idx:       int = 0
        self._gif_after_id:  Optional[str] = None

        # ── GPIO interrupt ────────────────────────────────────────────────────
        if _GPIO_AVAILABLE:
            GPIO.add_event_detect(
                SHUTTER_PIN, GPIO.FALLING,
                callback=self._gpio_shutter_cb,
                bouncetime=500,
            )
        else:
            # Fallback: spacebar triggers shutter
            root.bind("<space>", lambda _e: self._shutter_pressed())

        root.bind("<Escape>", lambda _e: self._quit())

        # ── Start viewfinder poll ─────────────────────────────────────────────
        self._poll_viewfinder()

    # ── GPIO callback (runs in GPIO thread – must be thread-safe) ─────────────
    def _gpio_shutter_cb(self, channel):
        self.root.after(0, self._shutter_pressed)

    # ── Shutter logic ─────────────────────────────────────────────────────────
    def _shutter_pressed(self):
        if self.state == self.STATE_VIEWFINDER:
            self._start_capture()
        elif self.state == self.STATE_GIF:
            self._return_to_viewfinder()

    # ── Extract camera 1 (top-right quadrant, rotated 90°) ───────────────────
    @staticmethod
    def _extract_cam1(frame: Image.Image) -> Image.Image:
        """Crop the top-right quadrant of the 2×2 grid and rotate 90° CW."""
        w, h = frame.size
        # top-right: col=1, row=0
        left  = w // 2
        upper = 0
        right = w
        lower = h // 2
        return frame.crop((left, upper, right, lower)).rotate(90, expand=True)

    # ── Viewfinder poll (runs every 33 ms on tkinter main thread) ─────────────
    def _poll_viewfinder(self):
        if self.state == self.STATE_VIEWFINDER:
            try:
                frame = self._frame_queue.get_nowait()
                cam1  = self._extract_cam1(frame)
                self._last_vf_frame = cam1
                self._show_pil(self._draw_crosshair(cam1))
            except queue.Empty:
                pass  # no new frame yet
        self.root.after(33, self._poll_viewfinder)

    # ── Draw a crosshair with gap in the centre ────────────────────────────────
    def _draw_crosshair(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img).copy()
        cx, cy = arr.shape[1] // 2, arr.shape[0] // 2
        g = CROSSHAIR_GAP
        L = CROSSHAIR_LEN
        t = 2  # thickness

        def hline(y, x0, x1):
            arr[y - t:y + t, x0:x1] = CROSSHAIR_SHD
            arr[y - t + 1:y + t - 1, x0 + 1:x1 - 1] = CROSSHAIR_CLR

        def vline(x, y0, y1):
            arr[y0:y1, x - t:x + t] = CROSSHAIR_SHD
            arr[y0 + 1:y1 - 1, x - t + 1:x + t - 1] = CROSSHAIR_CLR

        hline(cy, cx - L, cx - g)   # left arm
        hline(cy, cx + g, cx + L)   # right arm
        vline(cx, cy - L, cy - g)   # top arm
        vline(cx, cy + g, cy + L)   # bottom arm
        return Image.fromarray(arr)

    # ── Scale a PIL image to fit the screen and push to canvas ────────────────
    def _show_pil(self, img: Image.Image):
        sw, sh = self.screen_w, self.screen_h
        img_w, img_h = img.size
        scale = min(sw / img_w, sh / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._current_tk_img = tk_img   # prevent GC

    # ── Start capture sequence ────────────────────────────────────────────────
    def _start_capture(self):
        self.state = self.STATE_CAPTURING
        self.status_var.set("Capturing…")

        # Stop the MJPEG viewfinder to free the camera
        self._mjpeg.stop()
        # Show last frozen frame while camera re-initialises
        if self._last_vf_frame:
            self._show_pil(self._draw_crosshair(self._last_vf_frame))

        def capture_thread():
            input_file, output_dir = do_capture()
            if input_file is None:
                self.root.after(0, self._return_to_viewfinder)
                return
            self.root.after(0, lambda: self._start_processing(input_file, output_dir))

        threading.Thread(target=capture_thread, daemon=True).start()

    # ── Build 4-up composite from raw 2×2 capture ──────────────────────────
    @staticmethod
    def _build_4up(raw: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Split the 2x2 grid, rotate each 90° CW, lay them out side-by-side."""
        iw, ih = raw.size
        pw, ph = iw // 2, ih // 2
        # split_grid order: row-major → indices 0=TL,1=TR,2=BL,3=BR
        quads = [
            raw.crop((0,   0,   pw,  ph)),   # cam 0 – top-left
            raw.crop((pw,  0,   iw,  ph)),   # cam 1 – top-right
            raw.crop((0,   ph,  pw,  ih)),   # cam 2 – bottom-left
            raw.crop((pw,  ph,  iw,  ih)),   # cam 3 – bottom-right
        ]
        rotated = [q.rotate(90, expand=True) for q in quads]
        # Each rotated piece is ph×pw; lay 4 side by side
        cell_w = target_w // 4
        cell_h = target_h
        composite = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        for i, piece in enumerate(rotated):
            piece.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            pw2, ph2 = piece.size
            x = i * cell_w + (cell_w - pw2) // 2
            y = (cell_h - ph2) // 2
            composite.paste(piece, (x, y))
        return composite

    # ── Start processing ──────────────────────────────────────────────────────
    def _start_processing(self, input_file: Path, output_dir: Path):
        self.state = self.STATE_PROCESSING
        self.status_var.set("Processing…")

        # Show all 4 cameras side by side
        try:
            raw = Image.open(str(input_file)).convert("RGB")
            # Reserve bottom bar space
            composite = self._build_4up(raw, self.screen_w, self.screen_h - 60)
            self._show_pil(composite)
        except Exception as exc:
            logger.warning("4-up preview failed: %s", exc)

        # Show and start the indeterminate progress bar
        self._progress.pack(side="top", padx=20, pady=(2, 4))
        self._progress.start(20)   # step every 20 ms

        def process_thread():
            try:
                        # Extract timestamp from input filename (e.g., photo_20260224_140322.jpg)
                timestamp = input_file.stem.replace("photo_", "")
                # call pipeline using the passed output dir directly
                run_pipeline(str(input_file), str(output_dir), timestamp)
                gif_path = output_dir / f"{timestamp}.gif"
                self.root.after(0, lambda: self._show_gif(gif_path))
            except Exception as exc:
                logger.error("Pipeline failed: %s", exc)
                self.root.after(0, self._return_to_viewfinder)

        threading.Thread(target=process_thread, daemon=True).start()

    # ── GIF playback ──────────────────────────────────────────────────────────
    def _show_gif(self, gif_path: Path):
        if not gif_path.exists():
            logger.error("GIF not found: %s", gif_path)
            self._return_to_viewfinder()
            return

        # Stop + hide progress bar
        self._progress.stop()
        self._progress.pack_forget()

        self.state = self.STATE_GIF
        self.status_var.set("Done! Press shutter for next shot.")

        frames, durations = load_gif_frames(gif_path)
        if not frames:
            self._return_to_viewfinder()
            return

        # Scale frames once to screen size
        gif_raw = Image.open(gif_path)
        gif_w, gif_h = gif_raw.size
        sw, sh = self.screen_w, self.screen_h
        scale = min(sw / gif_w, sh / gif_h)
        disp_w, disp_h = int(gif_w * scale), int(gif_h * scale)

        scaled_frames: List[ImageTk.PhotoImage] = []
        try:
            while True:
                frame = gif_raw.copy().convert("RGB").resize(
                    (disp_w, disp_h), Image.Resampling.LANCZOS
                )
                scaled_frames.append(ImageTk.PhotoImage(frame))
                durations_scaled = durations  # same list
                gif_raw.seek(gif_raw.tell() + 1)
        except EOFError:
            pass

        self._gif_frames    = scaled_frames
        self._gif_durations = durations
        self._gif_idx       = 0
        self._animate_gif()

    def _animate_gif(self):
        if self.state != self.STATE_GIF:
            return
        idx = self._gif_idx % len(self._gif_frames)
        tk_img = self._gif_frames[idx]
        self.canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._current_tk_img = tk_img
        self._gif_idx = (idx + 1) % len(self._gif_frames)
        dur = self._gif_durations[idx] if self._gif_durations else 100
        self._gif_after_id = self.root.after(dur, self._animate_gif)

    # ── Return to viewfinder ──────────────────────────────────────────────────
    def _return_to_viewfinder(self):
        # Cancel GIF animation
        if self._gif_after_id:
            self.root.after_cancel(self._gif_after_id)
            self._gif_after_id = None
        self._gif_frames = []

        # Ensure progress bar is stopped and hidden
        self._progress.stop()
        self._progress.pack_forget()

        self.state = self.STATE_VIEWFINDER
        self.status_var.set("Ready")

        # Restart MJPEG reader
        self._frame_queue = queue.Queue(maxsize=2)
        self._mjpeg = MJPEGReader(self._frame_queue)
        self._mjpeg.start()

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def _quit(self):
        self._mjpeg.stop()
        if _GPIO_AVAILABLE:
            GPIO.cleanup()
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = WiggleApp(root)
    root.mainloop()
