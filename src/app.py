import logging
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk, ImageFile, ImageDraw, ImageFont

import config as cfg
from camera import do_capture, SHUTTER_US
from gif_utils import load_gif_frames
from pipeline import run_pipeline
from viewfinder import MJPEGReader

# Logging configuration - write to both console and file
LOG_FILE = Path(__file__).parent.parent / "wigglegram.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# UI Configuration
VF_WIDTH       = 2028
VF_HEIGHT      = 1520
VF_FPS         = 15
CROSSHAIR_LEN  = 30
CROSSHAIR_GAP  = 6
CROSSHAIR_CLR  = (255, 255, 255)
CROSSHAIR_SHD  = (0, 0, 0)

# GPIO setup
SHUTTER_PIN = 5
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SHUTTER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    _GPIO_AVAILABLE = True
    logger.info("GPIO ready on BCM%d", SHUTTER_PIN)
except ImportError:
    _GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available - keyboard spacebar will act as shutter")


class WiggleApp:
    STATE_VIEWFINDER = "viewfinder"
    STATE_CAPTURING  = "capturing"
    STATE_PROCESSING = "processing"
    STATE_GIF        = "gif"

    def __init__(self, root: tk.Tk):
        self.root  = root
        self.state = self.STATE_VIEWFINDER

        # Window setup
        root.title("Wigglegram")
        root.configure(bg="black")
        root.attributes("-fullscreen", True)
        root.update_idletasks()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        # Canvas
        self.canvas = tk.Canvas(
            root, width=self.screen_w, height=self.screen_h,
            bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Bottom bar: status label + progress bar
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
        
        # Hide status bar initially (viewfinder mode)
        self._bar_frame.place_forget()

        # Image item on canvas
        self._canvas_img_id = self.canvas.create_image(
            self.screen_w // 2, self.screen_h // 2, anchor="center"
        )
        self._current_tk_img: Optional[ImageTk.PhotoImage] = None

        # Viewfinder
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._mjpeg = MJPEGReader(self._frame_queue, width=VF_WIDTH, height=VF_HEIGHT, fps=VF_FPS)
        self._mjpeg.start()
        self._last_vf_frame: Optional[Image.Image] = None

        # GIF playback state
        self._gif_frames:    List[ImageTk.PhotoImage] = []
        self._gif_durations: List[int] = []
        self._gif_idx:       int = 0
        self._gif_after_id:  Optional[str] = None

        # GPIO interrupt
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

        # Start viewfinder poll
        self._poll_viewfinder()

    # GPIO callback (runs in GPIO thread – must be thread-safe)
    def _gpio_shutter_cb(self, channel):
        self.root.after(0, self._shutter_pressed)

    # Shutter logic
    def _shutter_pressed(self):
        if self.state == self.STATE_VIEWFINDER:
            self._start_capture()
        elif self.state == self.STATE_GIF:
            self._return_to_viewfinder()

    # Extract camera 1 (top-right quadrant, rotated 90°)
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

    # Viewfinder poll (runs every 33 ms on tkinter main thread)
    def _poll_viewfinder(self):
        if self.state == self.STATE_VIEWFINDER:
            try:
                frame = self._frame_queue.get_nowait()
                cam1  = self._extract_cam1(frame)
                self._last_vf_frame = cam1
                cam1 = self._draw_crosshair(cam1)
                cam1 = self._draw_camera_info(cam1)
                self._show_pil(cam1)
            except queue.Empty:
                pass  # no new frame yet
        self.root.after(33, self._poll_viewfinder)

    # Draw a crosshair with gap in the center
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

    # Draw DSLR-style camera info overlay
    def _draw_camera_info(self, img: Image.Image) -> Image.Image:
        """Draw camera settings and stats overlay like a DSLR viewfinder."""
        # Ensure we have a mutable RGB image
        img = img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        logger.debug(f"Drawing camera info overlay on {w}x{h} image")
        
        # Try to load fonts with multiple fallback paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        font_large = None
        font_med = None
        font_small = None
        
        for font_path in font_paths:
            try:
                font_large = ImageFont.truetype(font_path, 40)
                font_med = ImageFont.truetype(font_path, 28)
                font_small = ImageFont.truetype(font_path, 24)
                logger.debug(f"Loaded font: {font_path}")
                break
            except:
                continue
        
        # Final fallback to default
        if font_large is None:
            font_large = ImageFont.load_default()
            font_med = ImageFont.load_default()
            font_small = ImageFont.load_default()
            logger.warning("Using default bitmap font - text may be small")
        
        # Helper to draw text with shadow
        def draw_text_shadow(xy, text, font, fill=(255, 255, 255)):
            x, y = xy
            # Shadow (thick black outline for visibility)
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
            # Main text
            draw.text((x, y), text, font=font, fill=fill)
        
        # Top-left: Mode (bright yellow/orange for visibility)
        draw_text_shadow((20, 20), "WIGGLEGRAM", font_large, fill=(255, 200, 0))
        
        # Top-right: Time and storage
        current_time = datetime.now().strftime("%H:%M:%S")
        try:
            disk = shutil.disk_usage(cfg.OUTPUT_DIR)
            gb_free = disk.free / (1024**3)
            storage_text = f"{gb_free:.1f}GB"
        except:
            storage_text = "--GB"
        
        time_text = f"{current_time}  {storage_text}"
        bbox = draw.textbbox((0, 0), time_text, font=font_med)
        text_w = bbox[2] - bbox[0]
        draw_text_shadow((w - text_w - 20, 20), time_text, font_med)
        
        # Bottom-left: Camera settings
        shutter_speed = SHUTTER_US / 1_000_000  # Convert to seconds
        if shutter_speed >= 1:
            shutter_text = f"{shutter_speed:.1f}s"
        else:
            shutter_text = f"1/{int(1/shutter_speed)}"
        
        settings_lines = [
            f"SS: {shutter_text}",
            f"EV: +0.5",
            f"AWB: Auto",
        ]
        
        y_offset = h - 20 - (len(settings_lines) * 28)
        for line in settings_lines:
            draw_text_shadow((20, y_offset), line, font_small)
            y_offset += 28
        
        # Bottom-right: Resolution and FPS
        info_lines = [
            f"{VF_WIDTH}×{VF_HEIGHT}",
            f"{VF_FPS} FPS",
        ]
        
        y_offset = h - 20 - (len(info_lines) * 28)
        for line in info_lines:
            bbox = draw.textbbox((0, 0), line, font=font_small)
            text_w = bbox[2] - bbox[0]
            draw_text_shadow((w - text_w - 20, y_offset), line, font_small)
            y_offset += 28
        
        return img

    # Scale a PIL image to fit the screen and push to canvas
    def _show_pil(self, img: Image.Image):
        sw, sh = self.screen_w, self.screen_h
        img_w, img_h = img.size
        scale = min(sw / img_w, sh / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._canvas_img_id, image=tk_img)
        self._current_tk_img = tk_img   # prevent GC

    # Start capture sequence
    def _start_capture(self):
        self.state = self.STATE_CAPTURING
        self.status_var.set("Capturing…")
        
        # Show status bar during capture
        PROGRESS_H = 48
        self._bar_frame.place(
            x=0, y=self.screen_h - PROGRESS_H,
            width=self.screen_w, height=PROGRESS_H
        )

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

    # Build 4-up composite from raw 2×2 capture
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

    # Start processing
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

    # GIF playback
    def _show_gif(self, gif_path: Path):
        if not gif_path.exists():
            logger.error("GIF not found: %s", gif_path)
            self._return_to_viewfinder()
            return

        # Stop + hide progress bar
        self._progress.stop()
        self._progress.pack_forget()

        self.state = self.STATE_GIF
        self.status_var.set("Done! Press shutter for next shot!")

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

    # Return to viewfinder
    def _return_to_viewfinder(self):
        # Cancel GIF animation
        if self._gif_after_id:
            self.root.after_cancel(self._gif_after_id)
            self._gif_after_id = None
        self._gif_frames = []

        # Ensure progress bar is stopped and hidden
        self._progress.stop()
        self._progress.pack_forget()
        
        # Hide status bar in viewfinder mode
        self._bar_frame.place_forget()

        self.state = self.STATE_VIEWFINDER
        self.status_var.set("Ready")

        # Restart MJPEG reader
        self._frame_queue = queue.Queue(maxsize=2)
        self._mjpeg = MJPEGReader(self._frame_queue, width=VF_WIDTH, height=VF_HEIGHT, fps=VF_FPS)
        self._mjpeg.start()

    # Clean shutdown
    def _quit(self):
        self._mjpeg.stop()
        if _GPIO_AVAILABLE:
            GPIO.cleanup()
        self.root.destroy()

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app  = WiggleApp(root)
    root.mainloop()
