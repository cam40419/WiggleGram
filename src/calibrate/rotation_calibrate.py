"""
rotation_calibrate.py – Interactive rotation calibration for the 4-camera wigglegram rig.

Usage:
    DISPLAY=:0 python3 src/rotation_calibrate.py [input_image.jpg]

Flow:
  1. A fullscreen prompt asks you to press the physical shutter button (or SPACE)
     while a reference rectangle (e.g. a sheet of paper) is in frame.
  2. rpicam-still captures the composite image.
  3. For each of the 4 camera quadrants (after the standard 90° pipeline rotation):
       – Click TWO points on the SAME horizontal edge of the reference rectangle.
       – The angle of that line vs. horizontal is computed and a corrected preview shown.
       – Confirm or redo the clicks.
  4. A summary of all 4 corrected images is shown; Save writes rotation_calibration.npz.

If you pass an existing image path the capture step is skipped.
"""

import sys
import io
import os
import math
import queue
import signal
import argparse
import subprocess
import threading
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageTk, ImageDraw

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
CALIB_OUT  = REPO_ROOT / "rotation_calibration.npz"
NUM_CAMS   = 4

# ── GPIO / shutter ─────────────────────────────────────────────────────────────
SHUTTER_PIN = 5          # BCM GPIO pin (active-LOW, pull-up) – same as prod.py
AWB_SETTLE_S = 2.0       # seconds to let AWB settle before capture
SHUTTER_US   = 16667     # 1/60 s fixed shutter (µs)

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SHUTTER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    _GPIO_AVAILABLE = True
except Exception:
    _GPIO_AVAILABLE = False


VF_WIDTH  = 2028
VF_HEIGHT = 1520
VF_FPS    = 15


# ── MJPEG viewfinder reader ────────────────────────────────────────────────────

class MJPEGReader(threading.Thread):
    """Streams MJPEG frames from rpicam-vid into a queue (daemon thread)."""
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, frame_queue: queue.Queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    def run(self):
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--codec", "mjpeg",
            "-o", "-",
            "--width",     str(VF_WIDTH),
            "--height",    str(VF_HEIGHT),
            "--framerate", str(VF_FPS),
            "--nopreview",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        buf = b""
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                soi = buf.find(self._SOI)
                if soi == -1:
                    buf = b""
                    break
                eoi = buf.find(self._EOI, soi + 2)
                if eoi == -1:
                    buf = buf[soi:]
                    break
                jpeg = buf[soi:eoi + 2]
                buf  = buf[eoi + 2:]
                try:
                    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(img)
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _extract_cam1(frame: Image.Image) -> Image.Image:
    """Crop top-right quadrant of the 2×2 composite and rotate 90° CW."""
    w, h = frame.size
    return frame.crop((w // 2, 0, w, h // 2)).rotate(90, expand=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_input_image(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input image not found: {arg}")
    candidates = sorted((REPO_ROOT / "input").glob("photo_*.jpg"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        "No photo_*.jpg found in input/. Provide a path as an argument."
    )


def split_grid(img: Image.Image, rows: int, cols: int):
    w, h = img.size
    pw, ph = w // cols, h // rows
    pieces = []
    for r in range(rows):
        for c in range(cols):
            pieces.append(img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)))
    return pieces


def load_camera_images(raw_path: Path):
    """Reproduce the first two pipeline steps: split 2×2 grid, rotate each 90° CW."""
    raw = Image.open(raw_path).convert("RGB")
    pieces = split_grid(raw, 2, 2)
    return [p.rotate(90, expand=True) for p in pieces]


def best_rotation_from_corners(pts: list) -> float:
    """Given 4 corner points of a rectangle (any order), return the CCW degrees
    to rotate the image so the rectangle sides become axis-aligned.

    Uses all 4 edges: horizontal edges contribute their raw tilt angle; vertical
    edges contribute (edge_angle − 90°).  A circular mean of all 4 tilt estimates
    gives a least-squares-style robust result.
    """
    # Sort into TL / TR / BL / BR by y then x
    by_y = sorted(pts, key=lambda p: p[1])
    tl, tr = sorted(by_y[:2], key=lambda p: p[0])
    bl, br = sorted(by_y[2:], key=lambda p: p[0])

    def edge_angle(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return math.degrees(math.atan2(dy, dx))

    # Horizontal edges: tilt = edge_angle (should be 0° when level)
    # Vertical edges:   tilt = edge_angle − 90° (should be 0° when level)
    tilts = [
        edge_angle(tl, tr),          # top edge
        edge_angle(bl, br),          # bottom edge
        edge_angle(tl, bl) - 90.0,   # left edge
        edge_angle(tr, br) - 90.0,   # right edge
    ]

    # Circular mean (handles wrap-around near ±180°)
    rads = [math.radians(t) for t in tilts]
    mean_angle = math.degrees(
        math.atan2(
            sum(math.sin(r) for r in rads) / 4,
            sum(math.cos(r) for r in rads) / 4,
        )
    )
    return mean_angle   # rotate CCW by this amount to level the rectangle


def apply_rotation(img: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate CCW by angle_deg, keeping the original canvas size (no expand)."""
    if abs(angle_deg) < 1e-4:
        return img
    return img.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=False)


# ── Per-camera calibration UI ──────────────────────────────────────────────────

class CameraCalibUI:
    """Blocking Tkinter session: click all 4 corners of the reference rectangle
    to compute a robust rotation angle using all 4 sides."""

    POINT_RADIUS = 8
    LINE_WIDTH   = 3

    # Labels shown sequentially as corners are clicked
    CORNER_PROMPTS = [
        "Click corner 1 of 4  (any corner of the rectangle).",
        "Click corner 2 of 4  (an adjacent or opposite corner).",
        "Click corner 3 of 4.",
        "Click corner 4 of 4  —  then review and confirm.",
    ]
    CORNER_COLORS = ["#ffdd00", "#ff9900", "#ff4444", "#dd00ff"]

    def __init__(self, cam_idx: int, image: Image.Image, prev_angle: float | None):
        self.cam_idx    = cam_idx
        self.image      = image
        self.prev_angle = prev_angle
        self.result_angle: float | None = None

        # ── Build window ───────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.configure(bg="#1a1a1a")
        self.root.title(f"Rotation Calibration — Camera {cam_idx} of {NUM_CAMS - 1}")
        self.root.attributes("-fullscreen", True)
        self.root.update_idletasks()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        bold14 = tkfont.Font(family="Helvetica", size=14, weight="bold")
        norm12 = tkfont.Font(family="Helvetica", size=12)

        # ── Instruction banner ─────────────────────────────────────────────────
        banner = (
            f"Camera {cam_idx}  ({cam_idx + 1} of {NUM_CAMS})  —  "
            "Click all 4 corners of your reference rectangle."
        )
        if prev_angle is not None:
            banner += f"   (Previous: {prev_angle:+.2f}°)"

        tk.Label(
            self.root, text=banner,
            font=bold14, fg="white", bg="#1a1a1a", pady=8,
        ).pack(side="top", fill="x")

        # ── Canvas ─────────────────────────────────────────────────────────────
        canvas_h = sh - 120
        canvas_w = sw

        self.canvas = tk.Canvas(
            self.root, width=canvas_w, height=canvas_h,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack(side="top")

        # Scale image to fill available square
        img_w, img_h   = image.size
        side           = min(canvas_w, canvas_h)
        scale          = side / max(img_w, img_h)
        self.disp_w    = int(img_w * scale)
        self.disp_h    = int(img_h * scale)
        self.offset_x  = (canvas_w - self.disp_w) // 2
        self.offset_y  = (canvas_h - self.disp_h) // 2
        self.scale     = scale

        self._base_img = image.resize((self.disp_w, self.disp_h), Image.Resampling.LANCZOS)
        self._photo    = ImageTk.PhotoImage(self._base_img)
        self._img_id   = self.canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=self._photo
        )
        self.canvas.bind("<Button-1>", self._on_click)

        # ── Status label ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value=self.CORNER_PROMPTS[0])
        tk.Label(
            self.root, textvariable=self._status_var,
            font=norm12, fg="#cccccc", bg="#1a1a1a", pady=4,
        ).pack(side="top")

        # ── Button row ─────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(side="top", pady=6)

        self._confirm_btn = tk.Button(
            btn_frame, text="✓  Confirm angle",
            font=bold14, fg="white", bg="#2a7a2a",
            state="disabled", padx=16, pady=6,
            command=self._confirm,
        )
        self._confirm_btn.pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="↺  Redo",
            font=bold14, fg="white", bg="#7a2a2a",
            padx=16, pady=6,
            command=self._redo,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame, text="Skip (0°)",
            font=norm12, fg="white", bg="#444444",
            padx=10, pady=6,
            command=self._skip,
        ).pack(side="left", padx=10)

        self.root.bind("<Escape>", lambda _: self._skip())

        # ── State ──────────────────────────────────────────────────────────────
        self._clicks: list[tuple[int, int]] = []   # canvas coords
        self._overlay_ids: list = []

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _canvas_to_image(self, cx: int, cy: int) -> tuple[float, float]:
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def _image_to_canvas(self, ix: float, iy: float) -> tuple[int, int]:
        return (int(ix * self.scale + self.offset_x),
                int(iy * self.scale + self.offset_y))

    # ── Overlay helpers ────────────────────────────────────────────────────────

    def _clear_overlay(self):
        for oid in self._overlay_ids:
            self.canvas.delete(oid)
        self._overlay_ids.clear()

    def _draw_point(self, cx: int, cy: int, color: str, label: str):
        r = self.POINT_RADIUS
        oid = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="white", fill=color, width=1,
        )
        self._overlay_ids.append(oid)
        oid2 = self.canvas.create_text(
            cx + r + 4, cy - r - 2,
            text=label, fill=color,
            font=("Helvetica", 11, "bold"), anchor="nw",
        )
        self._overlay_ids.append(oid2)

    def _draw_rectangle_and_preview(self):
        """Draw the fitted rectangle and corrected inset after all 4 clicks."""
        img_pts = [self._canvas_to_image(*c) for c in self._clicks]
        angle   = best_rotation_from_corners(img_pts)
        self._current_angle = angle

        # Sort corners into TL/TR/BR/BL order for drawing
        by_y = sorted(img_pts, key=lambda p: p[1])
        tl, tr = sorted(by_y[:2], key=lambda p: p[0])
        bl, br = sorted(by_y[2:], key=lambda p: p[0])

        # Draw rectangle outline
        corners_c = [self._image_to_canvas(*p) for p in [tl, tr, br, bl]]
        outline = self.canvas.create_polygon(
            *[v for pt in corners_c for v in pt],
            outline="#00ccff", fill="", width=self.LINE_WIDTH,
        )
        self._overlay_ids.append(outline)

        # Draw individual edge angles as small text
        edge_pairs = [(tl, tr, "top"), (bl, br, "bot"), (tl, bl, "left"), (tr, br, "right")]
        for (a, b, lbl) in edge_pairs:
            ca, cb = self._image_to_canvas(*a), self._image_to_canvas(*b)
            mx, my = (ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2
            ea = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            if "left" in lbl or "right" in lbl:
                ea -= 90.0
            eid = self.canvas.create_text(
                mx, my,
                text=f"{ea:+.1f}°",
                fill="#aaffaa", font=("Helvetica", 9), anchor="center",
            )
            self._overlay_ids.append(eid)

        # Corrected preview inset (bottom-left of image area)
        corrected = apply_rotation(self.image, angle)
        prev_w = self.disp_w // 3
        prev_h = self.disp_h // 3
        prev_img = corrected.resize((prev_w, prev_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(prev_img)
        draw.rectangle([(0, 0), (prev_w - 1, 22)], fill=(0, 0, 0))
        draw.text((4, 4), f"Corrected ({angle:+.2f}°)", fill="white")
        self._preview_photo = ImageTk.PhotoImage(prev_img)
        px = self.offset_x + self.disp_w - prev_w - 8
        py = self.offset_y + 8
        poid = self.canvas.create_image(px, py, anchor="nw", image=self._preview_photo)
        self._overlay_ids.append(poid)

        self._status_var.set(
            f"4 corners → angle = {angle:+.3f}°  "
            "(avg of all 4 sides).  Confirm to accept or Redo to re-click."
        )
        self._confirm_btn.config(state="normal")

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        if len(self._clicks) >= 4:
            return

        cx = max(self.offset_x, min(event.x, self.offset_x + self.disp_w))
        cy = max(self.offset_y, min(event.y, self.offset_y + self.disp_h))
        n  = len(self._clicks)

        self._clicks.append((cx, cy))
        self._draw_point(cx, cy, self.CORNER_COLORS[n], str(n + 1))

        if len(self._clicks) < 4:
            self._status_var.set(self.CORNER_PROMPTS[len(self._clicks)])
        else:
            self._draw_rectangle_and_preview()

    def _redo(self):
        self._clear_overlay()
        self._clicks.clear()
        self._confirm_btn.config(state="disabled")
        self._status_var.set(self.CORNER_PROMPTS[0])

    def _confirm(self):
        self.result_angle = getattr(self, "_current_angle", 0.0)
        self.root.destroy()

    def _skip(self):
        self.result_angle = 0.0
        self.root.destroy()

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> float:
        self.root.mainloop()
        return self.result_angle if self.result_angle is not None else 0.0


# ── Summary window ─────────────────────────────────────────────────────────────

class SummaryUI:
    """Shows all 4 corrected previews and saves the calibration file."""

    def __init__(self, images: list, angles: list[float]):
        self.images = images
        self.angles = angles

        self.root = tk.Tk()
        self.root.configure(bg="#1a1a1a")
        self.root.title("Rotation Calibration — Summary")
        self.root.attributes("-fullscreen", True)
        self.root.update_idletasks()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        bold14 = tkfont.Font(family="Helvetica", size=14, weight="bold")
        norm12 = tkfont.Font(family="Helvetica", size=12)

        tk.Label(
            self.root,
            text="Calibration complete — review corrected images, then Save or Quit.",
            font=bold14, fg="white", bg="#1a1a1a", pady=8,
        ).pack(side="top", fill="x")

        # 2×2 grid of corrected previews
        grid_frame = tk.Frame(self.root, bg="#1a1a1a")
        grid_frame.pack(side="top", expand=True, fill="both", padx=20, pady=10)

        cell_w = (sw - 60) // 2
        cell_h = (sh - 160) // 2

        for idx, (img, angle) in enumerate(zip(images, angles)):
            row, col = divmod(idx, 2)
            corrected = apply_rotation(img, angle)

            scale = min((cell_w - 20) / corrected.width, (cell_h - 40) / corrected.height)
            disp = corrected.resize(
                (int(corrected.width * scale), int(corrected.height * scale)),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(disp)
            label = f"Cam {idx}  {angle:+.2f}°"
            draw.rectangle([(0, 0), (disp.width - 1, 24)], fill=(0, 0, 0))
            draw.text((4, 4), label, fill="white")

            photo = ImageTk.PhotoImage(disp)

            frame = tk.Frame(grid_frame, bg="#333333", relief="solid", bd=1)
            frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            grid_frame.rowconfigure(row, weight=1)
            grid_frame.columnconfigure(col, weight=1)

            lbl = tk.Label(frame, image=photo, bg="black")
            lbl.image = photo   # prevent GC
            lbl.pack(fill="both", expand=True)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1a1a1a")
        btn_frame.pack(side="top", pady=8)

        tk.Button(
            btn_frame, text=f"💾  Save to {CALIB_OUT.name}",
            font=bold14, fg="white", bg="#2a7a2a",
            padx=20, pady=8,
            command=self._save,
        ).pack(side="left", padx=12)

        tk.Button(
            btn_frame, text="✗  Discard & Quit",
            font=bold14, fg="white", bg="#7a2a2a",
            padx=20, pady=8,
            command=self.root.destroy,
        ).pack(side="left", padx=12)

        self.root.bind("<Escape>", lambda _: self.root.destroy())

    def _save(self):
        np.savez(str(CALIB_OUT), angles=np.array(self.angles, dtype=np.float32))
        print(f"Saved rotation calibration → {CALIB_OUT}")
        print("  angles (CCW degrees per camera):", self.angles)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ── Capture screen with live viewfinder ───────────────────────────────────────

class CaptureUI:
    """Fullscreen live viewfinder; press shutter to freeze and fire rpicam-still."""

    OVERLAY_H = 48   # height of the top instruction bar (px)
    STATUS_H  = 36   # height of the bottom status bar (px)

    def __init__(self):
        self.result_path: Path | None = None
        self._triggered = False

        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.title("Rotation Calibration — Capture")
        self.root.attributes("-fullscreen", True)
        self.root.update_idletasks()

        self._sw = self.root.winfo_screenwidth()
        self._sh = self.root.winfo_screenheight()

        # ── Full-screen canvas (viewfinder lives here) ─────────────────────────
        self.canvas = tk.Canvas(
            self.root, width=self._sw, height=self._sh,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Reserve image item in the canvas centre
        self._img_id  = self.canvas.create_image(
            self._sw // 2, self._sh // 2, anchor="center"
        )
        self._tk_img: ImageTk.PhotoImage | None = None

        # ── Overlaid instruction text (top) ────────────────────────────────────
        self.canvas.create_rectangle(
            0, 0, self._sw, self.OVERLAY_H,
            fill="black", stipple="gray50", outline="",
        )
        self.canvas.create_text(
            self._sw // 2, self.OVERLAY_H // 2,
            text="Frame your reference rectangle, then press SHUTTER (or SPACE) to capture.",
            fill="white",
            font=("Helvetica", 14, "bold"),
            anchor="center",
        )

        # ── Status text (bottom) ───────────────────────────────────────────────
        self._status_rect = self.canvas.create_rectangle(
            0, self._sh - self.STATUS_H, self._sw, self._sh,
            fill="black", stipple="gray50", outline="",
        )
        self._status_text = self.canvas.create_text(
            self._sw // 2, self._sh - self.STATUS_H // 2,
            text="", fill="#88cc88",
            font=("Helvetica", 13),
            anchor="center",
        )

        # ── MJPEG viewfinder ───────────────────────────────────────────────────
        self._fq: queue.Queue = queue.Queue(maxsize=2)
        self._mjpeg = MJPEGReader(self._fq)
        self._mjpeg.start()
        self._poll()

        # ── Input bindings ─────────────────────────────────────────────────────
        self.root.bind("<space>",  lambda _e: self._trigger())
        self.root.bind("<Escape>", lambda _e: self._cancel())
        if _GPIO_AVAILABLE:
            GPIO.add_event_detect(
                SHUTTER_PIN, GPIO.FALLING,
                callback=lambda _ch: self.root.after(0, self._trigger),
                bouncetime=500,
            )

    # ── Viewfinder poll ────────────────────────────────────────────────────────

    def _poll(self):
        if self._triggered:
            return   # stop polling once shutter pressed
        try:
            frame = self._fq.get_nowait()
            cam1  = _extract_cam1(frame)
            self._show(cam1)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _show(self, img: Image.Image):
        """Scale img to fill the screen as a square, centred."""
        side   = min(self._sw, self._sh - self.OVERLAY_H - self.STATUS_H)
        scaled = img.resize((side, side), Image.Resampling.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(scaled)
        self.canvas.itemconfig(self._img_id, image=self._tk_img)
        # Reposition to centre vertically in the usable band
        cy = self.OVERLAY_H + (self._sh - self.OVERLAY_H - self.STATUS_H) // 2
        self.canvas.coords(self._img_id, self._sw // 2, cy)

    # ── Shutter logic ──────────────────────────────────────────────────────────

    def _trigger(self):
        if self._triggered:
            return
        self._triggered = True
        self._mjpeg.stop()
        self._set_status("Settling AWB… please wait…")
        threading.Thread(target=self._capture, daemon=True).start()

    def _set_status(self, msg: str):
        self.canvas.itemconfig(self._status_text, text=msg)

    def _cancel(self):
        self._mjpeg.stop()
        self.root.destroy()

    def _capture(self):
        import time as _time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = REPO_ROOT / "input" / f"rotcalib_{timestamp}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.root.after(0, lambda: self._set_status("Capturing…"))
        try:
            proc = subprocess.Popen([
                "rpicam-still",
                "-o", str(out_path),
                "--nopreview",
                "--quality", "100",
                "--awb", "auto",
                "--denoise", "cdn_hq",
                "--sharpness", "1.5",
                "-t", "0",
                "--signal",
                "--shutter", str(SHUTTER_US),
            ])
            _time.sleep(AWB_SETTLE_S)
            os.kill(proc.pid, signal.SIGUSR1)
            _time.sleep(1.5)
            proc.send_signal(signal.SIGINT)
            proc.wait()

            self.result_path = out_path
            name = out_path.name
            self.root.after(0, lambda: self._set_status(f"Captured: {name}"))
            self.root.after(800, self.root.destroy)
        except Exception as exc:
            self.root.after(0, lambda: self._set_status(f"Capture failed: {exc}"))
            self._triggered = False   # allow retry

    def run(self) -> Path | None:
        self.root.mainloop()
        if _GPIO_AVAILABLE:
            try:
                GPIO.remove_event_detect(SHUTTER_PIN)
            except Exception:
                pass
        return self.result_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Skip capture and use this existing composite image.",
    )
    args = parser.parse_args()

    if args.image:
        # Image supplied directly — skip capture
        raw_path = resolve_input_image(args.image)
        print(f"Using supplied image: {raw_path}")
    else:
        # Show capture screen, wait for shutter press
        capture_ui = CaptureUI()
        raw_path = capture_ui.run()
        if raw_path is None or not raw_path.exists():
            print("Capture cancelled or failed — exiting.")
            sys.exit(1)
        print(f"Captured: {raw_path}")

    camera_images = load_camera_images(raw_path)
    print(f"Loaded {len(camera_images)} camera images.")

    angles: list[float] = []
    for cam_idx, cam_img in enumerate(camera_images):
        prev = angles[-1] if angles else None
        ui   = CameraCalibUI(cam_idx, cam_img, prev_angle=prev)
        angle = ui.run()
        angles.append(angle)
        print(f"  Camera {cam_idx}: rotation = {angle:+.4f}°")

    summary = SummaryUI(camera_images, angles)
    summary.run()


if __name__ == "__main__":
    main()
