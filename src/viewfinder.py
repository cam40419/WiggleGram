import io
import logging
import queue
import subprocess
import threading
from collections import deque
from typing import Optional

from PIL import Image

from camera_settings import VF_WIDTH, VF_HEIGHT, VF_FPS, viewfinder_command

logger = logging.getLogger(__name__)


class MJPEGReader(threading.Thread):
    """Continuously reads MJPEG frames from rpicam-vid stdout into a queue."""

    # JPEG SOI / EOI markers
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, frame_queue: queue.Queue, camera: Optional[int] = None,
                 width: int = VF_WIDTH, height: int = VF_HEIGHT, fps: int = VF_FPS):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.camera      = camera
        self.width       = width
        self.height      = height
        self.fps         = fps
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._proc_lock = threading.Lock()
        self.error: Optional[str] = None
        self._stderr_tail = deque(maxlen=30)

    def _read_stderr(self):
        # Drain stderr separately: stdout contains binary JPEGs, and an unread
        # stderr pipe can block the camera process even while frames are flowing.
        for line in self._proc.stderr:
            message = line.decode("utf-8", errors="replace").rstrip()
            self._stderr_tail.append(message)
            logger.info("Camera: %s", message)

    def _failure_message(self) -> str:
        output = "\n".join(self._stderr_tail).lower()
        if "no cameras available" in output:
            return "No cameras available. Run: python3 scripts/setup_imx519.py --check"
        if "failed to acquire camera" in output or "device or resource busy" in output:
            return "Camera is busy. Close other camera apps and restart WiggleGram."
        return "Camera preview failed. See wigglegram.log for details."

    def run(self):
        stderr_reader = None
        failed = False
        try:
            cmd = viewfinder_command(camera=self.camera, width=self.width,
                                     height=self.height, fps=self.fps)
            logger.info("Starting viewfinder: %s", " ".join(cmd))
            with self._proc_lock:
                if self._stop_event.is_set():
                    return
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
            stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
            stderr_reader.start()
            self._read_frames()
            failed = not self._stop_event.is_set()
        except Exception:
            failed = True
            logger.exception("Viewfinder failed; check the IMX519 driver and camera configuration")
        finally:
            self._stop_process()
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()
            if stderr_reader is not None:
                stderr_reader.join(timeout=3)
                if not stderr_reader.is_alive():
                    self._proc.stderr.close()
            if failed and not self._stop_event.is_set():
                self.error = self._failure_message()
                logger.error("%s", self.error)

    def _read_frames(self):
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

    def _stop_process(self):
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=3)

    def stop(self):
        self._stop_event.set()
        self._stop_process()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=3)
        logger.info("Viewfinder stopped")
