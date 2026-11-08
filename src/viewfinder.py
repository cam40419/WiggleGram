import io
import logging
import queue
import subprocess
import threading
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

    def run(self):
        try:
            cmd = viewfinder_command(camera=self.camera, width=self.width,
                                     height=self.height, fps=self.fps)
            logger.info("Starting viewfinder: %s", " ".join(cmd))
            with self._proc_lock:
                if self._stop_event.is_set():
                    return
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            self._read_frames()
        except Exception:
            logger.exception("Viewfinder failed; check the IMX519 driver and camera configuration")
        finally:
            self._stop_process()
            if self._proc is not None and self._proc.stdout is not None:
                self._proc.stdout.close()

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
