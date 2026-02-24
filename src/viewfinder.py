import io
import logging
import queue
import subprocess
import threading
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class MJPEGReader(threading.Thread):
    """Continuously reads MJPEG frames from rpicam-vid stdout into a queue."""

    # JPEG SOI / EOI markers
    _SOI = b"\xff\xd8"
    _EOI = b"\xff\xd9"

    def __init__(self, frame_queue: queue.Queue, camera: int = 0,
                 width: int = 2028, height: int = 1520, fps: int = 15):
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
