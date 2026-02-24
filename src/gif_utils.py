from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageTk


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
