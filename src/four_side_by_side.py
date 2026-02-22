import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from PIL import Image, ImageFile, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]   # src/ -> repo root

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def resolve_input_file() -> Path:
    captured_files = sorted((REPO_ROOT / "input").glob("photo_*.jpg"))
    if captured_files:
        return captured_files[-1]
    return REPO_ROOT / "input" / "ben" / "IMG_1638.JPG"


def split_grid(img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    width, height = img.size
    piece_width = width // cols
    piece_height = height // rows

    pieces: List[Image.Image] = []
    for row in range(rows):
        for col in range(cols):
            left = col * piece_width
            upper = row * piece_height
            right = (col + 1) * piece_width
            lower = (row + 1) * piece_height
            pieces.append(img.crop((left, upper, right, lower)))

    return pieces


def stitch_side_by_side(images: List[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images provided")

    base_w, base_h = images[0].size
    normalized = [im if im.size == (base_w, base_h) else im.resize((base_w, base_h), Image.Resampling.BILINEAR) for im in images]

    out = Image.new("RGB", (base_w * len(normalized), base_h))
    for idx, piece in enumerate(normalized):
        out.paste(piece, (idx * base_w, 0))

    return out


def main(raw_path: str, save_path: str) -> str:
    raw = Image.open(raw_path)
    raw = ImageOps.exif_transpose(raw).convert("RGB")

    pieces = split_grid(raw, 2, 2)
    pieces[0] = pieces[0].rotate(270, expand=False)
    pieces[1] = pieces[1].rotate(270, expand=False)
    pieces[2] = pieces[2].rotate(90, expand=False)
    pieces[3] = pieces[3].rotate(90, expand=False)

    stitched = stitch_side_by_side(pieces)

    os.makedirs(save_path, exist_ok=True)
    out_path = Path(save_path) / "four_side_by_side.jpg"
    stitched.save(out_path, format="JPEG", quality=95)

    logger.info("Saved side-by-side image to %s", out_path)
    return str(out_path)


# Capture photo on RPi and process it

def capture_and_process() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file = REPO_ROOT / "input" / f"photo_{timestamp}.jpg"
    output_dir = REPO_ROOT / "output" / timestamp

    input_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["rpicam-still", "-o", str(input_file), "--nopreview"],
            check=True,
        )
        logger.info("Photo captured: %s", input_file)
    except subprocess.CalledProcessError as err:
        logger.error("Failed to capture photo: %s", err)
        return

    main(str(input_file), str(output_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    capture_and_process()
