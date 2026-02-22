import os
import logging
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def main(raw_path, save_path):
    raw = Image.open(raw_path)
    
    # pieces = split_grid(raw, 2, 2)
    pieces = load_images("input/ben")
    
    wigglegram(pieces, save_path)

def load_images(input_path):
    files = sorted(os.listdir(input_path))
    image_files = [
        f
        for f in files
        if f.lower().endswith((".jpg", ".jpeg"))
    ]
    
    images = [
        ImageOps.exif_transpose(Image.open(os.path.join(input_path, f))).convert("RGB")
        for f in image_files
    ]
    
    return images
    

def wigglegram(images, save_dir):
    total = len(images)
    shifted = [crop_and_shift(img, i, total) for i, img in enumerate(images)]
    # shifted = images
    snake = shifted + shifted[-2:0:-1]

    pal0 = snake[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    out = [pal0]
    for fr in snake[1:]:
        out.append(fr.quantize(palette=pal0))

    snake[0].save(
        f"{save_dir}/wiggle.gif",
        save_all=True,
        append_images=out,
        duration=100,
        loop=0,
    )


def split_grid(img, rows, cols):
    width, height = img.size
    
    piece_width = width // cols
    piece_height = height // rows

    # Create a list to hold the pieces
    pieces = []

    # Loop through the image and add each piece to the list
    for i in range(rows):
        for j in range(cols):
            # Calculate the position of the current piece
            left = j * piece_width
            upper = i * piece_height
            right = (j + 1) * piece_width
            lower = (i + 1) * piece_height
            # Crop the image and add it to the list
            pieces.append(img.crop((left, upper, right, lower)))

    return pieces


def crop_and_shift(img, index: int, total_images: int, scale=0.9, shift_amount=0.0):
    shift_amount = max(0.0, min(1.0, float(shift_amount)))

    w, h = img.size
    crop_w = int(round(w * scale))
    crop_h = h

    y = int(round((h - crop_h) / 2))

    available = w - crop_w

    if total_images <= 1:
        x = int(round(available / 2))
    else:
        t = (total_images - 1 - index) / (total_images - 1)
        x = int(round((available / 2) + (t - 0.5) * available * shift_amount))

    return img.crop((x, y, x + crop_w, y + crop_h))


if __name__ == "__main__":
    main("input/test.jpg", "output")
