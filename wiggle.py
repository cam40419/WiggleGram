import os
from PIL import Image, ImageOps


def wigglegram(img_dir: str):
    files = sorted(os.listdir(img_dir))
    image_files = [
        f
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))
    ]

    images = [
        ImageOps.exif_transpose(Image.open(os.path.join(img_dir, f))).convert("RGB")
        for f in image_files
    ]

    total = len(images)
    shifted = [crop_and_shift(img, i, total) for i, img in enumerate(images)]
    # shifted = images
    snake = shifted + shifted[-2:0:-1]

    snake[0].save(
        f"{img_dir}.gif",
        save_all=True,
        append_images=snake[1:],
        duration=100,
        loop=0,
    )


def crop_and_shift(img, index: int, total_images: int):
    scale = 0.9
    w, h = img.size

    x = w * (1 - scale) / (total_images - 1) * (total_images - 1 - index)
    y = h * (1 - scale) / 2
    width = w * scale

    return img.crop((x, y, x + width, h))


if __name__ == "__main__":
    wigglegram("input/oil")
