"""Frame individual camera views without resampling their captured pixels."""

from math import gcd
from typing import Tuple

from PIL import Image


def aspect_crop_size(size: Tuple[int, int], aspect: Tuple[int, int] = (16, 9)) -> Tuple[int, int]:
    """Return the largest exact aspect-ratio rectangle that fits in size."""
    numerator, denominator = aspect
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Aspect-ratio terms must be positive")
    divisor = gcd(numerator, denominator)
    numerator, denominator = numerator // divisor, denominator // divisor
    scale = min(size[0] // numerator, size[1] // denominator)
    if scale < 1:
        raise ValueError(f"Image {size} is too small for aspect ratio {aspect}")
    return numerator * scale, denominator * scale


def aspect_crop_box(size: Tuple[int, int], aspect: Tuple[int, int] = (16, 9)) -> Tuple[int, int, int, int]:
    width, height = aspect_crop_size(size, aspect)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    return left, top, left + width, top + height


def crop_to_aspect(image: Image.Image, aspect: Tuple[int, int] = (16, 9)) -> Image.Image:
    """Center-crop a view at its original pixel scale, with no stretching."""
    return image.crop(aspect_crop_box(image.size, aspect))


def pad_to_aspect(image: Image.Image, aspect: Tuple[int, int] = (16, 9)) -> Image.Image:
    """Fit the entire view in an exact aspect-ratio canvas at native pixel scale."""
    numerator, denominator = aspect
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Aspect-ratio terms must be positive")
    divisor = gcd(numerator, denominator)
    numerator, denominator = numerator // divisor, denominator // divisor
    scale = max((image.width + numerator - 1) // numerator,
                (image.height + denominator - 1) // denominator)
    canvas = Image.new("RGB", (numerator * scale, denominator * scale), "black")
    canvas.paste(image, ((canvas.width - image.width) // 2,
                         (canvas.height - image.height) // 2))
    return canvas


def preview_view(composite: Image.Image, camera_index: int = 1) -> Image.Image:
    """Extract the complete camera view in its original sensor orientation."""
    if not 0 <= camera_index < 4:
        raise ValueError("Camera index must be between 0 and 3")
    width, height = composite.width // 2, composite.height // 2
    row, column = divmod(camera_index, 2)
    view = composite.crop((column * width, row * height,
                           (column + 1) * width, (row + 1) * height))
    return view


def camera_views(composite: Image.Image) -> list:
    """Return all four full views without rotating or flipping any pixels."""
    return [preview_view(composite, camera) for camera in range(4)]
