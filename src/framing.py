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


def crop_to_aspect(image: Image.Image, aspect: Tuple[int, int] = (16, 9)) -> Image.Image:
    """Center-crop a view at its original pixel scale, with no stretching."""
    width, height = aspect_crop_size(image.size, aspect)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def preview_view(composite: Image.Image, camera_index: int = 1) -> Image.Image:
    """Extract one quadrant, apply the rig's 90° CCW mounting, and frame it."""
    if not 0 <= camera_index < 4:
        raise ValueError("Camera index must be between 0 and 3")
    width, height = composite.width // 2, composite.height // 2
    row, column = divmod(camera_index, 2)
    view = composite.crop((column * width, row * height,
                           (column + 1) * width, (row + 1) * height))
    return crop_to_aspect(view.transpose(Image.Transpose.ROTATE_90))
