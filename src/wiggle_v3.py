import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

def main(raw_path, save_path):
    raw = Image.open(raw_path)
    
    pieces = split_grid(raw, 2, 2)
    identified_subjects = identify_subjects(pieces)

    wigglegram(pieces, identified_subjects, save_path)

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

def identify_subjects(images):
    subjects_info = []
    
    # Update path to the location where you saved the haarcascade file
    cascade_path = 'haarcascade_frontalface_default.xml'  # Change this to the correct path
    face_cascade = cv2.CascadeClassifier(cascade_path)

    for img in images:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            subject_distance = estimate_distance(faces[0])  # You need to define this
            subjects_info.append((x, y, w, h, subject_distance))
        else:
            subjects_info.append(None)

    return subjects_info


def estimate_distance(face):
    # Implement a simple distance estimation based on size or other properties
    # Placeholder for an actual implementation
    return 1.0  # Replace with actual distance calculation

def wigglegram(images, subjects, save_dir):
    total = len(images)
    shifted = [crop_and_shift(img, subjects[i], i, total) for i, img in enumerate(images)]
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

    pieces = []

    for i in range(rows):
        for j in range(cols):
            left = j * piece_width
            upper = i * piece_height
            right = (j + 1) * piece_width
            lower = (i + 1) * piece_height
            pieces.append(img.crop((left, upper, right, lower)))

    return pieces

def crop_and_shift(img, subject_info, index: int, total_images: int, scale=0.9, min_shift=0.3, max_shift=1.0):
    if subject_info is None:
        return img  # No subject detected

    shift_amount = calculate_shift_based_on_distance(subject_info)

    shift_amount = max(min_shift, min(max_shift, shift_amount))

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

def calculate_shift_based_on_distance(subject_info):
    # Determine the shift amount based on estimated subject distance
    # Placeholder logic for calculating shift from subject distance
    x, y, w, h, distance = subject_info
    max_distance = 2.0  # Max distance to consider
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))  # Example scale to shift amount

if __name__ == "__main__":
    main("input/test.jpg", "output")
