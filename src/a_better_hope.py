import os
import logging
from typing import List, Tuple, Optional, Literal

import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps, ImageFile

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # src/ -> repo root
INPUT_DIR = REPO_ROOT / "input" / "ben/IMG_1638.JPG"
OUTPUT_DIR = REPO_ROOT / "output" / "ben"

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

AxisType = Literal["x", "xy"]

# Main pipeline
def main(raw_path: str, save_path: str):
    # pieces = load_images(raw_path)
    
    raw = Image.open(raw_path)
    pieces = split_grid(raw, 2, 2)

    # Detect face bbox first, then run landmarker only on a small ROI
    anchors = pick_nose_anchors_all_frames_face_first(
        pieces,
        detector_model_path="models/face_detector.tflite",
        landmarker_model_path="face_landmarker.task",
        detect_max_dim=320,
        roi_max_dim=384,
        roi_expand=0.25,
        fail_mode="none",
    )

    w, h = pieces[0].size
    target_xy = (w // 2, h // 2)

    wigglegram_anchors(
        pieces,
        anchors=anchors,
        save_dir=save_path,
        target_xy=target_xy,
        axis="x",
        duration_ms=100,
        crop_common=True,
    )


# Image loading
def load_images(input_path: str) -> List[Image.Image]:
    files = sorted(os.listdir(input_path))
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise ValueError(f"No images found in: {input_path}")

    images: List[Image.Image] = []
    for f in image_files:
        p = os.path.join(input_path, f)
        im = Image.open(p)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.load()
        images.append(im)

    return images


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


# Fast face-first nose-anchor detection
def pick_nose_anchors_all_frames_face_first(
    images: List[Image.Image],
    *,
    detector_model_path: str,
    landmarker_model_path: str,
    max_num_faces: int = 1,
    # performance knobs (Pi-friendly):
    detect_max_dim: int = 320,   # face bbox detection input max dimension
    roi_max_dim: int = 384,      # landmarker ROI input max dimension
    roi_expand: float = 0.25,    # bbox expansion fraction
    min_face_confidence: float = 0.5,
    fail_mode: Literal["raise", "previous", "center", "none"] = "none",
) -> List[Optional[Tuple[int, int]]]:
    if not images:
        raise ValueError("No images provided")

    # MediaPipe Tasks imports
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    # Detector options
    detector_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=detector_model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=min_face_confidence,
    )

    landmarker_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=max_num_faces,
        output_face_blendshapes=False,
    )

    anchors: List[Optional[Tuple[int, int]]] = []
    prev_anchor: Optional[Tuple[int, int]] = None
    prev_bbox: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1 in full-res

    with FaceDetector.create_from_options(detector_options) as detector, \
         FaceLandmarker.create_from_options(landmarker_options) as landmarker:

        for idx, pil_im in enumerate(images):
            w_full, h_full = pil_im.size

            # Face bbox detect (downscaled)
            scale_det = 1.0
            if max(w_full, h_full) > detect_max_dim:
                scale_det = detect_max_dim / float(max(w_full, h_full))
                w_det = int(round(w_full * scale_det))
                h_det = int(round(h_full * scale_det))
                det_im = pil_im.resize((w_det, h_det), Image.Resampling.BILINEAR)
            else:
                det_im = pil_im
                w_det, h_det = w_full, h_full

            det_rgb = np.asarray(det_im)  # RGB
            det_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_rgb)
            det_result = detector.detect(det_mp_image)

            bbox_full: Optional[Tuple[int, int, int, int]] = None

            # Take best detection (highest score)
            if det_result.detections:
                best = None
                best_score = -1.0
                for d in det_result.detections:
                    score = float(d.categories[0].score) if d.categories else 0.0
                    if score > best_score:
                        best_score = score
                        best = d

                if best is not None and best.bounding_box is not None:
                    bb = best.bounding_box  # in DET image pixels
                    x0_det = bb.origin_x
                    y0_det = bb.origin_y
                    x1_det = bb.origin_x + bb.width
                    y1_det = bb.origin_y + bb.height

                    # Map bbox back to full-res
                    if scale_det != 1.0:
                        x0 = int(round(x0_det / scale_det))
                        y0 = int(round(y0_det / scale_det))
                        x1 = int(round(x1_det / scale_det))
                        y1 = int(round(y1_det / scale_det))
                    else:
                        x0 = int(round(x0_det))
                        y0 = int(round(y0_det))
                        x1 = int(round(x1_det))
                        y1 = int(round(y1_det))

                    # Expand bbox a bit
                    bw = x1 - x0
                    bh = y1 - y0
                    pad_x = int(round(bw * roi_expand))
                    pad_y = int(round(bh * roi_expand))

                    x0 = max(0, x0 - pad_x)
                    y0 = max(0, y0 - pad_y)
                    x1 = min(w_full, x1 + pad_x)
                    y1 = min(h_full, y1 + pad_y)

                    # sanity clamp
                    if (x1 - x0) >= 2 and (y1 - y0) >= 2:
                        bbox_full = (x0, y0, x1, y1)

            # If detection failed, optionally reuse previous bbox
            if bbox_full is None and prev_bbox is not None:
                bbox_full = prev_bbox

            # If still no bbox, decide behavior
            if bbox_full is None:
                # True "no transform" fallback
                if fail_mode == "none":
                    anchors.append(None)
                    continue

                msg = f"No face bbox detected in frame {idx+1}/{len(images)}"

                if fail_mode == "raise":
                    raise RuntimeError(msg)

                if fail_mode == "previous" and prev_anchor is not None:
                    anchors.append(prev_anchor)
                    continue

                if fail_mode == "center":
                    center = (w_full // 2, h_full // 2)
                    anchors.append(center)
                    prev_anchor = center
                    continue

                # If previous requested but we don't have one yet, fall back to none.
                anchors.append(None)
                continue

            prev_bbox = bbox_full

            # Landmark nose on ROI (downscaled)
            x0, y0, x1, y1 = bbox_full
            roi = pil_im.crop((x0, y0, x1, y1))
            roi_w, roi_h = roi.size

            scale_roi = 1.0
            if max(roi_w, roi_h) > roi_max_dim:
                scale_roi = roi_max_dim / float(max(roi_w, roi_h))
                rw = int(round(roi_w * scale_roi))
                rh = int(round(roi_h * scale_roi))
                roi_det = roi.resize((rw, rh), Image.Resampling.BILINEAR)
            else:
                roi_det = roi
                rw, rh = roi_w, roi_h

            roi_rgb = np.asarray(roi_det)
            roi_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
            lm_result = landmarker.detect(roi_mp_image)

            if not lm_result.face_landmarks:
                if fail_mode == "none":
                    anchors.append(None)
                    continue

                msg = f"No landmarks in ROI for frame {idx+1}/{len(images)}"

                if fail_mode == "raise":
                    raise RuntimeError(msg)

                if fail_mode == "previous" and prev_anchor is not None:
                    anchors.append(prev_anchor)
                    continue

                if fail_mode == "center":
                    center = (w_full // 2, h_full // 2)
                    anchors.append(center)
                    prev_anchor = center
                    continue

                anchors.append(None)
                continue

            # Nose tip landmark index = 1 (normalized)
            nose = lm_result.face_landmarks[0][1]
            x_roi_det = nose.x * rw
            y_roi_det = nose.y * rh

            # Map to ROI full-res
            if scale_roi != 1.0:
                x_roi = x_roi_det / scale_roi
                y_roi = y_roi_det / scale_roi
            else:
                x_roi = x_roi_det
                y_roi = y_roi_det

            # Map to full-res image
            x_full = int(round(x0 + x_roi))
            y_full = int(round(y0 + y_roi))

            # Clamp
            x_full = max(0, min(x_full, w_full - 1))
            y_full = max(0, min(y_full, h_full - 1))

            anchor = (x_full, y_full)
            anchors.append(anchor)
            prev_anchor = anchor

    return anchors


# Wiggle creation using anchors
def wigglegram_anchors(
    images: List[Image.Image],
    *,
    anchors: List[Optional[Tuple[int, int]]],
    save_dir: str,
    target_xy: Optional[Tuple[int, int]] = None,
    axis: AxisType = "x",
    duration_ms: int = 100,
    crop_common: bool = True,
):
    if not images:
        raise ValueError("No images provided")
    if len(anchors) != len(images):
        raise ValueError(f"anchors length ({len(anchors)}) must match images length ({len(images)})")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "wiggle.gif")

    base_w, base_h = images[0].size
    if any(im.size != (base_w, base_h) for im in images):
        images = [im.resize((base_w, base_h), Image.Resampling.BILINEAR) for im in images]

    if target_xy is None:
        target_xy = (base_w // 2, base_h // 2)

    tx, ty = target_xy

    shifts: List[Tuple[int, int]] = []
    for anchor in anchors:
        if anchor is None:
            shifts.append((0, 0))  # no transform
            continue

        ax, ay = anchor
        dx = int(round(tx - ax))
        dy = 0 if axis == "x" else int(round(ty - ay))
        shifts.append((dx, dy))

    shifted = [translate_image(images[i], shifts[i]) for i in range(len(images))]

    if crop_common:
        shifted = crop_to_common_valid_area(shifted, shifts)

    snake = shifted + shifted[-2:0:-1]

    pal0 = snake[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    out_frames = [pal0]
    for fr in snake[1:]:
        out_frames.append(fr.quantize(palette=pal0))

    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
        format="GIF",
    )

    logger.info("Saved wiggle GIF to %s (nose anchors, axis=%s)", out_path, axis)


def translate_image(im: Image.Image, shift_xy: Tuple[int, int]) -> Image.Image:
    dx, dy = shift_xy
    w, h = im.size
    return im.transform(
        (w, h),
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=None,
    )


def crop_to_common_valid_area(frames: List[Image.Image], shifts: List[Tuple[int, int]]) -> List[Image.Image]:
    if not frames:
        return frames

    w, h = frames[0].size
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]

    max_right = max(0, max(dxs))
    max_left = max(0, -min(dxs))
    max_down = max(0, max(dys))
    max_up = max(0, -min(dys))

    x0 = max_right
    x1 = w - max_left
    y0 = max_down
    y1 = h - max_up

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    return [fr.crop((x0, y0, x1, y1)) for fr in frames]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(str(INPUT_DIR), str(OUTPUT_DIR))