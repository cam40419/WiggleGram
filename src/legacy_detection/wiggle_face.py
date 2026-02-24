import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    center_dist_px: float  # distance from image center to box center

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def clamp_box(x1, y1, x2, y2, W, H) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W - 1)))
    y2 = int(max(0, min(y2, H - 1)))
    # Ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def pick_center_person(
    yolo_result,
    img_shape: Tuple[int, int, int],
    min_conf: float = 0.25,
    min_area_frac: float = 0.01,
) -> Optional[PersonBox]:
    """
    Selects the 'center subject' person as the one whose bbox center is closest
    to the image center, subject to confidence + minimum area filtering.
    """
    H, W = img_shape[:2]
    img_cx, img_cy = W / 2.0, H / 2.0
    img_area = float(W * H)

    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    candidates: List[PersonBox] = []
    for b in boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())

        # COCO class 0 = person in YOLOv8
        if cls_id != 0:
            continue
        if conf < min_conf:
            continue

        # xyxy is tensor shape (1,4) or (4,)
        xyxy = b.xyxy.squeeze().cpu().numpy().tolist()
        x1, y1, x2, y2 = map(float, xyxy)

        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area / img_area < min_area_frac:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = float(np.hypot(cx - img_cx, cy - img_cy))

        candidates.append(PersonBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, center_dist_px=dist))

    if not candidates:
        return None

    # Primary: closest to center
    # Tie-breakers: higher confidence, then larger area
    candidates.sort(
        key=lambda p: (
            p.center_dist_px,
            -p.conf,
            -(p.w * p.h),
        )
    )
    return candidates[0]


def annotate(img: np.ndarray, person: PersonBox) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (person.x1, person.y1), (person.x2, person.y2), (0, 255, 0), 2)
    label = f"person conf={person.conf:.3f} dist={person.center_dist_px:.1f}px"
    cv2.putText(out, label, (person.x1, max(0, person.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def detect_center_person_box(
    img_bgr: np.ndarray,
    model: YOLO,
    imgsz: int = 960,       # higher = generally more accurate boxes (slower)
    conf: float = 0.25,
    iou: float = 0.5,
    min_area_frac: float = 0.01,
) -> Optional[PersonBox]:
    """
    Runs YOLO and returns the center-most person bounding box.
    """
    # Running on the original image preserves pixel coords directly.
    res = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        device=None,   # set "0" for GPU
    )[0]
    return pick_center_person(res, img_bgr.shape, min_conf=conf, min_area_frac=min_area_frac)


def run_on_folder(
    in_dir: str,
    out_dir: str,
    model_path: str = "yolov8x.pt",  # try yolov8x for better accuracy; yolov8n is faster but less precise
    save_annotated: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    model = YOLO(model_path)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    results: Dict[str, dict] = {}

    for name in sorted(os.listdir(in_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in exts:
            continue

        path = os.path.join(in_dir, name)
        img = cv2.imread(path)
        if img is None:
            results[name] = {"error": "failed_to_read"}
            continue

        person = detect_center_person_box(
            img_bgr=img,
            model=model,
            imgsz=960,
            conf=0.25,
            iou=0.5,
            min_area_frac=0.01,
        )

        if person is None:
            results[name] = {"found": False}
            continue

        results[name] = {
            "found": True,
            "box_xyxy": [person.x1, person.y1, person.x2, person.y2],
            "confidence": person.conf,
            "center_dist_px": person.center_dist_px,
            "image_wh": [img.shape[1], img.shape[0]],
        }

        if save_annotated:
            vis = annotate(img, person)
            cv2.imwrite(os.path.join(ann_dir, name), vis)

    with open(os.path.join(out_dir, "center_person_boxes.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Example usage:
    #   python center_person_bbox.py
    # Put images in ./input_images and outputs will go to ./output
    run_on_folder(
        in_dir="input_images",
        out_dir="output",
        model_path="yolov8x.pt",
        save_annotated=True,
    )
    print("Done. See output/center_person_boxes.json and output/annotated/")