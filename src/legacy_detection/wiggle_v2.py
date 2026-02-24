import cv2
import numpy as np
import os
from PIL import Image

# -------------------------
# Person detection (YOLOv3)
# -------------------------
def load_yolo(weights_path, config_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_people_yolo(image, net, output_layers, conf_thresh=0.5, nms_thresh=0.4):
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            # YOLO COCO class 0 = person
            if class_id == 0 and confidence >= conf_thresh:
                cx = int(det[0] * W)
                cy = int(det[1] * H)
                w = int(det[2] * W)
                h = int(det[3] * H)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    people = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            people.append((x1, y1, x2, y2, confidences[i]))

    return people

def pick_center_person(people, image_shape):
    """Prefer the person whose bbox center is closest to the image center.
       Tie-breaker: larger area."""
    if not people:
        return None

    H, W = image_shape[:2]
    img_cx, img_cy = W / 2, H / 2

    best = None
    best_score = None
    for (x1, y1, x2, y2, conf) in people:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist2 = (cx - img_cx) ** 2 + (cy - img_cy) ** 2
        area = (x2 - x1) * (y2 - y1)

        # Lower distance is better; incorporate area as a small bonus
        score = dist2 - 0.0001 * area
        if best_score is None or score < best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    return best

# -------------------------
# Face detection (YuNet)
# -------------------------
def load_yunet(onnx_path, input_size=(320, 320)):
    # YuNet is shipped as an ONNX model; you download it once.
    # See: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    detector = cv2.FaceDetectorYN.create(
        onnx_path,
        "",
        input_size,
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector

def detect_face_in_roi(image, face_detector, roi):
    """Return face center_x in full-image coordinates, or None."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = roi
    roi_img = image[y1:y2, x1:x2]
    if roi_img.size == 0:
        return None

    # YuNet requires you to set input size to current image size you pass in
    face_detector.setInputSize((roi_img.shape[1], roi_img.shape[0]))
    _, faces = face_detector.detect(roi_img)

    if faces is None or len(faces) == 0:
        return None

    # Pick the largest face in ROI
    best = None
    best_area = 0
    for f in faces:
        fx, fy, fw, fh = f[:4]
        area = fw * fh
        if area > best_area:
            best_area = area
            best = (fx, fy, fw, fh)

    fx, fy, fw, fh = best
    face_cx_roi = fx + fw / 2
    face_cx_full = x1 + face_cx_roi
    return float(face_cx_full)

# -------------------------
# Alignment by horizontal shift
# -------------------------
def shift_image_horiz(image, dx):
    """Shift entire image by dx pixels (positive shifts right). Keeps same size."""
    H, W = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(
        image, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT  # looks nicer than black bars
    )
    return shifted

def center_crop(image, out_w, out_h):
    H, W = image.shape[:2]
    x1 = max(0, (W - out_w) // 2)
    y1 = max(0, (H - out_h) // 2)
    return image[y1:y1 + out_h, x1:x1 + out_w]

# -------------------------
# Main
# -------------------------
def make_wigglegram(
    image_dir,
    yolo_weights,
    yolo_cfg,
    yunet_onnx,
    out_gif="wigglegram.gif",
    out_size=(512, 512),
    duration_ms=80
):
    net, out_layers = load_yolo(yolo_weights, yolo_cfg)
    face_detector = load_yunet(yunet_onnx)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not image_files:
        print("No images found.")
        return

    frames = []
    target_x = None  # face x to align to

    for fname in image_files:
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        people = detect_people_yolo(img, net, out_layers, conf_thresh=0.45)
        person_roi = pick_center_person(people, img.shape)

        if person_roi is None:
            # If no person, skip (or you can fallback to no shift)
            continue

        face_x = detect_face_in_roi(img, face_detector, person_roi)
        if face_x is None:
            # If no face found, skip (or fallback to bbox center)
            # bbox center fallback:
            x1, y1, x2, y2 = person_roi
            face_x = (x1 + x2) / 2

        if target_x is None:
            # Use first valid frame as reference
            target_x = face_x

        dx = target_x - face_x  # shift so face_x moves to target_x
        shifted = shift_image_horiz(img, dx)

        # Make output consistent size
        cropped = center_crop(shifted, out_size[0], out_size[1])
        frames.append(cropped)

    if not frames:
        print("No usable frames (person/face not detected).")
        return

    gif_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    gif_frames[0].save(
        out_gif,
        save_all=True,
        append_images=gif_frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0
    )
    print(f"Wigglegram created as '{out_gif}' with {len(frames)} frames.")

if __name__ == "__main__":
    make_wigglegram(
        image_dir="input/oil",
        yolo_weights="C:/path/to/yolov3.weights",
        yolo_cfg="C:/path/to/yolov3.cfg",
        yunet_onnx="C:/path/to/face_detection_yunet_2023mar.onnx",
        out_gif="wigglegram.gif",
        out_size=(512, 512),
        duration_ms=80
    )