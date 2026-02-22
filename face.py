import cv2
import mediapipe as mp
import numpy as np

def highlight_nose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("No face detected.")
            return

        h, w, _ = image.shape

        # Nose tip landmark index
        nose = result.face_landmarks[0][1]

        x = int(nose.x * w)
        y = int(nose.y * h)

        cv2.circle(image, (x, y), 8, (0,255,0), -1)

    show_resized("Nose", image)
    
    
def show_resized(window_name, img, max_width=1000, max_height=800):
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h, 1)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    highlight_nose("input/ben/IMG_1638.JPG")