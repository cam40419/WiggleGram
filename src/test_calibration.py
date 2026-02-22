import os
import cv2
import numpy as np


def load_calib(npz_path="calibration_4cam.npz"):
    data = np.load(npz_path, allow_pickle=True)

    Ks = np.asarray(data["Ks"], dtype=np.float64)

    raw_dists = data["dists"]
    dists = []
    for i in range(4):
        di = np.asarray(raw_dists[i], dtype=np.float64).reshape(-1, 1)
        dists.append(di)

    return Ks, dists


def undistort_image(img, K, dist):
    return cv2.undistort(img, K, dist)


def process_camera_folder(in_dir, out_dir, K, dist):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(in_dir))

    for name in files:
        path = os.path.join(in_dir, name)

        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        out = undistort_image(img, K, dist)

        save_path = os.path.join(out_dir, name)
        cv2.imwrite(save_path, out)

        print("Saved:", save_path)


def process_all_cameras(base_input, base_output, calib_file):

    Ks, dists = load_calib(calib_file)

    for i in range(4):
        in_dir = os.path.join(base_input, f"cam{i}")
        out_dir = os.path.join(base_output, f"cam{i}")

        print(f"\nProcessing cam{i}")
        process_camera_folder(in_dir, out_dir, Ks[i], dists[i])


if __name__ == "__main__":

    process_all_cameras(
        base_input="calibration",
        base_output="undistorted",
        calib_file="calibration_4cam.npz",
    )
