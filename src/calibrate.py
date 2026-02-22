import os
import glob
import numpy as np
import cv2


def list_synced_sets(base_dir: str):
    """Return list of per-set paths: [(cam0_path, cam1_path, cam2_path, cam3_path), ...]"""
    cam_dirs = [os.path.join(base_dir, f"cam{i}") for i in range(4)]
    for d in cam_dirs:
        if not os.path.isdir(d):
            raise ValueError(f"Missing directory: {d}")

    cam0_files = sorted(glob.glob(os.path.join(cam_dirs[0], "*")))
    if not cam0_files:
        raise ValueError("No files found in cam0 folder")

    sets = []
    for p0 in cam0_files:
        name = os.path.basename(p0)
        ps = [os.path.join(cam_dirs[i], name) for i in range(4)]
        if all(os.path.isfile(p) for p in ps):
            sets.append(tuple(ps))

    if len(sets) < 5:
        print(f"Warning: Only {len(sets)} synced sets found. Calibration may be poor.")

    return sets


def find_corners(image, pattern_size):
    """Find & refine chessboard corners. Returns (ok, corners)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not ok:
        return False, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    return True, corners


def build_object_points(pattern_size, square_size):
    """Create object points for the chessboard: (0,0,0), (1,0,0)... scaled by square_size."""
    cols, rows = (
        pattern_size  # OpenCV uses (columns, rows)
    )
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    return objp


def calibrate_four_cameras(
    base_dir: str,
    pattern_size=(9, 6),  # inner corners (columns, rows)
    square_size=25.0,  # mm (or any unit, but be consistent)
    output_npz="calibration_4cam.npz",
):
    sets = list_synced_sets(base_dir)

    # Collect per-camera image points + shared object points
    objp = build_object_points(pattern_size, square_size)

    objpoints = []  # list of objp per successful set
    imgpoints = [[] for _ in range(4)]  # per cam: list of corners

    image_size = None
    used = 0

    for p0, p1, p2, p3 in sets:
        paths = [p0, p1, p2, p3]
        imgs = []
        for p in paths:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                imgs = None
                break
            imgs.append(im)

        if imgs is None:
            continue

        if image_size is None:
            h, w = imgs[0].shape[:2]
            image_size = (w, h)

        corners_all = []
        ok_all = True
        for i in range(4):
            ok, corners = find_corners(imgs[i], pattern_size)
            if not ok:
                ok_all = False
                break
            corners_all.append(corners)

        # Only keep sets where ALL 4 cameras see the board
        if not ok_all:
            continue

        objpoints.append(objp)
        for i in range(4):
            imgpoints[i].append(corners_all[i])

        used += 1

    print(f"Using {used} synced sets for calibration.")

    # Calibrate each camera intrinsics independently
    Ks = []
    dists = []
    rvecs_all = []
    tvecs_all = []
    reproj_errs = []

    for i in range(4):
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints[i], image_size, None, None
        )
        Ks.append(K)
        dists.append(dist)
        rvecs_all.append(rvecs)
        tvecs_all.append(tvecs)
        reproj_errs.append(ret)
        print(f"cam{i}: reprojection RMS = {ret:.4f}")

    # Calibrate extrinsics between cam0 and each other cam using stereoCalibrate
    # We keep intrinsics fixed to the per-camera results to stabilize.
    R_0i = [np.eye(3, dtype=np.float64)]
    T_0i = [np.zeros((3, 1), dtype=np.float64)]
    E_0i = [np.zeros((3, 3), dtype=np.float64)]
    F_0i = [np.zeros((3, 3), dtype=np.float64)]
    stereo_rms = [0.0]

    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    for i in [1, 2, 3]:
        rms, K0, d0, Ki, di, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints[0],
            imgpoints[i],
            Ks[0],
            dists[0],
            Ks[i],
            dists[i],
            image_size,
            criteria=criteria,
            flags=stereo_flags,
        )
        R_0i.append(R)
        T_0i.append(T)
        E_0i.append(E)
        F_0i.append(F)
        stereo_rms.append(rms)
        print(f"stereo cam0-cam{i}: RMS = {rms:.4f}")

    # Save to NPZ for instant runtime loading
    np.savez_compressed(
        output_npz,
        pattern_size=np.array(pattern_size),
        square_size=np.array([square_size], dtype=np.float64),
        image_size=np.array(image_size),
        Ks=np.array(Ks),
        dists=np.array(dists, dtype=object),
        R_0i=np.array(R_0i, dtype=object),
        T_0i=np.array(T_0i, dtype=object),
        stereo_rms=np.array(stereo_rms),
        per_cam_rms=np.array(reproj_errs),
    )
    print(f"Saved calibration to: {output_npz}")


if __name__ == "__main__":
    calibrate_four_cameras(
        base_dir="calibration",
        pattern_size=(9, 6),  # change to your board's inner corners
        square_size=18.0,  # change to your square size (mm)
        output_npz="calibration_4cam.npz",
    )
