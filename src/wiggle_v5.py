import cv2
import os
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

def read_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img

def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def detect_and_match(img1_gray, img4_gray, nfeatures=4000):
    # ORB is fast and good enough for lots of scenes.
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)

    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k4, d4 = orb.detectAndCompute(img4_gray, None)
    if d1 is None or d4 is None or len(k1) < 10 or len(k4) < 10:
        raise RuntimeError("Not enough features detected. Try increasing nfeatures or improving texture/lighting.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # KNN match + ratio test
    matches_knn = bf.knnMatch(d1, d4, k=2)
    good = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError(f"Too few good matches after ratio test: {len(good)}")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts4 = np.float32([k4[m.trainIdx].pt for m in good])
    dists = np.float32([m.distance for m in good])

    return k1, k4, good, pts1, pts4, dists

def estimate_global_affine(pts1, pts4):
    # Estimate a robust affine transform from img4 -> img1
    # This helps eliminate small rotation/vertical offsets before measuring disparity.
    M, inliers = cv2.estimateAffinePartial2D(pts4, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None or inliers.sum() < 8:
        return None, None
    return M, inliers.ravel().astype(bool)

def pick_anchor_point(pts1, pts4, match_distances, inliers_mask=None, margin=40):
    """
    Choose a single correspondence to "wiggle on".
    Heuristic:
      - Prefer RANSAC inliers (geometrically consistent)
      - Avoid borders
      - Prefer strong match (low distance)
      - Prefer decent horizontal disparity magnitude (so wiggle is noticeable)
    """
    N = len(pts1)
    if inliers_mask is None:
        inliers_mask = np.ones(N, dtype=bool)

    # Normalize terms for scoring
    dist_norm = (match_distances - match_distances.min()) / (np.ptp(match_distances) + 1e-9)
    dx = pts4[:, 0] - pts1[:, 0]
    dx_norm = np.abs(dx) / (np.abs(dx).max() + 1e-9)

    # Score: lower is better for dist, higher is better for dx.
    # We want good match + visible disparity.
    score = (0.65 * (1.0 - dist_norm)) + (0.35 * dx_norm)

    # Apply inliers preference
    score = score + 0.20 * inliers_mask.astype(np.float32)  # small bonus for inliers

    # Border penalty (we don’t know image size here; caller should filter with size)
    # We'll just return best index and let caller optionally re-check.
    best_idx = int(np.argmax(score))
    return best_idx

def shift_image(bgr, dx, dy=0.0):
    h, w = bgr.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def common_crop_region(shifts, w, h, pad=0):
    """
    Given shifts applied to each frame (dx, dy), compute a safe overlap crop.
    We assume images are same size.
    """
    dxs = np.array([s[0] for s in shifts], dtype=float)
    dys = np.array([s[1] for s in shifts], dtype=float)

    # After shifting, pixels at edges can become "new". We crop to intersection.
    left = int(np.ceil(max(0, dxs.max()) + pad))
    right = int(np.floor(w - max(0, -dxs.min()) - pad))
    top = int(np.ceil(max(0, dys.max()) + pad))
    bottom = int(np.floor(h - max(0, -dys.min()) - pad))

    left = max(left, 0); top = max(top, 0)
    right = min(right, w); bottom = min(bottom, h)

    if right - left < 10 or bottom - top < 10:
        raise RuntimeError("Crop region too small. Reduce pad or check shifts/rig alignment.")
    return left, top, right, bottom

def make_wigglegram(paths, out_gif="wiggle.gif", fps=8, crop_pad=2, force_horizontal=True):
    if len(paths) != 4:
        raise ValueError("This script expects exactly 4 images (one per camera).")

    imgs = [read_bgr(p) for p in paths]
    h, w = imgs[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in imgs):
        raise ValueError("All images must have the same dimensions.")

    g1 = to_gray(imgs[0])
    g4 = to_gray(imgs[3])

    k1, k4, good, pts1, pts4, dists = detect_and_match(g1, g4)

    # Optional global affine rectify (img4 -> img1)
    M_aff, inliers = estimate_global_affine(pts1, pts4)
    if M_aff is not None:
        imgs[3] = cv2.warpAffine(imgs[3], M_aff, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        g4r = to_gray(imgs[3])

        # Re-match after rectification for more accurate anchor disparity
        k1, k4, good, pts1, pts4, dists = detect_and_match(g1, g4r)
        M_aff2, inliers2 = estimate_global_affine(pts1, pts4)
        inliers = inliers2 if inliers2 is not None else None
    else:
        inliers = None

    # Pick anchor
    best_idx = pick_anchor_point(pts1, pts4, dists, inliers_mask=inliers)

    x1, y1 = pts1[best_idx]
    x4, y4 = pts4[best_idx]

    dx_total = float(x4 - x1)
    dy_total = float(y4 - y1)

    if force_horizontal:
        dy_total = 0.0

    dx_step = dx_total / 3.0
    dy_step = dy_total / 3.0

    # Compute shifts to apply to each frame so anchor aligns to frame 0’s anchor
    # Frame 0 shift: 0
    # Frame i shift: -i * step
    shifts = [(0.0, 0.0)]
    for i in range(1, 4):
        shifts.append((-i * dx_step, -i * dy_step))

    shifted = []
    for im, (dx, dy) in zip(imgs, shifts):
        shifted.append(shift_image(im, dx, dy))

    # Crop to common overlap
    l, t, r, b = common_crop_region(shifts, w, h, pad=crop_pad)
    shifted = [im[t:b, l:r] for im in shifted]

    # Wiggle order: forward then backward (without repeating endpoints)
    seq = shifted + shifted[-2:0:-1]

    # Write GIF
    rgb_seq = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in seq]
    imageio.mimsave(out_gif, rgb_seq, duration=1.0 / fps)

    return {
        "out_gif": out_gif,
        "anchor_img1": (x1, y1),
        "anchor_img4": (x4, y4),
        "dx_total": dx_total,
        "dy_total": dy_total,
        "dx_step": dx_step,
        "dy_step": dy_step,
        "crop": (l, t, r, b),
    }

if __name__ == "__main__":
    path = "input/oil"
    files = sorted(os.listdir(path))
    image_files = [
        f"{path}/{f}"
        for f in files
        if f.lower().endswith((".jpg", ".jpeg"))
    ]
    info = make_wigglegram(image_files, out_gif="wiggle.gif", fps=10, crop_pad=4, force_horizontal=True)
    print("Done:", info)