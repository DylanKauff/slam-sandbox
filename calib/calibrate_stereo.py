#!/usr/bin/env python3
"""Solve stereo calibration from pairs captured by capture_pairs.py.

Pipeline:
  1. detect checkerboard corners in every saved left/right pair
  2. calibrate each camera individually (intrinsics + distortion)
  3. stereoCalibrate (fix intrinsics) -> relative rotation R, translation T
  4. stereoRectify -> rectification rotations, projection matrices, Q, maps
  5. write left.yaml / right.yaml (ROS camera_info) + rectify.npz
  6. write epipolar_check.png so you can visually confirm row alignment

Run:
    python3 calib/calibrate_stereo.py
    python3 calib/calibrate_stereo.py --captures calib/captures --out calib
"""

import argparse
import glob
import os

import cv2
import numpy as np


def load_board(captures):
    b = np.load(os.path.join(captures, "board.npz"))
    cols, rows, square = int(b["cols"]), int(b["rows"]), float(b["square"])
    # Object points: (0,0,0),(square,0,0)... in mm so T/baseline come out in mm.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return cols, rows, square, objp


def detect(path, cols, rows):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # findChessboardCornersSB: subpixel-accurate out of the box and far faster
    # than the classic detector (which needs ~1 s/frame on the Jetson CPU).
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows),
                                              cv2.CALIB_CB_ACCURACY)
    if ok:
        return gray, corners.astype(np.float32)
    # Fallback for frames SB rejects (harsh lighting, partial blur).
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return gray, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return gray, corners


def camera_info_yaml(name, w, h, K, D, R, P):
    """Emit a ROS camera_info YAML (consumed by the Phase 2 ROS bridge)."""
    def mat(m):
        return "[" + ", ".join(f"{v:.10g}" for v in np.asarray(m).flatten()) + "]"
    return f"""image_width: {w}
image_height: {h}
camera_name: {name}
camera_matrix:
  rows: 3
  cols: 3
  data: {mat(K)}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {np.asarray(D).size}
  data: {mat(D)}
rectification_matrix:
  rows: 3
  cols: 3
  data: {mat(R)}
projection_matrix:
  rows: 3
  cols: 4
  data: {mat(P)}
"""


def main():
    p = argparse.ArgumentParser(description="Solve stereo calibration")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--captures", default=os.path.join(here, "captures"))
    p.add_argument("--out", default=here)
    p.add_argument("--swap", action="store_true",
                   help="treat the saved right_* images as the LEFT eye (use when "
                        "a previous run warned that T[0] > 0)")
    p.add_argument("--rational", action="store_true",
                   help="use the 8-coeff rational distortion model. Default is the "
                        "5-coeff model: the rational one needs strong edge/corner "
                        "coverage or it overfits and the stereo solve diverges.")
    args = p.parse_args()

    cols, rows, square, objp = load_board(args.captures)
    lefts = sorted(glob.glob(os.path.join(args.captures, "left_*.png")))
    if not lefts:
        raise SystemExit(f"No captures in {args.captures}. Run capture_pairs.py first.")

    objpoints, lpts, rpts = [], [], []
    img_size = None
    used = 0
    if args.swap:
        print("--swap: saved right_* images are the LEFT eye")
    for lf in lefts:
        rf = lf.replace("left_", "right_")
        if args.swap:
            lf, rf = rf, lf
        if not os.path.exists(rf):
            continue
        lg, lc = detect(lf, cols, rows)
        rg, rc = detect(rf, cols, rows)
        if lc is None or rc is None:
            print(f"  skip {os.path.basename(lf)} (board not found in both)")
            continue
        img_size = (lg.shape[1], lg.shape[0])
        objpoints.append(objp)
        lpts.append(lc)
        rpts.append(rc)
        used += 1

    print(f"Using {used}/{len(lefts)} pairs for calibration.")
    if used < 8:
        raise SystemExit("Too few valid pairs (<8). Capture more, varied positions.")

    flags_mono = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
    rms_l, Kl, Dl, *_ = cv2.calibrateCamera(objpoints, lpts, img_size, None, None, flags=flags_mono)
    rms_r, Kr, Dr, *_ = cv2.calibrateCamera(objpoints, rpts, img_size, None, None, flags=flags_mono)
    print(f"Per-camera RMS reprojection error:  left={rms_l:.3f}px  right={rms_r:.3f}px")

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    # Solve with per-view outlier rejection: a single bad frame (blurred board,
    # detector glitch) can pull the stereo solve from <1px to >20px RMS, so we
    # iteratively drop views whose reprojection error is far off the median.
    keep = list(range(len(objpoints)))
    for it in range(3):
        op = [objpoints[i] for i in keep]
        lp = [lpts[i] for i in keep]
        rp = [rpts[i] for i in keep]
        rms_s, Kl, Dl, Kr, Dr, R, T, E, F, *rest = cv2.stereoCalibrateExtended(
            op, lp, rp, Kl, Dl, Kr, Dr, img_size, np.eye(3), np.zeros(3),
            flags=cv2.CALIB_FIX_INTRINSIC, criteria=term)
        perr = rest[-1]  # perViewErrors is the last return value on cv 4.x
        view_err = np.asarray(perr).max(axis=1)
        thresh = max(2.0 * float(np.median(view_err)), 1.5)
        bad = [keep[i] for i in range(len(keep)) if view_err[i] > thresh]
        if not bad:
            break
        for i in range(len(keep)):
            if view_err[i] > thresh:
                print(f"  dropping outlier view {keep[i]} "
                      f"(per-view err {view_err[i]:.2f}px > {thresh:.2f})")
        keep = [k for k in keep if k not in bad]
        if len(keep) < 8:
            raise SystemExit("Too few views left after outlier rejection.")
    print(f"Views used after outlier rejection: {len(keep)}/{len(objpoints)}")
    baseline = float(np.linalg.norm(T))
    print(f"Stereo RMS reprojection error: {rms_s:.3f}px")
    print(f"Baseline: {baseline:.1f} mm  ({baseline/10:.1f} cm)")
    if rms_s > 0.5:
        print("WARNING: stereo RMS > 0.5px — recapture with better edge coverage / sharper images.")

    # Handedness check. T is the right camera's position in the left frame; for a
    # standard left|right rig the right eye sits at negative x, so T[0] < 0. If it
    # is positive, the left/right eyes are swapped (common on the upside-down rig)
    # — rectification still 'works' but disparities go negative and downstream
    # stereo/SLAM expect the standard convention.
    if float(T[0]) > 0:
        print("WARNING: T[0] > 0 — left/right eyes look SWAPPED. Recapture (or "
              "re-run) with the two camera columns exchanged so disparity is positive.")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Kl, Dl, Kr, Dr, img_size, R, T, alpha=0,
        flags=cv2.CALIB_ZERO_DISPARITY)

    m1x, m1y = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "rectify.npz"),
             map1x=m1x, map1y=m1y, map2x=m2x, map2y=m2y,
             Q=Q, baseline_mm=baseline, img_size=np.array(img_size),
             Kl=Kl, Dl=Dl, Kr=Kr, Dr=Dr, R=R, T=T, P1=P1, P2=P2)

    w, h = img_size
    with open(os.path.join(args.out, "left.yaml"), "w") as f:
        f.write(camera_info_yaml("left", w, h, Kl, Dl, R1, P1))
    with open(os.path.join(args.out, "right.yaml"), "w") as f:
        f.write(camera_info_yaml("right", w, h, Kr, Dr, R2, P2))

    # Visual epipolar check: rectify the first pair, draw rows. Same scene point
    # should land on the same row in both halves.
    lf = lefts[0]
    rf = lf.replace("left_", "right_")
    if args.swap:
        lf, rf = rf, lf
    lg = cv2.remap(cv2.imread(lf, 0), m1x, m1y, cv2.INTER_LINEAR)
    rg = cv2.remap(cv2.imread(rf, 0), m2x, m2y, cv2.INTER_LINEAR)
    vis = cv2.cvtColor(np.hstack([lg, rg]), cv2.COLOR_GRAY2BGR)
    for y in range(0, vis.shape[0], 30):
        cv2.line(vis, (0, y), (vis.shape[1], y), (0, 200, 0), 1)
    cv2.imwrite(os.path.join(args.out, "epipolar_check.png"), vis)

    print(f"\nWrote: rectify.npz, left.yaml, right.yaml, epipolar_check.png  -> {args.out}")
    print("Open epipolar_check.png: matching features should sit on the SAME green line.")
    print("Then verify live:  python3 stereo_source.py --rectify calib/rectify.npz")


if __name__ == "__main__":
    main()
