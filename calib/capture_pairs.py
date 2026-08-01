#!/usr/bin/env python3
"""Capture synchronized stereo image pairs for calibration.

Hold a printed checkerboard in front of the rig and move it around so it appears
at many positions / angles / distances and near the frame edges (that is where
lens distortion is strongest, so edge coverage matters most).

This tool only keeps a pair when the board is detected in BOTH eyes, so every
saved pair is usable by calibrate_stereo.py.

Controls (display window must be focused):
    SPACE / c   capture the current pair (only works when both boards detected)
    a           toggle auto-capture (grabs a good pair every ~1s while you move)
    u           undo last capture
    q / ESC     quit

Output: calib/captures/left_000.png, right_000.png, ...

Defaults assume a 9x6 *inner-corner* checkerboard (a 10x7 square board).
Measure your printed square size in mm and pass --square.
"""

import argparse
import os
import time

import cv2
import numpy as np

# Make the sibling stereo_source importable whether run from SLAM/ or SLAM/calib/.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_source import StereoSource  # noqa: E402


def find_board(gray, cols, rows):
    """Return corner array if the full checkerboard is found, else None."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)


def main():
    p = argparse.ArgumentParser(description="Capture stereo calibration pairs")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--left", type=int, default=0, help="left camera column 0..3")
    p.add_argument("--right", type=int, default=1, help="right camera column 0..3")
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square", type=float, default=25.0, help="square size (mm)")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "captures"))
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Stash board geometry next to the images so calibrate_stereo can read it back.
    np.savez(os.path.join(args.outdir, "board.npz"),
             cols=args.cols, rows=args.rows, square=args.square)

    src = StereoSource(device=args.device, left=args.left, right=args.right)
    print(f"Board: {args.cols}x{args.rows} inner corners, {args.square} mm squares")
    print("Move the board around; capture when BOTH halves show colored corners.")

    count = 0
    auto = False
    last_auto = 0.0
    try:
        while True:
            ts, left, right = src.read()
            if left is None:
                print("frame read failed")
                break

            cl = find_board(left, args.cols, args.rows)
            cr = find_board(right, args.cols, args.rows)
            both = cl is not None and cr is not None

            disp = cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_GRAY2BGR)
            w = left.shape[1]
            if cl is not None:
                cv2.drawChessboardCorners(disp[:, :w], (args.cols, args.rows), cl, True)
            if cr is not None:
                cv2.drawChessboardCorners(disp[:, w:], (args.cols, args.rows), cr, True)

            status = "BOTH OK - capture!" if both else "move board into both views"
            color = (0, 220, 0) if both else (0, 140, 255)
            cv2.putText(disp, f"saved: {count}   {status}   auto:{'ON' if auto else 'off'}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            scale = 1600.0 / disp.shape[1]
            cv2.imshow("calib capture  (SPACE=save a=auto u=undo q=quit)",
                       cv2.resize(disp, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF

            do_save = False
            if key in (ord(' '), ord('c')):
                do_save = both
                if not both:
                    print("  no board in both views — not saved")
            elif key == ord('a'):
                auto = not auto
            elif key == ord('u') and count > 0:
                count -= 1
                for side in ("left", "right"):
                    f = os.path.join(args.outdir, f"{side}_{count:03d}.png")
                    if os.path.exists(f):
                        os.remove(f)
                print(f"  undid capture {count}")
            elif key in (ord('q'), 27):
                break

            if auto and both and (time.time() - last_auto) > 1.0:
                do_save = True
                last_auto = time.time()

            if do_save:
                cv2.imwrite(os.path.join(args.outdir, f"left_{count:03d}.png"), left)
                cv2.imwrite(os.path.join(args.outdir, f"right_{count:03d}.png"), right)
                print(f"  saved pair {count}")
                count += 1
    finally:
        src.release()
        cv2.destroyAllWindows()

    print(f"\nDone. {count} pairs in {args.outdir}")
    if count < 20:
        print("WARNING: aim for 30-50 pairs with varied board positions for a good calibration.")
    else:
        print("Next: python3 calib/calibrate_stereo.py")


if __name__ == "__main__":
    main()
