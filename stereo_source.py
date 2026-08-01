#!/usr/bin/env python3
"""Shared stereo source for the Arducam CamArray HAT (4x OV9281, mono GS).

The HAT stitches up to 4 hardware-synchronized cameras side-by-side into one
V4L2 frame on /dev/video0 (GREY, 5120x800 = 4 x 1280x800).  This module splits
that frame into a chosen left/right camera pair, optionally rectifies them using
a calibration produced by calib/calibrate_stereo.py, and hands back grayscale
uint8 frames with a capture timestamp.

Every later phase (ORB-SLAM3 feeder, the ROS 2 bridge, the dataset recorder)
imports StereoSource so there is a single place that owns camera I/O + geometry.

Standalone smoke test (shows the rectified pair with a horizontal grid so you can
eyeball epipolar alignment):
    python3 stereo_source.py                 # raw split, cams 0 & 1
    python3 stereo_source.py --rectify calib/rectify.npz
    python3 stereo_source.py --left 0 --right 1 --no-display --frames 100
"""

import argparse
import time

import cv2
import numpy as np

# Stitched-frame geometry for the 4x OV9281 quad kit.
FULL_W = 5120
FULL_H = 800
NUM_CAMS = 4
CAM_W = FULL_W // NUM_CAMS          # 1280
CAM_H = FULL_H                      # 800


class StereoSource:
    """Yields rectified (or raw) grayscale stereo pairs from the CamArray HAT.

    Parameters
    ----------
    device : int
        /dev/videoX index.
    left, right : int
        Which stitched camera columns (0..NUM_CAMS-1) feed the left / right eye.
        Defaults 0 and 1 (adjacent pair).  Change to match how you physically
        mounted the rig.
    rectify_file : str or None
        Path to an .npz from calibrate_stereo.py holding map1x/map1y/map2x/map2y.
        If None, frames are returned raw (still split, but not undistorted).
    width, height : int
        Stitched capture resolution.  5120x800 is full res; 2560x400 is the
        binned mode (halves bandwidth/RAM — useful if the pipeline is tight).
    use_gstreamer : bool
        Try a GStreamer pipeline first (lower latency); fall back to plain V4L2.
    """

    def __init__(self, device=0, left=0, right=1, rectify_file=None,
                 width=FULL_W, height=FULL_H, use_gstreamer=True,
                 rotate180=False, backend="opencv"):
        self.device = device
        # backend="v4l2pipe" streams raw GREY frames from a v4l2-ctl subprocess
        # through a pipe instead of using cv2.VideoCapture. Inside the SLAM
        # container OpenCV's capture (both its GStreamer and V4L2 paths) tops
        # out at ~2 fps on the 5120x800 GREY stream, while the pipe delivers
        # the full sensor rate (~42 fps measured).
        self.backend = backend
        self.left_idx = left
        self.right_idx = right
        self.width = width
        self.height = height
        self.cam_w = width // NUM_CAMS
        self.use_gstreamer = use_gstreamer
        # Cameras on this rig are mounted upside-down; rotating each sub-image
        # 180 deg puts the scene upright BEFORE corner detection / rectification.
        # A 180 deg rotation keeps a horizontal stereo baseline horizontal, so
        # rectification still yields horizontal epipolar lines. Note: it also
        # flips disparity sign, so an upside-down pair usually also wants left
        # and right indices swapped (verify with the epipolar/disparity check).
        self.rotate180 = rotate180

        self._maps = None
        if rectify_file:
            self._load_rectify(rectify_file)

        self._pipe = None
        self.cap = None
        if self.backend == "v4l2pipe":
            self._open_pipe()
        else:
            self.cap = self._open()
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open /dev/video{device} at {width}x{height}. "
                    "Is another process (camera_stream.py / auto_exposure.py) using it?"
                )

    # -- setup ------------------------------------------------------------

    def _load_rectify(self, path):
        d = np.load(path)
        self._maps = (
            (d["map1x"], d["map1y"]),   # left
            (d["map2x"], d["map2y"]),   # right
        )
        print(f"StereoSource: loaded rectification maps from {path}")

    def _open_pipe(self):
        import subprocess
        self._frame_bytes = self.width * self.height
        self._pipe = subprocess.Popen(
            ["v4l2-ctl", "-d", f"/dev/video{self.device}",
             f"--set-fmt-video=width={self.width},height={self.height},pixelformat=GREY",
             "--stream-mmap", "--stream-count=0", "--stream-to=-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes)
        self._pipe_buf = bytearray(self._frame_bytes)
        print("StereoSource: v4l2-ctl pipe capture active")

    def _read_pipe_frame(self):
        view = memoryview(self._pipe_buf)
        got = 0
        while got < self._frame_bytes:
            r = self._pipe.stdout.readinto(view[got:])
            if not r:
                return None
            got += r
        return np.frombuffer(self._pipe_buf, dtype=np.uint8).reshape(
            self.height, self.width)

    def _open(self):
        if self.use_gstreamer:
            gst = (
                f"v4l2src device=/dev/video{self.device} ! "
                f"video/x-raw,width={self.width},height={self.height},format=GRAY8 ! "
                f"appsink drop=1 max-buffers=2 sync=false"
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("StereoSource: GStreamer pipeline active")
                return cap
            print("StereoSource: GStreamer failed, falling back to V4L2")

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)   # keep raw mono, no RGB conversion
        if cap.isOpened():
            print("StereoSource: V4L2 direct capture active")
            return cap
        return None

    # -- capture ----------------------------------------------------------

    @staticmethod
    def _to_gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _slice(self, full, idx):
        return full[:, idx * self.cam_w:(idx + 1) * self.cam_w]

    def read(self):
        """Return (timestamp, left, right) or (None, None, None) on failure.

        timestamp is a single monotonic time for the pair: the cameras are
        hardware-synced so one stamp is correct for both eyes.
        """
        if self._pipe is not None:
            full = self._read_pipe_frame()
            ts = time.monotonic()
            if full is None:
                return None, None, None
        else:
            ok, full = self.cap.read()
            ts = time.monotonic()
            if not ok or full is None:
                return None, None, None
            full = self._to_gray(full)
        left = np.ascontiguousarray(self._slice(full, self.left_idx))
        right = np.ascontiguousarray(self._slice(full, self.right_idx))

        if self.rotate180:
            left = cv2.rotate(left, cv2.ROTATE_180)
            right = cv2.rotate(right, cv2.ROTATE_180)

        if self._maps is not None:
            (m1x, m1y), (m2x, m2y) = self._maps
            left = cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR)
            right = cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR)

        return ts, left, right

    # -- lifecycle --------------------------------------------------------

    def release(self):
        if self._pipe is not None:
            self._pipe.kill()
            self._pipe = None
        elif self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _smoke_test(args):
    src = StereoSource(
        device=args.device, left=args.left, right=args.right,
        rectify_file=args.rectify, use_gstreamer=not args.no_gstreamer,
        rotate180=args.rotate180,
    )
    n, t0 = 0, time.time()
    try:
        while True:
            ts, left, right = src.read()
            if left is None:
                print("frame read failed")
                break
            n += 1

            if not args.no_display:
                pair = np.hstack([left, right])
                pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
                # Horizontal reference lines: in a good rectification the same
                # scene point sits on the same row in both halves.
                for y in range(0, pair.shape[0], 40):
                    cv2.line(pair, (0, y), (pair.shape[1], y), (0, 180, 0), 1)
                cv2.line(pair, (left.shape[1], 0),
                         (left.shape[1], pair.shape[0]), (0, 0, 255), 2)
                scale = 1280.0 / pair.shape[1]
                pair = cv2.resize(pair, None, fx=scale, fy=scale)
                cv2.imshow("StereoSource L|R", pair)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if args.frames and n >= args.frames:
                break

            if time.time() - t0 >= 1.0:
                print(f"fps: {n / (time.time() - t0):.1f}  "
                      f"L{left.shape} R{right.shape}", end="\r")
                n, t0 = 0, time.time()
    finally:
        src.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stereo source smoke test")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--left", type=int, default=0, help="left camera column 0..3")
    p.add_argument("--right", type=int, default=1, help="right camera column 0..3")
    p.add_argument("--rectify", default=None, help="path to rectify.npz")
    p.add_argument("--rotate180", action="store_true",
                   help="rotate each eye 180 deg (cameras mounted upside-down)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-gstreamer", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0=forever)")
    _smoke_test(p.parse_args())
