#!/usr/bin/env python3
"""ROS 2 bridge for the Arducam CamArray HAT -> rectified stereo topics.

Runs inside the slam:latest container (rclpy comes from ros-humble-ros-base).
Wraps StereoSource (../stereo_source.py) for camera I/O, applies the Pair A
rectification from calib/pairA/rectify.npz, and publishes what cuVSLAM,
the SGM depth pipeline and nvblox expect:

    /stereo/left/image_rect        sensor_msgs/Image   mono8, 1280x800
    /stereo/left/camera_info       sensor_msgs/CameraInfo
    /stereo/right/image_rect       sensor_msgs/Image
    /stereo/right/camera_info      sensor_msgs/CameraInfo
    /stereo/preview/compressed     sensor_msgs/CompressedImage (left eye, ~4 fps)
    /camera_0/depth/image          sensor_msgs/Image 32FC1 metres, 640x400 ~5Hz
    /camera_0/depth/camera_info    sensor_msgs/CameraInfo (half-res intrinsics)

Depth is CPU SGBM at half resolution: the GPU (SGM via NITROS) route ran out
of memory on the 8GB board next to cuVSLAM + nvblox, and ~5 Hz depth is plenty
for nvblox at 5 cm voxels. Topic names match nvblox's camera_0 inputs directly.

plus static TF:  base_link -> camera_left_optical -> camera_right_optical.

Conventions handled here so nothing downstream has to care:
  * calibration was solved in millimetres -> P_right[0,3] and the TF baseline
    are converted to metres (ROS standard)
  * images are already rectified -> camera_info carries D=0, R=I, K=P[:3,:3]
  * one timestamp per pair (cameras are hardware-synced on the HAT)

Run:
    python3 /workspace/ros/arducam_bridge.py            # defaults = Pair A
    python3 /workspace/ros/arducam_bridge.py --fps 15
"""

import argparse
import os
import struct
import subprocess
import sys
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_source import StereoSource  # noqa: E402

try:
    import vpi  # host JetPack bindings, mounted by docker/run_slam.sh
except ImportError:
    vpi = None

# Pair A geometry (calibrated 2026-07-02): cam2 = left eye, cam1 = right eye,
# cameras mounted upside-down.
DEFAULT_LEFT_CAM = 2
DEFAULT_RIGHT_CAM = 1


# --- raw CDR serialization for sensor_msgs/Image -----------------------------
# rclpy's python->C message conversion needs ~165 ms for a 1 MB Image on this
# CPU (~6 msgs/s), which capped the whole pipeline at ~2 fps. Building the CDR
# bytes ourselves and using Publisher.publish(bytes) is ~1000x cheaper. The
# layout is verified byte-for-byte against rclpy's serializer at startup.

def _cdr_align(buf, n):
    buf += b"\x00" * ((-len(buf)) % n)


def _cdr_string(buf, s):
    b = s.encode() + b"\x00"
    _cdr_align(buf, 4)
    buf += struct.pack("<I", len(b))
    buf += b


def cdr_image(sec, nsec, frame_id, height, width, encoding, step, data):
    """Serialized sensor_msgs/Image (CDR little-endian, 4-byte encap header)."""
    buf = bytearray(b"\x00\x01\x00\x00")
    buf += struct.pack("<iI", sec, nsec)
    _cdr_string(buf, frame_id)
    _cdr_align(buf, 4)
    buf += struct.pack("<II", height, width)
    _cdr_string(buf, encoding)
    buf += struct.pack("<B", 0)          # is_bigendian
    _cdr_align(buf, 4)
    buf += struct.pack("<II", step, len(data))
    return bytes(buf) + data


def cdr_self_test():
    from rclpy.serialization import serialize_message
    ref = Image()
    ref.header.stamp.sec = 123
    ref.header.stamp.nanosec = 456
    ref.header.frame_id = "camera_left_optical"
    ref.height, ref.width = 2, 3
    ref.encoding = "mono8"
    ref.step = 3
    ref.data = bytes(range(6))
    expect = serialize_message(ref)
    got = cdr_image(123, 456, "camera_left_optical", 2, 3, "mono8", 3,
                    bytes(range(6)))
    return got == expect


def rotmat_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


# Auto-exposure limits for this OV9281/Arducam sensor (manual-exposure only —
# no hardware AE control, so we drive exposure/analogue_gain over v4l2 ourselves).
EXP_MIN, EXP_MAX = 1, 65523
GAIN_MIN, GAIN_MAX = 100, 1590
AE_MAX_STEP = 0.35     # cap per-step change so brightness doesn't oscillate
AE_TOLERANCE = 6       # brightness units within which we leave it alone


class ArducamBridge(Node):
    def __init__(self, args):
        super().__init__("arducam_bridge")

        self.device = args.device
        d = np.load(args.calib)
        w, h = int(d["img_size"][0]), int(d["img_size"][1])
        self.size = (w, h)
        # Integer maps remap ~2x faster than the float maps in the npz.
        self.m1 = cv2.convertMaps(d["map1x"], d["map1y"], cv2.CV_16SC2)
        self.m2 = cv2.convertMaps(d["map2x"], d["map2y"], cv2.CV_16SC2)

        P1, P2 = d["P1"].copy(), d["P2"].copy()
        self.baseline_m = float(d["baseline_mm"]) / 1000.0
        # Calibration ran in mm; ROS camera_info wants the rectified projection
        # translation fx*b in metres.
        P2[0, 3] /= 1000.0

        self.info_l = self._camera_info("camera_left_optical", P1)
        self.info_r = self._camera_info("camera_right_optical", P2)

        # Depth branch. Prefer VPI SGM on the GPU: full 1280x800 at ~52 ms/frame
        # on the Orin Nano vs ~200 ms CPU SGBM at half res. Falls back to CPU
        # SGBM if the vpi module isn't mounted or its CUDA context can't
        # allocate (memory pressure on the 8 GB board).
        self.use_vpi = False
        if vpi is not None:
            try:
                z = np.zeros((h, w), np.uint8)
                with vpi.Backend.CUDA:
                    vpi.stereodisp(vpi.asimage(z), vpi.asimage(z),
                                   window=5, maxdisp=64).cpu()
                self.use_vpi = True
            except Exception as e:
                self.get_logger().warn(
                    f"VPI depth unavailable ({e}); falling back to CPU SGBM")
        self.depth_scale = 1.0 if self.use_vpi else 0.5
        Ph = P1.copy()
        Ph[:2] *= self.depth_scale
        self.info_depth = self._camera_info("camera_left_optical", Ph)
        self.info_depth.width = int(w * self.depth_scale)
        self.info_depth.height = int(h * self.depth_scale)
        self.fxb_depth = float(Ph[0, 0]) * self.baseline_m
        if not self.use_vpi:
            self.sgbm = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=64, blockSize=7,
                P1=8 * 49, P2=32 * 49, uniquenessRatio=10,
                speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                mode=cv2.STEREO_SGBM_MODE_SGBM)
        self.get_logger().info(
            f"depth backend: {'VPI SGM (GPU, full res)' if self.use_vpi else 'CPU SGBM (half res)'}")

        # "Only integrate sure points" knobs (env-tunable, no re-edit for A/B).
        # Defaults OFF => identical to prior behaviour.
        #   VPI_CONF_MIN   : drop disparity where VPI confidence < this (0-65535,
        #                    0 disables). Higher = fewer but more certain points.
        #   VPI_UNIQUENESS : reject ambiguous matches (blank walls / repeats),
        #                    [0,1]; -1 disables. ~0.7 is a good starting point.
        self.conf_min = int(os.environ.get("VPI_CONF_MIN", "0"))
        self.vpi_uniq = float(os.environ.get("VPI_UNIQUENESS", "-1.0"))
        # SGM smoothness penalties. Higher p2 propagates depth across low-texture
        # surfaces (fills flat chair backs / walls) without relaxing uniqueness.
        self.vpi_p1 = int(os.environ.get("VPI_P1", "3"))
        self.vpi_p2 = int(os.environ.get("VPI_P2", "48"))
        # Census window: larger = more context to match faint low-texture
        # gradients (corner shading), at the cost of some fine-edge sharpness.
        self.vpi_window = int(os.environ.get("VPI_WINDOW", "5"))
        # Edge-preserving depth filter (guided by the rectified left image):
        # smooths on-surface grain but keeps corners / object boundaries crisp.
        self.depth_filter = os.environ.get("DEPTH_FILTER", "0") == "1"
        self.df_r = int(os.environ.get("DEPTH_FILTER_R", "8"))
        self.df_eps = float(os.environ.get("DEPTH_FILTER_EPS", "500"))
        self._vpi_confmap = None
        self._cov_ctr = 0
        if self.depth_filter:
            self.get_logger().info(
                f"edge-preserving depth filter ON (guided r={self.df_r} eps={self.df_eps})")
        if self.use_vpi and (self.conf_min > 0 or self.vpi_uniq >= 0.0):
            self.get_logger().info(
                f"depth confidence gating ON: conf_min={self.conf_min} "
                f"uniqueness={self.vpi_uniq}")

        qos = qos_profile_sensor_data
        self.pub_il = self.create_publisher(Image, "/stereo/left/image_rect", qos)
        self.pub_ir = self.create_publisher(Image, "/stereo/right/image_rect", qos)
        self.pub_cl = self.create_publisher(CameraInfo, "/stereo/left/camera_info", qos)
        self.pub_cr = self.create_publisher(CameraInfo, "/stereo/right/camera_info", qos)
        self.pub_prev = self.create_publisher(CompressedImage, "/stereo/preview/compressed", 1)
        self.pub_depth = self.create_publisher(Image, "/camera_0/depth/image", qos)
        self.pub_dinfo = self.create_publisher(CameraInfo, "/camera_0/depth/camera_info", qos)

        self._publish_static_tf()

        self.src = StereoSource(device=args.device,
                                left=args.left_cam, right=args.right_cam,
                                rectify_file=None, rotate180=True,
                                use_gstreamer=not args.no_gstreamer,
                                backend="v4l2pipe")
        self.min_period = 1.0 / args.fps if args.fps > 0 else 0.0
        self.preview_period = 0.1   # 10 fps JPEG preview (~1 MB/s on the wire)
        self.get_logger().info(
            f"publishing rectified pair A ({w}x{h}), baseline "
            f"{self.baseline_m * 1000:.1f} mm, fps cap "
            f"{args.fps if args.fps > 0 else 'none'}")

        if not cdr_self_test():
            raise RuntimeError("CDR self-test failed: raw Image serialization "
                               "does not match rclpy's — refusing to publish garbage")
        self.get_logger().info("raw CDR image path verified against rclpy serializer")

        # Auto-exposure: the sensor was streaming at a FIXED exposure with gain
        # pinned to minimum, blowing out lit surfaces and crushing shadows ->
        # no texture -> stereo matching failed -> holey, speckled depth. Meter
        # the raw frame and drive exposure/gain to a brightness target so there
        # is texture everywhere for SGBM to match.
        self.ae_on = not args.no_auto_exposure
        self.ae_target = args.ae_target
        self.ae_interval = 1.5
        self._ae_brightness = None
        self._ae_meter_ctr = 0
        if self.ae_on:
            self.exposure = self._v4l2_get("exposure") or 681
            self.gain = self._v4l2_get("analogue_gain") or GAIN_MIN
            self.get_logger().info(
                f"auto-exposure ON (target 70th-pct brightness {self.ae_target}, "
                f"start exp={self.exposure} gain={self.gain})")

        self._stop = threading.Event()
        self._depth_lock = threading.Lock()
        self._depth_job = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.depth_thread = threading.Thread(target=self._depth_loop, daemon=True)
        self.depth_thread.start()
        if self.ae_on:
            self.ae_thread = threading.Thread(target=self._ae_loop, daemon=True)
            self.ae_thread.start()

    # -- auto-exposure ------------------------------------------------------

    def _v4l2_set(self, ctrl, value):
        try:
            subprocess.run(["v4l2-ctl", f"-d/dev/video{self.device}",
                            f"--set-ctrl={ctrl}={value}"],
                           check=True, capture_output=True)
        except Exception as e:
            self.get_logger().warn(f"v4l2 set {ctrl} failed: {e}")

    def _v4l2_get(self, ctrl):
        try:
            r = subprocess.run(["v4l2-ctl", f"-d/dev/video{self.device}",
                                f"--get-ctrl={ctrl}"],
                               check=True, capture_output=True, text=True)
            return int(r.stdout.strip().split(":")[-1])
        except Exception:
            return None

    def _ae_loop(self):
        import time
        time.sleep(2.0)   # let capture + metering warm up
        while not self._stop.is_set() and rclpy.ok():
            try:
                self._ae_step()
            except Exception as e:
                self.get_logger().warn(f"AE step error: {e}")
            time.sleep(self.ae_interval)

    def _ae_step(self):
        b = self._ae_brightness
        if b is None or abs(self.ae_target - b) <= AE_TOLERANCE:
            return
        ratio = float(np.clip(self.ae_target / max(b, 1.0),
                              1.0 - AE_MAX_STEP, 1.0 + AE_MAX_STEP))
        new_exp = int(np.clip(self.exposure * ratio, EXP_MIN, EXP_MAX))
        if new_exp == self.exposure:
            # exposure already at a rail -> trade on gain instead
            new_gain = int(np.clip(self.gain * ratio, GAIN_MIN, GAIN_MAX))
            if new_gain != self.gain:
                self.gain = new_gain
                self._v4l2_set("analogue_gain", self.gain)
        else:
            self.exposure = new_exp
            self._v4l2_set("exposure", self.exposure)
            # keep gain low (less noise) while exposure has headroom
            if self.gain != GAIN_MIN and self.exposure < EXP_MAX * 0.8:
                self.gain = GAIN_MIN
                self._v4l2_set("analogue_gain", self.gain)

    def _camera_info(self, frame_id, P):
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width, msg.height = self.size
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5                       # image_rect is undistorted
        msg.k = P[:3, :3].flatten().tolist()
        msg.r = np.eye(3).flatten().tolist()
        msg.p = P.flatten().tolist()
        return msg

    def _publish_static_tf(self):
        self.tf_static = StaticTransformBroadcaster(self)
        # base_link (x fwd, y left, z up) -> optical (x right, y down, z fwd)
        R_base_opt = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])
        qx, qy, qz, qw = rotmat_to_quat(R_base_opt)

        t0 = TransformStamped()
        t0.header.stamp = self.get_clock().now().to_msg()
        t0.header.frame_id = "base_link"
        t0.child_frame_id = "camera_left_optical"
        t0.transform.rotation.x = qx
        t0.transform.rotation.y = qy
        t0.transform.rotation.z = qz
        t0.transform.rotation.w = qw

        t1 = TransformStamped()
        t1.header.stamp = t0.header.stamp
        t1.header.frame_id = "camera_left_optical"
        t1.child_frame_id = "camera_right_optical"
        # After rectification the right eye sits at +baseline on the left
        # camera's x axis, orientations identical.
        t1.transform.translation.x = self.baseline_m
        t1.transform.rotation.w = 1.0

        self.tf_static.sendTransform([t0, t1])

    def _image_raw(self, frame, stamp, frame_id, encoding="mono8", bpp=1):
        return cdr_image(stamp.sec, stamp.nanosec, frame_id,
                         frame.shape[0], frame.shape[1], encoding,
                         frame.shape[1] * bpp, frame.tobytes())

    def _loop(self):
        import time
        last_pub = 0.0
        last_prev = 0.0
        misses = 0
        n_read = n_pub = 0
        t_rate = time.monotonic()
        while not self._stop.is_set() and rclpy.ok():
            ts, left, right = self.src.read()
            now_r = time.monotonic()
            if now_r - t_rate >= 5.0:
                self.get_logger().info(
                    f"capture {n_read / (now_r - t_rate):.1f} fps, "
                    f"publish {n_pub / (now_r - t_rate):.1f} fps")
                n_read = n_pub = 0
                t_rate = now_r
            if left is None:
                misses += 1
                if misses > 30:
                    self.get_logger().error("camera stopped delivering frames")
                    misses = 0
                continue
            misses = 0
            n_read += 1

            # Meter raw (pre-rectify) brightness for the AE thread. 70th
            # percentile of a centre crop: robust to a bright window or lamp in
            # frame, which a plain mean would chase and underexpose everything.
            if self.ae_on:
                self._ae_meter_ctr += 1
                if self._ae_meter_ctr % 10 == 0:
                    h0, w0 = left.shape[:2]
                    roi = left[int(h0 * 0.15):int(h0 * 0.85),
                               int(w0 * 0.15):int(w0 * 0.85)]
                    self._ae_brightness = float(np.percentile(roi[::4, ::4], 70))

            now = time.monotonic()
            if self.min_period and now - last_pub < self.min_period:
                continue
            last_pub = now
            n_pub += 1

            left = cv2.remap(left, self.m1[0], self.m1[1], cv2.INTER_LINEAR)
            right = cv2.remap(right, self.m2[0], self.m2[1], cv2.INTER_LINEAR)

            stamp = self.get_clock().now().to_msg()
            self.info_l.header.stamp = stamp
            self.info_r.header.stamp = stamp
            self.pub_il.publish(self._image_raw(left, stamp, "camera_left_optical"))
            self.pub_ir.publish(self._image_raw(right, stamp, "camera_right_optical"))
            self.pub_cl.publish(self.info_l)
            self.pub_cr.publish(self.info_r)

            # Hand the newest pair to the depth thread (never block the
            # capture loop: SGBM takes 100-300ms and runs at its own pace).
            with self._depth_lock:
                self._depth_job = (left, right, stamp)

            if now - last_prev >= self.preview_period:
                last_prev = now
                ok, jpg = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    prev = CompressedImage()
                    prev.header.stamp = stamp
                    prev.header.frame_id = "camera_left_optical"
                    prev.format = "jpeg"
                    prev.data = jpg.tobytes()
                    self.pub_prev.publish(prev)

    def _edge_preserve(self, depth, guide):
        """Edge-aware denoise: guided filter using the rectified left image as
        guide, normalized by a validity mask so holes (depth==0) don't bleed
        into real surfaces. Sharpens corners / object boundaries, smooths grain."""
        try:
            if guide.shape[:2] != depth.shape[:2]:
                guide = cv2.resize(guide, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_AREA)
            m = (depth > 0).astype(np.float32)
            num = cv2.ximgproc.guidedFilter(guide, depth * m, self.df_r, self.df_eps)
            den = cv2.ximgproc.guidedFilter(guide, m, self.df_r, self.df_eps)
            return np.where(den > 0.2, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f"depth filter failed ({e}); disabling")
            self.depth_filter = False
            return depth

    def _depth_loop(self):
        import time
        # Throttle depth to ~6 Hz. nvblox maps fine at this rate, and running
        # SGBM flat-out starved the image-publish loop -> cuVSLAM saw 68-600 ms
        # frame gaps and lost tracking. Depth is the low-priority consumer.
        depth_period = 1.0 / float(os.environ.get("DEPTH_HZ", "6"))
        last_depth = 0.0
        while not self._stop.is_set() and rclpy.ok():
            now = time.monotonic()
            if now - last_depth < depth_period:
                time.sleep(0.005)
                continue
            with self._depth_lock:
                job, self._depth_job = self._depth_job, None
            if job is None:
                time.sleep(0.01)
                continue
            last_depth = now
            left, right, stamp = job
            if self.use_vpi:
                try:
                    kw = dict(window=self.vpi_window, maxdisp=64,
                              p1=self.vpi_p1, p2=self.vpi_p2)
                    if self.vpi_uniq >= 0.0:
                        kw["uniqueness"] = self.vpi_uniq
                    with vpi.Backend.CUDA:
                        if self.conf_min > 0:
                            if self._vpi_confmap is None:
                                self._vpi_confmap = vpi.Image(
                                    (left.shape[1], left.shape[0]), vpi.Format.U16)
                            dv = vpi.stereodisp(vpi.asimage(left), vpi.asimage(right),
                                                out_confmap=self._vpi_confmap, **kw)
                        else:
                            dv = vpi.stereodisp(vpi.asimage(left), vpi.asimage(right), **kw)
                    disp16 = dv.cpu()
                    # kill isolated mismatches; maxDiff 32 = 1 disparity in Q10.5
                    cv2.filterSpeckles(disp16, 0, 200, 32)
                    disp = disp16.astype(np.float32) / 32.0
                    disp[disp > 512] = 0.0      # VPI invalid marker is 32767/32
                    if self.conf_min > 0:
                        conf = self._vpi_confmap.cpu()
                        disp[conf < self.conf_min] = 0.0
                        self._cov_ctr += 1
                        if self._cov_ctr % 30 == 0:
                            cov = float((disp > 0.5).mean()) * 100.0
                            p = np.percentile(conf, [10, 50, 90]).astype(int)
                            self.get_logger().info(
                                f"depth coverage {cov:.0f}% | VPI conf p10/50/90 "
                                f"{p[0]}/{p[1]}/{p[2]} (gate {self.conf_min})")
                except Exception as e:
                    # never let the confidence path kill depth: fall back to plain
                    self.get_logger().warn(
                        f"VPI confidence path failed ({e}); using plain disparity")
                    self.conf_min = 0
                    self.vpi_uniq = -1.0
                    with vpi.Backend.CUDA:
                        dv = vpi.stereodisp(vpi.asimage(left), vpi.asimage(right),
                                            window=5, maxdisp=64)
                    disp16 = dv.cpu()
                    cv2.filterSpeckles(disp16, 0, 200, 32)
                    disp = disp16.astype(np.float32) / 32.0
                    disp[disp > 512] = 0.0
            else:
                small_l = cv2.resize(left, None, fx=self.depth_scale,
                                     fy=self.depth_scale, interpolation=cv2.INTER_AREA)
                small_r = cv2.resize(right, None, fx=self.depth_scale,
                                     fy=self.depth_scale, interpolation=cv2.INTER_AREA)
                disp = self.sgbm.compute(small_l, small_r).astype(np.float32) / 16.0
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = np.where(disp > 0.5, self.fxb_depth / disp, 0.0).astype(np.float32)

            if self.depth_filter:
                depth = self._edge_preserve(depth, left)

            self.pub_depth.publish(self._image_raw(
                depth, stamp, "camera_left_optical", encoding="32FC1", bpp=4))
            self.info_depth.header.stamp = stamp
            self.pub_dinfo.publish(self.info_depth)

    def destroy_node(self):
        self._stop.set()
        self.thread.join(timeout=2.0)
        self.src.release()
        super().destroy_node()


def main():
    p = argparse.ArgumentParser(description="Arducam stereo -> ROS 2 bridge")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--calib", default=os.path.join(here, "calib/pairA/rectify.npz"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--left-cam", type=int, default=DEFAULT_LEFT_CAM)
    p.add_argument("--right-cam", type=int, default=DEFAULT_RIGHT_CAM)
    p.add_argument("--fps", type=float, default=15.0,
                   help="publish rate cap, 0 = every camera frame")
    p.add_argument("--ae-target", type=int, default=120,
                   help="auto-exposure target (70th-pct brightness, 0-255). "
                        "Lower if highlights blow out; raise if too dark.")
    p.add_argument("--no-auto-exposure", action="store_true",
                   help="disable auto-exposure (use the sensor's fixed exposure)")
    p.add_argument("--no-gstreamer", action="store_true")
    args, ros_args = p.parse_known_args()

    rclpy.init(args=ros_args)
    node = ArducamBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
