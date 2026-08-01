#!/usr/bin/env python3
"""Monocular depth prior (Depth Anything V2 small, TensorRT) — the ML support
for CORNERS.

Stereo goes blind in textureless concave corners (no disparity -> holes), which
is exactly where the map failed to close.  A monocular network predicts dense
depth *everywhere*, including those corners, from a single image.  This node
just runs the net and publishes its raw (relative, disparity-like) depth;
semantic_filter.py aligns it to the metric stereo depth per frame and fills the
stereo holes with it.

Streamlined: ViT-S at 252x392, FP16 TRT, MONO_HZ (default 2) with the result
cached between frames.  Runs on the mounted host tensorrt binding + cudart via
ctypes, same as object_seg (no torch/onnxruntime at runtime).

    /stereo/left/image_rect  ->  [DA-V2]  ->  /semantic/mono_depth (32FC1, HxW)
"""
import os
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

from object_seg import TRTEngine   # reuse the TRT+cudart wrapper

ENGINE = os.environ.get("MONO_ENGINE", "/workspace/models/depth_anything_v2_vits.engine")
MH     = int(os.environ.get("MONO_H", "252"))
MW     = int(os.environ.get("MONO_W", "392"))
MONO_HZ = float(os.environ.get("MONO_HZ", "2"))
# ImageNet normalisation (what the onnx-community DA-V2 export expects)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


class MonoDepth(Node):
    def __init__(self):
        super().__init__("mono_depth")
        self.bridge = CvBridge()
        self.get_logger().info(f"loading DA-V2 engine {ENGINE} ...")
        self.trt = TRTEngine(ENGINE)
        self.out_key = [e[0] for e in self.trt.io if not e[1]][0]
        self.in_key = [e[0] for e in self.trt.io if e[1]][0]
        self._t = 0.0
        self._tlog = 0.0

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub = self.create_publisher(Image, "/semantic/mono_depth", qos)
        self.create_subscription(Image, "/stereo/left/image_rect", self._on_img, qos)
        self.get_logger().info(f"mono_depth up: {MH}x{MW} hz={MONO_HZ}")

    def _on_img(self, msg):
        now = time.time()
        if now - self._t < 1.0 / MONO_HZ:
            return
        self._t = now
        try:
            self._process(msg)
        except Exception as e:
            self.get_logger().warn(f"mono skip: {e}")

    def _process(self, msg):
        gray = self.bridge.imgmsg_to_cv2(msg, "mono8")
        H, W = gray.shape
        # resize to (MH, MW) nearest, gray->3ch, /255, ImageNet norm
        ys = (np.arange(MH) * H / MH).astype(int).clip(0, H - 1)
        xs = (np.arange(MW) * W / MW).astype(int).clip(0, W - 1)
        small = gray[ys][:, xs].astype(np.float32) / 255.0
        blob = np.repeat(small[None], 3, axis=0)          # 3xMHxMW
        blob = (blob - _MEAN) / _STD
        blob = blob[None]                                  # 1x3xMHxMW

        out = self.trt.infer(blob)[self.out_key]
        d = np.asarray(out).reshape(MH, MW).astype(np.float32)  # relative inv-depth

        m = self.bridge.cv2_to_imgmsg(d, "32FC1")
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "camera_left_optical"
        self.pub.publish(m)
        now = time.time()
        if now - self._tlog > 5.0:
            self.get_logger().info(
                f"mono depth range [{d.min():.2f},{d.max():.2f}] std={d.std():.2f}")
            self._tlog = now


def main():
    rclpy.init()
    node = MonoDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
