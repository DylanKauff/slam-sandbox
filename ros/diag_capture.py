#!/usr/bin/env python3
"""Grab one live rectified stereo pair + the depth frame nvblox integrates,
save them as viewable PNGs so we can see WHICH stage is broken.

  - left_rect.png / right_rect.png : what cuVSLAM + SGBM actually see
  - epipolar.png    : the two rectified, stacked, with rows drawn. Matching
                      features MUST sit on the same row or depth is garbage.
  - depth_color.png : the depth image nvblox fuses, colourised (near=red)
  - depth_stats.txt : coverage %, min/max/median depth in metres
"""
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

OUT = "/workspace/maps/diag"
import os; os.makedirs(OUT, exist_ok=True)


def img_to_np(msg):
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("mono8", "8UC1"):
        return buf.reshape(msg.height, msg.width)
    if msg.encoding in ("32FC1",):
        return np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1",):
        return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
    return buf.reshape(msg.height, -1)


class Grab(Node):
    def __init__(self):
        super().__init__("diag_grab")
        self.l = self.r = self.d = None
        self.create_subscription(Image, "/stereo/left/image_rect", self._l, qos_profile_sensor_data)
        self.create_subscription(Image, "/stereo/right/image_rect", self._r, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera_0/depth/image", self._d, qos_profile_sensor_data)
    def _l(self, m): self.l = img_to_np(m)
    def _r(self, m): self.r = img_to_np(m)
    def _d(self, m): self.d = (img_to_np(m), m.encoding, m.width, m.height)


def main():
    rclpy.init()
    n = Grab()
    import time; t0 = time.time()
    while time.time() - t0 < 15 and (n.l is None or n.r is None or n.d is None):
        rclpy.spin_once(n, timeout_sec=0.3)
    got = []
    if n.l is not None and n.r is not None:
        cv2.imwrite(f"{OUT}/left_rect.png", n.l)
        cv2.imwrite(f"{OUT}/right_rect.png", n.r)
        h = min(n.l.shape[0], n.r.shape[0]); w = min(n.l.shape[1], n.r.shape[1])
        vis = cv2.cvtColor(np.hstack([n.l[:h, :w], n.r[:h, :w]]), cv2.COLOR_GRAY2BGR)
        for y in range(0, h, 30):
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 200, 0), 1)
        cv2.imwrite(f"{OUT}/epipolar.png", vis)
        got.append("rect pair + epipolar")
    else:
        got.append("NO rectified images received")
    if n.d is not None:
        depth, enc, w, hh = n.d
        depth = depth.astype(np.float32)
        # 16UC1 depth is usually millimetres
        if enc == "16UC1":
            depth = depth / 1000.0
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20)
        cov = 100.0 * valid.sum() / depth.size
        vals = depth[valid]
        with open(f"{OUT}/depth_stats.txt", "w") as f:
            f.write(f"encoding={enc} size={w}x{hh}\n")
            f.write(f"valid coverage={cov:.1f}%\n")
            if vals.size:
                f.write(f"depth min={vals.min():.2f}m  median={np.median(vals):.2f}m  max={vals.max():.2f}m\n")
        vis = np.zeros(depth.shape, np.uint8)
        if vals.size:
            lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
            norm = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
            vis = (norm * 255).astype(np.uint8)
        cm = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cm[~valid] = (0, 0, 0)
        cv2.imwrite(f"{OUT}/depth_color.png", cm)
        got.append(f"depth {enc} cov={cov:.1f}%")
    else:
        got.append("NO depth received")
    print("DIAG:", " | ".join(got))
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
