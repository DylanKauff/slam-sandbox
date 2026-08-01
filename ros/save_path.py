#!/usr/bin/env python3
"""Save the SLAM trajectory — where the rig actually walked — to JSON.

cuVSLAM already publishes the tracked path on /visual_slam/tracking/slam_path
(it's in foxglove_layout.json), but nothing ever persisted it, so a saved map
had no record of the walk that produced it.

Run it alongside the map save, in the SAME session — the poses are only
meaningful in that run's `odom` frame, so a path from one run cannot be drawn
against a map from another:

    docker exec slam python3 /workspace/ros/save_path.py [name]

Writes /workspace/maps/<name>.path.json. Feed it to make_map_viewer.py with
--path to draw the trajectory through the point cloud:

    python3 make_map_viewer.py maps/room.ply maps/room.html "Room scan" \
        --path maps/room.path.json

slam_path is latched-ish: the message carries the whole trajectory so far, so
we just keep the largest one seen rather than accumulating.
"""

import argparse
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path


class PathCollector(Node):
    def __init__(self):
        super().__init__("path_saver")
        self.poses = []
        self.last_msg = 0.0
        self.create_subscription(
            Path, "/visual_slam/tracking/slam_path", self.on_path, 10)

    def on_path(self, msg):
        # each message is the full path so far; keep the longest one
        if len(msg.poses) >= len(self.poses):
            self.poses = [(p.pose.position.x, p.pose.position.y,
                           p.pose.position.z) for p in msg.poses]
            self.last_msg = time.monotonic()

    def done(self, quiet_s):
        return self.poses and time.monotonic() - self.last_msg > quiet_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?",
                   default=time.strftime("room_%Y%m%d_%H%M%S"))
    p.add_argument("--quiet", type=float, default=3.0,
                   help="seconds without a longer path = trajectory complete")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    rclpy.init()
    node = PathCollector()
    print("waiting for /visual_slam/tracking/slam_path ...", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.5)
        if node.done(args.quiet):
            break

    if not node.poses:
        print("ERROR: no path received — is cuVSLAM running and tracking?")
        sys.exit(1)

    os.makedirs("/workspace/maps", exist_ok=True)
    out = f"/workspace/maps/{args.name}.path.json"
    with open(out, "w") as f:
        json.dump({"frame": "odom", "poses": node.poses}, f)

    d = sum(
        sum((b[k] - a[k]) ** 2 for k in range(3)) ** 0.5
        for a, b in zip(node.poses, node.poses[1:])
    )
    print(f"saved {out}: {len(node.poses)} poses, {d:.1f} m walked")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
