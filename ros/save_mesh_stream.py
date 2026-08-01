#!/usr/bin/env python3
"""Save the current nvblox mesh to PLY by draining the mesh *stream*.

Why this exists: the /nvblox_node/save_ply service makes nvblox serialize the
entire mesh into one giant GPU allocation — on the 8 GB Orin Nano that OOM-
crashed the whole SLAM container (and lost the map). This tool instead
subscribes to /nvblox_node/mesh: nvblox greets a new subscriber by re-sending
the full map block by block (bandwidth-limited), so memory use stays flat and
the mapping process is never at risk.

Run inside the container while the pipeline is up:

    docker exec slam python3 /workspace/ros/save_mesh_stream.py [name]

Writes /workspace/maps/<name>.ply (= ~/SLAM/maps/ on the Jetson). Waits for
the map replay to finish (no new blocks for --quiet seconds), then writes.
"""

import argparse
import os
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from nvblox_msgs.msg import Mesh


class MeshCollector(Node):
    def __init__(self, quiet_s):
        super().__init__("mesh_saver")
        self.blocks = {}          # (x,y,z) index -> (vertices, triangles)
        self.last_msg = time.monotonic()
        self.quiet_s = quiet_s
        self.create_subscription(Mesh, "/nvblox_node/mesh", self.on_mesh, 10)

    def on_mesh(self, msg):
        self.last_msg = time.monotonic()
        for idx, block in zip(msg.block_indices, msg.blocks):
            key = (idx.x, idx.y, idx.z)
            verts = [(v.x, v.y, v.z) for v in block.vertices]
            tris = list(block.triangles)
            if verts:
                self.blocks[key] = (verts, tris)
            else:
                self.blocks.pop(key, None)   # empty block = deleted

    def done(self):
        return time.monotonic() - self.last_msg > self.quiet_s

    def write_ply(self, path):
        n_vert = sum(len(v) for v, _ in self.blocks.values())
        n_tri = sum(len(t) // 3 for _, t in self.blocks.values())
        with open(path, "wb") as f:
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n_vert}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {n_tri}\n"
                "property list uchar int vertex_indices\nend_header\n"
            ).encode())
            for verts, _ in self.blocks.values():
                for v in verts:
                    f.write(struct.pack("<fff", *v))
            base = 0
            for verts, tris in self.blocks.values():
                for i in range(0, len(tris) - 2, 3):
                    f.write(struct.pack("<Biii", 3, base + tris[i],
                                        base + tris[i + 1], base + tris[i + 2]))
                base += len(verts)
        return n_vert, n_tri


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?",
                   default=time.strftime("room_%Y%m%d_%H%M%S"))
    p.add_argument("--quiet", type=float, default=5.0,
                   help="seconds without new mesh blocks = map fully received")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    rclpy.init()
    node = MeshCollector(args.quiet)
    print("collecting mesh stream (nvblox re-sends the full map to a new "
          "subscriber)...", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.5)
        if node.blocks and node.done():
            break
    if not node.blocks:
        print("ERROR: no mesh received — is the pipeline running and mapping?")
        sys.exit(1)

    os.makedirs("/workspace/maps", exist_ok=True)
    path = f"/workspace/maps/{args.name}.ply"
    n_vert, n_tri = node.write_ply(path)
    print(f"saved {path}: {n_vert} vertices, {n_tri} triangles")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
