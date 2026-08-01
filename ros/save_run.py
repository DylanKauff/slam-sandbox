#!/usr/bin/env python3
"""Standardized run archive: drain the current nvblox map to PLY, compress it,
and append the run's config + notes + metrics to maps/RUNS.md.

The container is launched with the tuning as -e env vars, so this script reads
its own os.environ to record exactly what produced the map — no manual copying.

Run inside the container (usually via docker/save_run.sh):
    python3 /workspace/ros/save_run.py <run_name> "notes: what changed / result"
"""
import gzip
import os
import shutil
import subprocess
import sys
import time

import numpy as np

MAPS = "/workspace/maps"
# env keys that define a run (mirror docker/start_slam.sh)
CONFIG_KEYS = ["VOXEL_SIZE", "NVBLOX_MAX_INTEG_M", "MESH_MIN_WEIGHT",
               "TSDF_MAX_WEIGHT", "DECAY_HZ", "DECAY_FACTOR",
               "DECAY_EXCLUDE_LAST_VIEW", "VPI_UNIQUENESS", "VPI_CONF_MIN",
               "VPI_P1", "VPI_P2", "VPI_WINDOW", "DEPTH_FILTER", "DEPTH_HZ",
               "VSLAM_JITTER_MS", "GROUND_CONSTRAINT"]


def read_ply_verts(path):
    d = open(path, "rb").read()
    he = d.index(b"end_header\n") + len(b"end_header\n")
    nv = s = 0
    for l in d[:he].decode("ascii", "ignore").splitlines():
        if l.startswith("element vertex"):
            nv = int(l.split()[-1])
        if l.startswith("property float"):
            s += 1
    return np.frombuffer(d[he:he + nv * s * 4], dtype="<f4").reshape(nv, s)[:, :3]


def main():
    if len(sys.argv) < 2:
        print('usage: save_run.py <name> "notes"')
        sys.exit(1)
    name = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""
    ply = f"{MAPS}/{name}.ply"

    # 1) drain the mesh stream to PLY (reuse the OOM-safe tool)
    subprocess.run(["python3", "/workspace/ros/save_mesh_stream.py", name], check=True)
    if not os.path.exists(ply):
        print("no mesh saved (is the pipeline up and the map non-empty?)")
        sys.exit(1)

    # 2) metrics
    V = read_ply_verts(ply)
    n = len(V)
    ext = (V.max(0) - V.min(0)) if n else np.zeros(3)

    # 3) compress (keep only the .gz)
    with open(ply, "rb") as f, gzip.open(ply + ".gz", "wb", compresslevel=9) as g:
        shutil.copyfileobj(f, g)
    raw, comp = os.path.getsize(ply), os.path.getsize(ply + ".gz")
    os.remove(ply)

    # 4) config from this container's env
    cfg = " ".join(f"{k}={os.environ[k]}" for k in CONFIG_KEYS if k in os.environ)

    # 5) append to the run log
    row = (f"\n## {name}  ({time.strftime('%Y-%m-%d %H:%M')})\n"
           f"- **notes:** {notes}\n"
           f"- **points:** {n:,}  |  **extent:** "
           f"{ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} m  |  "
           f"**size:** {comp/1024:.0f} KB (raw {raw/1024:.0f} KB)\n"
           f"- **config:** `{cfg}`\n")
    with open(f"{MAPS}/RUNS.md", "a") as f:
        f.write(row)

    total = sum(os.path.getsize(os.path.join(MAPS, f))
                for f in os.listdir(MAPS) if os.path.isfile(os.path.join(MAPS, f)))
    print(f"archived {name}.ply.gz ({comp/1024:.0f} KB) + logged to RUNS.md")
    print(f"maps/ total: {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
