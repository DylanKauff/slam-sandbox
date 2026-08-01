#!/bin/bash
# DO NOT USE on the 8 GB Orin Nano: the save_ply service serializes the whole
# mesh in one GPU allocation and OOM-crashed the SLAM container (map lost,
# 2026-07-12). Use save_mesh_stream.py instead — it drains the mesh topic
# incrementally and cannot hurt the running pipeline:
#     docker exec slam python3 /workspace/ros/save_mesh_stream.py [name]
#
# Save the current nvblox reconstruction as a PLY mesh (run inside the
# container while the SLAM pipeline is up, or via run_slam.sh):
#
#   /workspace/docker/run_slam.sh /workspace/ros/save_map.sh [name]
#
# The mesh lands in /workspace/maps/ (= ~/SLAM/maps on the Jetson) and can be
# opened in MeshLab/Blender or re-visualized any time.
set -e
source /opt/ros/humble/setup.bash
NAME="${1:-room_$(date +%Y%m%d_%H%M%S)}"
mkdir -p /workspace/maps
ros2 service call /nvblox_node/save_ply nvblox_msgs/srv/FilePath \
    "{file_path: '/workspace/maps/${NAME}.ply'}"
echo "saved /workspace/maps/${NAME}.ply"
