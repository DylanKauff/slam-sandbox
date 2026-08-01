#!/bin/bash
# Start the full SLAM pipeline detached, with the tuned "run J" settings that
# gave a clean, complete room map, then bring up both dashboards.
#
#   ./docker/start_slam.sh          # start pipeline + dashboards
#   ./docker/start_slam.sh --no-gui # pipeline only
#
# Dashboards:  http://<jetson-ip>:8091  (depth/epipolar diag)
#              http://<jetson-ip>:8092  (room map, trackpad-navigable)
set -e

docker rm -f slam 2>/dev/null || true
sleep 2

DEVS="--device /dev/video0 --device /dev/media0 --device /dev/host1x-fence"
for d in /dev/capture-vi-channel*; do DEVS="$DEVS --device $d"; done

VPI_MOUNTS="-v /opt/nvidia/vpi3:/opt/nvidia/vpi3:ro \
    -v /usr/lib/python3/dist-packages/vpi.cpython-310-aarch64-linux-gnu.so:/usr/lib/python3/dist-packages/vpi.cpython-310-aarch64-linux-gnu.so:ro"

# Layer 2 (neural object seg): mount the host TensorRT python binding into the
# container (host+container are both py3.10, and the runtime libnvinfer.so.10 /
# libcudart.so.12 already live in the image). No torch/onnxruntime at runtime.
TRT_MOUNT="-v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro"

# --- tuned pipeline settings (run J) ---------------------------------------
# Depth: full-res VPI SGM, permissive matching (fills textureless corners) with
#   high smoothness; noise rejection is handed to nvblox multi-view weighting.
# nvblox: 3.5 m range, mesh only well-observed voxels (mesh_integrator_min_weight).
# cuVSLAM: tolerant frame timing + ground constraint => holds pose through
#   rest/resume, no angular ghost.
docker run -d --name slam --rm \
    --runtime nvidia --network host --ipc host \
    $DEVS \
    $VPI_MOUNTS \
    $TRT_MOUNT \
    -v /home/dylan/SLAM:/workspace \
    -e ROS_DOMAIN_ID=0 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/ros/fastdds_shm.xml \
    -e LD_LIBRARY_PATH=/opt/nvidia/vpi3/lib/aarch64-linux-gnu \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVBLOX_MAX_INTEG_M=3.5 -e MESH_MIN_WEIGHT=1.0 -e VOXEL_SIZE=${VOXEL_SIZE:-0.05} -e TSDF_MAX_WEIGHT=8 \
    -e DECAY_HZ=0.5 -e DECAY_FACTOR=0.95 -e DECAY_EXCLUDE_LAST_VIEW=1 \
    -e MESH_PUB_HZ=${MESH_PUB_HZ:-1.0} \
    -e VPI_UNIQUENESS=0.95 -e VPI_CONF_MIN=1 -e VPI_P1=8 -e VPI_P2=192 -e VPI_WINDOW=7 \
    -e DEPTH_FILTER=1 -e DEPTH_HZ=4 \
    -e VSLAM_JITTER_MS=300 -e GROUND_CONSTRAINT=1 \
    -e SEMANTIC=1 \
    -e SEM_PLANE_THRESH=0.03 -e SEM_N_PLANES=3 -e SEM_MIN_PLANE=1200 \
    -e SEM_PROC_W=384 -e SEM_RANSAC_ITERS=40 \
    -e SEM_CORNER_FILL=1 -e SEM_CORNER_BAND=0.12 \
    -e SEM_LINE_REJECT=0.80 -e SEM_MIN_CLUSTER=40 -e SEM_CLUSTER_VOX=0.05 \
    -e SEMANTIC_ML=${SEMANTIC_ML:-1} \
    -e SEG_ENGINE=/workspace/models/yolov8n-seg.engine -e SEG_IMGSZ=480 -e SEG_HZ=${SEG_HZ:-0.5} \
    -e MONO_DEPTH=${MONO_DEPTH:-1} \
    -e MONO_ENGINE=/workspace/models/depth_anything_v2_vits.engine -e MONO_H=252 -e MONO_W=392 -e MONO_HZ=${MONO_HZ:-0.5} \
    -e MONO_FILL=1 -e MONO_FILL_DILATE=18 -e MONO_FILL_MAX=4.0 \
    slam:latest ros2 launch /workspace/ros/slam_launch.py

echo "pipeline starting (container: slam)"

if [ "$1" != "--no-gui" ]; then
    sleep 14
    docker exec -d slam bash -lc 'source /opt/ros/humble/setup.bash; python3 /workspace/ros/diag_gui.py'
    docker exec -d slam bash -lc 'source /opt/ros/humble/setup.bash; python3 /workspace/ros/room_map.py'
    echo "dashboards up: :8091 (diag)  :8092 (room map)"
fi
