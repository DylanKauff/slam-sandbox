#!/bin/bash
# Start the SLAM container (works fully offline).
#
#   ./run_slam.sh                 -> interactive shell in the container
#   ./run_slam.sh <command...>    -> run a command (e.g. a ros2 launch)
#
# --network host : ROS 2 DDS + foxglove websocket reachable from the laptop LAN
# --runtime nvidia: injects the Tegra GPU driver into the container
# Devices: /dev/video0 alone is NOT enough for CSI capture on Jetson — the VI
# DMA channels (/dev/capture-vi-channel*), /dev/host1x-fence and /dev/media0
# must be passed too, else V4L2 reads time out with zero frames.

DEVS="--device /dev/video0 --device /dev/media0 --device /dev/host1x-fence"
for d in /dev/capture-vi-channel*; do DEVS="$DEVS --device $d"; done

# VPI (GPU stereo depth): mount the host JetPack library + python bindings.
VPI_MOUNTS="-v /opt/nvidia/vpi3:/opt/nvidia/vpi3:ro \
    -v /usr/lib/python3/dist-packages/vpi.cpython-310-aarch64-linux-gnu.so:/usr/lib/python3/dist-packages/vpi.cpython-310-aarch64-linux-gnu.so:ro"

exec docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    $DEVS \
    $VPI_MOUNTS \
    -v /home/dylan/SLAM:/workspace \
    -e ROS_DOMAIN_ID=0 \
    -e LD_LIBRARY_PATH=/opt/nvidia/vpi3/lib/aarch64-linux-gnu \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    slam:latest "$@"
