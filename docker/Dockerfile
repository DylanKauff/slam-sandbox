# Slim Isaac ROS runtime for Jetson Orin Nano 8GB (JetPack 6.2 / L4T r36.4)
#
# Why not NVIDIA's official isaac_ros_dev image: it is ~13GB compressed /
# ~25-30GB unpacked and does not fit on the 57GB SD card next to JetPack.
# This image installs only what the SLAM pipeline needs from the same three
# apt repos NVIDIA's own Dockerfiles use (Jetson r36.4, ROS 2 Humble,
# Isaac ROS release-3). Everything is baked in at build time so the
# container runs with no internet access (hard project requirement).
#
# Build (online, once):  docker build -t slam:latest /home/dylan/SLAM/docker
# Run  (offline):        /home/dylan/SLAM/docker/run_slam.sh

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg2 lsb-release locales \
        software-properties-common \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Jetson L4T r36.4 repo (CUDA/VPI/TensorRT tegra builds — NOT the SBSA repo)
RUN apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
    && echo 'deb https://repo.download.nvidia.com/jetson/common r36.4 main' > /etc/apt/sources.list.d/nvidia-l4t.list \
    && echo 'deb https://repo.download.nvidia.com/jetson/t234 r36.4 main' >> /etc/apt/sources.list.d/nvidia-l4t.list

# ROS 2 Humble repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list

# Isaac ROS release-3 repo (prebuilt cuVSLAM / nvblox debs for jammy arm64)
RUN wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | apt-key add - \
    && echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 jammy release-3.0" \
        > /etc/apt/sources.list.d/isaac-ros.list

# CUDA runtime libs + VPI first (Isaac debs dlopen these but do not declare them)
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-12-6 \
        libnvvpi3 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 base + Isaac ROS SLAM stack + Foxglove bridge
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-isaac-ros-visual-slam \
        ros-humble-isaac-ros-nvblox \
        ros-humble-isaac-ros-image-proc \
        ros-humble-isaac-ros-stereo-image-proc \
        ros-humble-isaac-ros-depth-image-proc \
        ros-humble-foxglove-bridge \
        ros-humble-cv-bridge \
        ros-humble-image-transport \
        python3-opencv \
        python3-numpy \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# CUDA on the library path (cuda-libraries lands in /usr/local/cuda-12.6)
RUN echo "/usr/local/cuda-12.6/targets/aarch64-linux/lib" > /etc/ld.so.conf.d/cuda.conf \
    && ldconfig

# NVTX (libnvToolsExt) is not part of cuda-libraries but the Isaac ROS
# binaries link against it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvtx-12-6 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# GStreamer v4l2src for the camera bridge: OpenCV's plain-V4L2 backend only
# manages ~1-2 fps on the 5120x800 GREY stream (select() timeouts), while the
# GStreamer path (same as used on the host) runs at the full sensor rate.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Tell the NVIDIA container runtime to inject the Tegra driver (libcuda etc.).
# NVIDIA base images set these; a plain ubuntu base must set them itself,
# otherwise every CUDA node fails with "libcuda.so.1: cannot open shared object".
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
