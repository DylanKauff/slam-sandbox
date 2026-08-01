#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
# ROS 2 workspace with our bridge node / launch files, mounted by run_slam.sh
if [ -f /workspace/ros/install/setup.bash ]; then
    source /workspace/ros/install/setup.bash
fi
exec "$@"
