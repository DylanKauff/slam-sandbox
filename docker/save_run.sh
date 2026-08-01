#!/bin/bash
# Archive the current live map with notes into the standardized run log.
#   ./docker/save_run.sh <run_name> "notes: what we changed and the result"
# Writes maps/<name>.ply.gz (compressed) and appends to maps/RUNS.md.
if [ -z "$1" ]; then
  echo 'usage: ./docker/save_run.sh <run_name> "notes"'; exit 1
fi
docker exec slam bash -lc \
  'source /opt/ros/humble/setup.bash; python3 /workspace/ros/save_run.py "$1" "$2"' \
  _ "$1" "$2"
