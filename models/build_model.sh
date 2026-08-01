#!/bin/bash
# One-time (host) model prep, no venv/sudo: install into a target dir, export ONNX.
set -e
cd /home/dylan/SLAM/models
PD=/home/dylan/SLAM/models/pydeps
echo "[1/3] install ultralytics (CPU torch) into $PD  -- slow"
python3 -m pip install -q --target="$PD" ultralytics onnx onnxslim 2>&1 | tail -4
echo "[2/3] export yolov8n-seg -> onnx @480 opset12"
PYTHONPATH="$PD" python3 - <<'PY'
from ultralytics import YOLO
m = YOLO("yolov8n-seg.pt")            # auto-downloads weights
p = m.export(format="onnx", imgsz=480, opset=12, simplify=True, half=False)
print("EXPORTED:", p)
PY
echo "[3/3] done"; ls -la /home/dylan/SLAM/models/*.onnx
