#!/bin/bash
# Host: ONNX -> FP16 TensorRT engine (run once, after build_model.sh).
set -e
cd /home/dylan/SLAM/models
TRTEXEC=/usr/src/tensorrt/bin/trtexec
[ -f yolov8n-seg.onnx ] || { echo "no onnx yet -- run build_model.sh first"; exit 1; }
echo "building FP16 engine (a few min on Orin Nano)..."
$TRTEXEC --onnx=yolov8n-seg.onnx --saveEngine=yolov8n-seg.engine \
    --fp16 --memPoolSize=workspace:512M 2>&1 | tail -15
ls -la yolov8n-seg.engine
