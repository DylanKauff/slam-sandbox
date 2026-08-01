#!/bin/bash
# Host: Depth Anything V2 small ONNX -> fixed-shape FP16 TRT engine.
set -e
cd /home/dylan/SLAM/models
TRTEXEC=/usr/src/tensorrt/bin/trtexec
S=pixel_values:1x3x252x392
echo "building DA-V2 FP16 engine @252x392 (ViT, a few min)..."
$TRTEXEC --onnx=depth_anything_v2_vits.onnx --saveEngine=depth_anything_v2_vits.engine \
    --fp16 --minShapes=$S --optShapes=$S --maxShapes=$S \
    --memPoolSize=workspace:2048M 2>&1 | tail -8
ls -la depth_anything_v2_vits.engine
