#!/bin/bash

docker run -d --name triton-server \
  --gpus all \
  --runtime=nvidia \
  --shm-size=50g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /home/kmanasu/inference-experiment/triton/models:/models \
  kmanasu/triton-server:latest \
  tritonserver --model-repository=/models
