#!/bin/bash

docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/models-onnx:/models \
  nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver \
  --model-repository=/models \
  --strict-model-config=false
