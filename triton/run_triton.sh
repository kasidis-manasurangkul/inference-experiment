#!/bin/bash

# Script to deploy LLaMA 3 8B model on NVIDIA Triton Inference Server
# Mounts everything to /home/kmanasu/inference-experiment/triton

# Exit immediately if a command exits with a non-zero status
set -e

# Base directory
BASE_DIR="/home/kmanasu/inference-experiment/triton"

# Hugging Face token (replace with your actual token)
HF_TOKEN="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"

# Docker image
DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3"

# Model name
MODEL_NAME="llama-3-8b"

# Number of GPUs (adjust as needed)
NUM_GPUS=8

# Hugging Face repository
HF_REPO="meta-llama/Meta-Llama-3-8B"

# Check if Hugging Face token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Please set your Hugging Face token in the script."
    exit 1
fi

# Create the base directory
mkdir -p "$BASE_DIR"

# Navigate to the base directory
cd "$BASE_DIR"

# Clone the TensorRT-LLM backend repository if not already cloned
if [ ! -d "$BASE_DIR/tensorrtllm_backend" ]; then
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.12.0
fi

# Navigate to the backend directory
cd "$BASE_DIR/tensorrtllm_backend"

# Update submodules
git submodule update --init --recursive

# Create directories for models and engines
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/engines"

# Pull the Docker image
docker pull "$DOCKER_IMAGE"

# Start the Docker container in detached mode
docker run --gpus all -d \
    --name triton_llama \
    --net host \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$BASE_DIR/models":/workspace/models \
    -v "$BASE_DIR/engines":/workspace/engines \
    -v "$BASE_DIR/tensorrtllm_backend":/workspace/tensorrtllm_backend \
    -w /workspace/tensorrtllm_backend \
    "$DOCKER_IMAGE" sleep infinity

# Install Triton CLI inside the container
docker exec triton_llama pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.11

# Set environment variables for the following commands
ENGINE_DIR=/workspace/engines
MODEL_DIR=/workspace/models

# Run the triton import command inside the container
docker exec triton_llama bash -c "export HF_TOKEN=$HF_TOKEN && ENGINE_DEST_PATH=$ENGINE_DIR triton import -m $MODEL_NAME --backend tensorrtllm --repo $HF_REPO"

# Update the model's config.pbtxt to use all GPUs
docker exec triton_llama bash -c "sed -i '/instance_group/d' $MODEL_DIR/$MODEL_NAME/config.pbtxt && echo -e '\ninstance_group [\n  {\n    kind: KIND_GPU\n    count: $NUM_GPUS\n    gpus: [0,1,2,3,4,5,6,7]\n  }\n]' >> $MODEL_DIR/$MODEL_NAME/config.pbtxt"

# Start the Triton server inside the container
docker exec -d triton_llama tritonserver --model-repository=/workspace/models

echo "Triton Inference Server is running with model $MODEL_NAME"
