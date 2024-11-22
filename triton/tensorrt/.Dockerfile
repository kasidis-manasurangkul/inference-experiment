# Use the NVIDIA Triton image as the base
FROM nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

# Install additional tools (optional)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    python3-pip

# Install VS Code server dependencies
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Set default environment variables
ENV WORKSPACE_DIR=/workspace
ENV ENGINE_DIR=/workspace/engines
ENV HF_CACHE_DIR=/workspace/huggingface
ENV MODEL_REPO_DIR=/workspace/models
ENV HF_HOME=/workspace/huggingface

# Set default working directory
WORKDIR /workspace
