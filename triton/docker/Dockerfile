# Use Triton Inference Server with Python support as the base image
FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Install PyTorch, transformers, and other necessary dependencies
RUN pip install tritonclient[all] transformers torch
