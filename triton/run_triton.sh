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
DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3"

# # Model name
# MODEL_NAME="llama-3-8b"

# # Number of GPUs (adjust as needed)
# NUM_GPUS=4

# # Hugging Face repository
# HF_REPO="meta-llama/Meta-Llama-3-8B"

# # Check if Hugging Face token is set
# if [ -z "$HF_TOKEN" ]; then
#     echo "Please set your Hugging Face token in the script."
#     exit 1
# fi

# # Create the base directory
# mkdir -p "$BASE_DIR"

# # Navigate to the base directory
# cd "$BASE_DIR"

# # Clone the TensorRT-LLM backend repository if not already cloned
# if [ ! -d "$BASE_DIR/tensorrtllm_backend" ]; then
#     git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.12.0
# fi

# # Navigate to the backend directory
# cd "$BASE_DIR/tensorrtllm_backend"

# # Update submodules
# git submodule update --init --recursive

# # Create directories for models and engines
# mkdir -p "$BASE_DIR/models"
# mkdir -p "$BASE_DIR/engines"

# # Pull the Docker image
# docker pull "$DOCKER_IMAGE"

# # Start the Docker container in detached mode
# docker run --gpus '"device=0"' -d \
#     --name triton_container1 \
#     --net host \
#     --shm-size=8g \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     --cap-add=SYS_RAWIO \
#     -v "$BASE_DIR/":/workspace/ \
#     -w /workspace/ \
#     "$DOCKER_IMAGE" sleep infinity
BASE_DIR="/home/kmanasu/inference-experiment/"
docker run --gpus '"device=0"' -d \
    --name triton_container1 \
    --net host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_RAWIO \
    -v "$BASE_DIR/":/workspace/ \
    -w /workspace/ \
    "$DOCKER_IMAGE" sleep infinity
# # Install Triton CLI inside the container
# pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.11
# pip install tokenizers==0.20
# pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
# pip install tensorrt-llm==0.17.0.dev2024121700
# pip install tensorrt-llm==0.15.0
# git clone https://github.com/NVIDIA/TensorRT-LLM.git
# # Clone or have the model downloaded already:
# git lfs install
# git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ./tensorrt/tmp/Llama-3.1/8B-Instruct
# git clone https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct ./tensorrt/tmp/Llama-3.3/70B-Instruct
# git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct ./tensorrt/tmp/Qwen2.5/7B-Instruct
# git clone https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct ./tensorrt/tmp/Qwen2.5/VL72B-Instruct
# git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct ./tensorrt/tmp/Llama-3.2/1B-Instruct
# export MODEL_NAME="Qwen2-VL-7B-Instruct" # or Qwen2-VL-2B-Instruct
# git clone https://huggingface.co/Qwen/${MODEL_NAME} ./tensorrt/tmp/Qwen2.5/VL72B-Instruct
# # Convert the Qwen2.5-72B-Instruct model with tp_size=4 and pp_size=1 for a total of 4 GPUs.
# python3 tensorrt/TensorRT-LLM/examples/llama/convert_checkpoint.py \
#   --model_dir tensorrt/tmp/Llama-3.2/1B-Instruct/ \
#   --output_dir tensorrt/checkpoints/llama_3.2_1b_1gpu_tp2_pp1 \
#   --dtype bfloat16 \
#   --tp_size 2 \
#   --pp_size 1 \
#   --workers 16

# Build the TensorRT engines for 4 GPUs.
# trtllm-build \
#   --checkpoint_dir tensorrt/checkpoints/llama_3.2_1b_1gpu_tp1_pp1 \
#   --output_dir tensorrt/engines/Llama-3.2/1B-Instruct/trt_engines3/bf16/1-gpu \
#   --gemm_plugin bfloat16 \
#   --workers 16 \
#   --max_batch_size 2048 \
#   --kv_cache_type paged

# TOKENIZER_PATH="tensorrt/tmp/Llama-3.2/1B-Instruct/"
# MODEL_PATH="tensorrt/models/llama3.2-1B/"
# ENGINE_PATH="tensorrt/engines/Llama-3.2/1B-Instruct/trt_engines2/bf16/1-gpu/"

# # Create model directory
# mkdir -p "$MODEL_PATH"

# # Copy inflight batcher files
# cp -r tensorrt/tensorrtllm_backend/all_models/inflight_batcher_llm/* "$MODEL_PATH"

# # Update preprocessing config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i "$MODEL_PATH/preprocessing/config.pbtxt" \
#   tokenizer_dir:"$TOKENIZER_PATH",tokenizer_type:llama,triton_max_batch_size:2048,preprocessing_instance_count:8,stream:False,engine_dir:"$ENGINE_PATH"

# # Update postprocessing config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i "$MODEL_PATH/postprocessing/config.pbtxt" \
#   tokenizer_dir:"$TOKENIZER_PATH",tokenizer_type:llama,triton_max_batch_size:2048,postprocessing_instance_count:8,stream:False

# # Update tensorrt_llm_bls config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i "$MODEL_PATH/tensorrt_llm_bls/config.pbtxt" \
#   triton_max_batch_size:2048,decoupled_mode:True,bls_instance_count:4,accumulate_tokens:True,stream:False

# # Update ensemble config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i "$MODEL_PATH/ensemble/config.pbtxt" \
#   triton_max_batch_size:32,stream:False

# # Update tensorrt_llm config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i "$MODEL_PATH/tensorrt_llm/config.pbtxt" \
#   triton_max_batch_size:32,decoupled_mode:True,max_beam_width:1,engine_dir:"$ENGINE_PATH",max_tokens_in_paged_kv_cache:20480,max_attention_window_size:20480,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600,triton_backend:tensorrtllm,encoder_engine_dir:,decoding_mode:gpt,stream:False,streaming:False,batch_scheduler_policy:max_utilization

# pip install /opt/tritonserver/python/triton*.whl
# # git clone https://github.com/triton-inference-server/server.git
export TRITON_CUDA_MEMORY_POOL_BYTE_SIZE=37580963840
export TRTLLM_ORCHESTRATOR=1
# # # python3 tensorrt/tensorrtllm_backend/scripts/launch_triton_server.py --world_size 4 --model_repo=tensorrt/models/qwen-2.5-7B --http_port=8080
# python3 tensorrt/server/python/openai/openai_frontend/main.py \
#     --model-repository tensorrt/models/llama3.1-8B \
#     --tokenizer tensorrt/tmp/Llama-3.1/8B-Instruct/ \
#     --openai-port 8080
CUDA_VISIBLE_DEVICES=0 python3 tensorrt/server/python/openai/openai_frontend/main.py \
    --model-repository tensorrt/models/llama3.2-1B \
    --tokenizer tensorrt/tmp/Llama-3.2/1B-Instruct/ \
    --openai-port 8080




# Convert the Qwen2.5-72B-Instruct model with tp_size=4 and pp_size=1 for a total of 4 GPUs.
# python3 tensorrt/TensorRT-LLM/examples/qwen/convert_checkpoint.py \
#   --model_dir tensorrt/tmp/Qwen2.5/72B-Instruct/ \
#   --output_dir tensorrt/checkpoints/qwen_2.5_vl72b_8gpu_tp8_pp1 \
#   --dtype float16 \
#   --tp_size 8 \
#   --pp_size 1 \
#   --workers 16

# # Build the TensorRT engines for 4 GPUs.
# trtllm-build \
#   --checkpoint_dir tensorrt/checkpoints/qwen_2.5_vl72b_8gpu_tp8_pp1 \
#   --output_dir tensorrt/engines/Qwen2.5/VL72B-Instruct/trt_engines2/fp16/4-gpu \
#   --gemm_plugin float16 \
#   --max_batch_size 2048 \
#   --workers 16 \
#   --multiple_profiles enable


# mkdir -p  tensorrt/models/qwen-2.5-VL72B/
# cp tensorrt/tensorrtllm_backend/all_models/inflight_batcher_llm/* tensorrt/models/qwen-2.5-VL72B/ -r

# # Update preprocessing config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i tensorrt/models/qwen-2.5-VL72B/preprocessing/config.pbtxt \
#   tokenizer_dir:tensorrt/tmp/Qwen2.5/VL72B-Instruct/,tokenizer_type:llama,triton_max_batch_size:2048,preprocessing_instance_count:4,stream:True

# # Update postprocessing config
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i tensorrt/models/qwen-2.5-VL72B/postprocessing/config.pbtxt \
#   tokenizer_dir:tensorrt/tmp/Qwen2.5/VL72B-Instruct/,tokenizer_type:llama,triton_max_batch_size:2048,postprocessing_instance_count:4,stream:True

# # Update tensorrt_llm_bls (often Python backend)
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i tensorrt/models/qwen-2.5-VL72B/tensorrt_llm_bls/config.pbtxt \
#   triton_max_batch_size:2048,decoupled_mode:True,bls_instance_count:4,accumulate_tokens:True,stream:True

# # Update ensemble (often 'platform: "ensemble"')
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i tensorrt/models/qwen-2.5-VL72B/ensemble/config.pbtxt \
#   triton_max_batch_size:2048,stream:True

# # **Update tensorrt_llm config** to fill in backend: "${triton_backend}" -> "tensorrt_llm"
# python3 tensorrt/tensorrtllm_backend/tools/fill_template.py \
#   -i tensorrt/models/qwen-2.5-VL72B/tensorrt_llm/config.pbtxt \
#   triton_max_batch_size:2048,decoupled_mode:True,max_beam_width:1,engine_dir:tensorrt/engines/Qwen2.5/72B-Instruct/trt_engines4/fp16/4-gpu/,max_tokens_in_paged_kv_cache:40960,max_attention_window_size:40960,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:300,triton_backend:tensorrtllm,encoder_engine_dir:,decoding_mode:gpt,stream:True,streaming:True,batch_scheduler_policy:max_utilization

# pip install /opt/tritonserver/python/triton*.whl
# # git clone https://github.com/triton-inference-server/server.git
# export TRITON_CUDA_MEMORY_POOL_BYTE_SIZE=37580963840
# # # # export TRTLLM_ORCHESTRATOR=1
# # python3 tensorrt/tensorrtllm_backend/scripts/launch_triton_server.py --world_size 8 --model_repo=tensorrt/models/qwen-2.5-VL72B --http_port=8080
# mpirun -n 8 --allow-run-as-root python3 tensorrt/server/python/openai/openai_frontend/main.py \
#     --model-repository tensorrt/models/qwen-2.5-VL72B \
#     --tokenizer tensorrt/tmp/Qwen2.5/VL72B-Instruct/ \
#     --openai-port 8080

