BASE_DIR="/home/kmanasu/inference-experiment/tensorrt"
# git clone https://github.com/NVIDIA/TensorRT-LLM.git

# # Clone or have the model downloaded already:
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2.5-72B-Instruct ./tmp/Qwen2.5/72B-Instruct

# # Convert the Qwen2.5-72B-Instruct model with tp_size=4 and pp_size=1 for a total of 4 GPUs.
# python3.10 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
#   --model_dir ./tmp/Qwen2.5/72B-Instruct \
#   --output_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
#   --dtype float16 \
#   --tp_size 4 \
#   --pp_size 1

# # Build the TensorRT engines for 4 GPUs.
trtllm-build \
  --checkpoint_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
  --output_dir ./tmp/Qwen2.5/72B-Instruct/trt_engines/fp16/4-gpu \
  --gemm_plugin float16
