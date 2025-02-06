MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"

CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching
