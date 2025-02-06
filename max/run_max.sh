# Base directory
BASE_DIR="/home/kmanasu/inference-experiment"

# Hugging Face token (replace with your actual token)
HF_TOKEN="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"
DOCKER_IMAGE="docker.modular.com/modular/max-openai-api:24.6.0"

docker run --gpus '"device=0"' -d \
    --name max \
    --net host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_RAWIO \
    -v "$BASE_DIR/":/workspace/ \
    -w /workspace/ \
    --entrypoint /bin/bash \
    "$DOCKER_IMAGE" -c "sleep infinity"

# apt update && apt install -y git && apt install -y curl
# curl -ssL https://magic.modular.com/deb17c3b-d3e8-4d42-8455-44e7a13049ec | bash
# source /root/.bashrc

# magic run serve --huggingface-repo-id=Qwen/Qwen2.5-1.5B-Instruct --use-gpu=7 --max-cache-batch-size=256 --max-length=2048 --quantization-encoding=bfloat16
# curl -N http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen2.5-1.5B-Instruct",
#         "stream": true,
#         "messages": [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "What is the capital of Mongolia?"}
#         ]
#     }' | grep -o '"content":"[^"]*"' | sed 's/"content":"//g' | sed 's/"//g' | tr -d '\n' | sed 's/\\n/\n/g'