# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
BASE_DIR="/home/kmanasu/inference-experiment/ollama"
DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3"
# docker pull ollama/ollama:latest
docker run --gpus '"device=0"' -d \
    --name ollama \
    --net host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_RAWIO \
    -v "$BASE_DIR/":/workspace/ \
    -w /workspace/ \
    "$DOCKER_IMAGE" sleep infinity

docker run -d --gpus '"device=0"' --name ollama2 -p 11500:11500 -v "$BASE_DIR/":/workspace/ -e OLLAMA_HOST=http://0.0.0.0:11500 ollama/ollama
# CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.0:8080 ollama serve
