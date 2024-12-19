#!/bin/bash

# Set variables
IMAGE_NAME="ubuntu-tensorrt"
CONTAINER_NAME="tensorrt-container"
HOST_DIR="$(pwd)"  # Current directory
CONTAINER_DIR="/app"  # Target directory inside the container

# Build the Docker image
echo "Building the Docker image..."
docker build -t ${IMAGE_NAME} . || { echo "Docker build failed. Exiting."; exit 1; }

# Remove any existing container with the same name
echo "Removing any existing container with the same name..."
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Run the Docker container with GPUs 4-7
echo "Running the Docker container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus '"device=4,5,6,7"' \
    --ipc=host --shm-size=2g \
    -v ${HOST_DIR}:${CONTAINER_DIR} \
    -w /app \
    ${IMAGE_NAME} tail -f /dev/null || { echo "Docker run failed. Exiting."; exit 1; }

# Wait for a moment to ensure the container is up
echo "Waiting for the container to start..."
sleep 2

# Check if the container is running
if [ "$(docker inspect -f '{{.State.Running}}' ${CONTAINER_NAME})" == "true" ]; then
    echo "Entering the container..."
    docker exec -it ${CONTAINER_NAME} bash
else
    echo "The container failed to start. Check the logs for details."
    docker logs ${CONTAINER_NAME}
fi
