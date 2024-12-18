# Use the Hugging Face Optimum NVIDIA base image
FROM huggingface/optimum-nvidia:latest

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements for installing dependencies
COPY requirements.txt .

# Install additional Python libraries (if needed)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port (if necessary)
EXPOSE 8080

CMD ["tail", "-f", "/dev/null"]