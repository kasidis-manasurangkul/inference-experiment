import tritonclient.http as httpclient
import numpy as np
from transformers import AutoTokenizer

# Set up the client for Triton
node_ip = "<node-ip>"  # Replace with your actual node IP
url = f"{node_ip}:30081"  # HTTP endpoint of Triton server service
client = httpclient.InferenceServerClient(url=url)

# Initialize the tokenizer for LLaMA
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Use your specific model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare text input
text_input = "What is the capital of France?"  # Replace with any text input
input_ids = tokenizer.encode(text_input, return_tensors="np").astype(np.int64)

# Create Triton input tensor
inputs = httpclient.InferInput("input_ids", input_ids.shape, "INT64")
inputs.set_data_from_numpy(input_ids)

# Make an inference request to the model
try:
    response = client.infer(model_name="llama-3.1-8B", inputs=[inputs])
    output_ids = response.as_numpy("output_ids")  # Adjust if your output name is different

    # Decode the output IDs back to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Model output:", output_text)
except Exception as e:
    print("Error during inference:", e)
