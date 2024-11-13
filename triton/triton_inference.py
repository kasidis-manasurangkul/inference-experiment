import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "llama-3.1-8B"  # Name of the deployed LLaMA model
text_input = "How would you describe the taste of a rainbow to someone who has never seen one?"  # Text input for the model
url = "localhost:8000"  # Triton server URL

# Configure client connection to Triton
with httpclient.InferenceServerClient(url, connection_timeout= 600, network_timeout= 600) as client:
    # Prepare input tensor
    input_text_data = np.array([text_input], dtype=object)  # Convert text to array for Triton
    inputs = [
        httpclient.InferInput("text_input", input_text_data.shape, "BYTES")  # Define model input
    ]
    inputs[0].set_data_from_numpy(input_text_data)  # Set input data

    # Define model output
    outputs = [
        httpclient.InferRequestedOutput("generated_text")  # Specify desired output tensor
    ]

    # Perform inference request
    response = client.infer(model_name, inputs, request_id="1", outputs=outputs, timeout=600)

    # Extract response data
    generated_text = response.as_numpy("generated_text")[0].decode("utf-8")

    print(f"Input: {text_input}")
    print(f"Generated Output: {generated_text}")

    # Perform a basic check on the response
    if not generated_text:
        print("Error: No output generated from the model.")
        sys.exit(1)

    print("PASS: Inference completed successfully.")
    sys.exit(0)