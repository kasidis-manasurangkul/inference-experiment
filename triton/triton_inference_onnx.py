import time
import requests
import json

# Triton server URL
TRITON_SERVER_URL = "http://localhost:8000"

# Model name
MODEL_NAME = "llama-3-8b"

# Inference request payload
payload = {
    "text_input": "What is machine learning?",
    "max_tokens": 50,
    "bad_words": "",
    "stop_words": "",
    "pad_id": 2,
    "end_id": 2
}

def main():
    url = f"{TRITON_SERVER_URL}/v2/models/{MODEL_NAME}/generate"
    
    # Measure the time taken for the request
    start_time = time.time()
    
    response = requests.post(url, data=json.dumps(payload))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response.status_code == 200:
        result = response.json()
        print("Inference Result:")
        print(result)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
    
    print(f"\nTime taken for inference: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
