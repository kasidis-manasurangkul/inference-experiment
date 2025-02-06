import time
import asyncio
import aiohttp
import json
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

# Triton server URL
TRITON_SERVER_URL = "http://localhost:8080"

# Model name
MODEL_NAME = "llama-3-8b"

# Number of samples to use
NUM_SAMPLES = 1000

# Maximum number of concurrent requests
MAX_CONCURRENT_REQUESTS = 2  # Adjust based on your server's capacity

# Output JSON file
OUTPUT_FILE = "inference_results.json"

async def send_request(session, url, payload, idx, semaphore, results):
    async with semaphore:
        start_time = time.time()
        async with session.post(url, data=json.dumps(payload)) as response:
            end_time = time.time()
            elapsed_time = end_time - start_time

            if response.status == 200:
                result = await response.json()
                output_text = result.get("text_output", "")
                # Save the input, output, and runtime
                results.append({
                    "input": payload["text_input"],
                    "output": output_text,
                    "runtime": elapsed_time
                })
                return True
            else:
                error_text = await response.text()
                print(f"Request {idx+1} failed with status code {response.status}: {error_text}")
                # Save the error information
                results.append({
                    "input": payload["text_input"],
                    "output": "",
                    "runtime": elapsed_time,
                    "error": error_text
                })
                return False

async def main():
    # Load the SQuAD dataset
    print("Loading the SQuAD dataset...")
    dataset = load_dataset("squad_v2", split="validation")
    questions = dataset["question"][:NUM_SAMPLES]  # Extract the first NUM_SAMPLES questions

    url = f"{TRITON_SERVER_URL}/v2/models/{MODEL_NAME}/generate"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # List to store the results
    results = []

    # Create an HTTP session
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, text_input in enumerate(questions):
            # Prepare the payload for each sample
            payload = {
                "text_input": text_input,
                "max_tokens": 2048,
                "bad_words": "",
                "stop_words": "",
                "pad_id": 2,
                "end_id": 2
            }
            task = asyncio.ensure_future(send_request(session, url, payload, idx, semaphore, results))
            tasks.append(task)

        print(f"Sending {NUM_SAMPLES} requests...")
        overall_start_time = time.time()
        # Run tasks concurrently with tqdm progress bar
        responses = []
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            response = await f
            responses.append(response)
        overall_end_time = time.time()
        total_elapsed_time = overall_end_time - overall_start_time

        success_count = sum(responses)
        print(f"\nTotal time for {NUM_SAMPLES} requests: {total_elapsed_time:.2f} seconds")
        print(f"Average time per request: {total_elapsed_time / NUM_SAMPLES:.2f} seconds")
        print(f"Successful requests: {success_count}/{NUM_SAMPLES}")

    # Save the results to a JSON file
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Results saved.")

if __name__ == "__main__":
    asyncio.run(main())
