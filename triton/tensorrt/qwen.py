#!/usr/bin/env python3

import asyncio
import json
import time
from transformers import AutoTokenizer
from datasets import load_dataset  # Import for loading SQuAD dataset
import openai  # Import OpenAI client
import logging  # Import logging for better monitoring
import re
import os

# -------------------
# Configure Logging
# -------------------
logging.basicConfig(
    filename='benchmark_async.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -------------------
# Utility functions
# -------------------
def build_prompt(question, context):
    """Build the full prompt string for SQuAD."""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return prompt

def extract_text_output(raw_output):
    """Return raw_output directly since it's plain text."""
    return raw_output

# ------------------------------------------------------------------------
# Triton Inference settings
# ------------------------------------------------------------------------
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "http://localhost:8080/v1")  # OpenAI-Compatible Frontend base URL
MODEL_NAME = os.getenv("MODEL_NAME", "ensemble")  # Update to your model's name
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1000))  # Set to 10 as per request
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "inference_results_async.json")
METRICS_FILE = os.getenv("METRICS_FILE", "inference_metrics_async.txt")

# ------------------------------------------------------------------------
# Initialize the Qwen tokenizer
# ------------------------------------------------------------------------
# Replace 'Qwen/Qwen2.5-7B-Instruct' with the actual model name or path if different
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    print(f"Error loading tokenizer: {e}")
    exit(1)

# ------------------------------------------------------------------------
# Initialize OpenAI Client
# ------------------------------------------------------------------------
def initialize_openai_client(base_url, api_key="EMPTY"):
    """
    Initialize the OpenAI client to communicate with Triton's OpenAI-Compatible Frontend.

    Args:
        base_url (str): The base URL of the OpenAI-Compatible Frontend.
        api_key (str, optional): API key for authentication. Defaults to "EMPTY".

    Returns:
        None: Sets the OpenAI API base and key.
    """
    openai.api_base = base_url
    openai.api_key = api_key
    return

# ------------------------------------------------------------------------
# Asynchronous Function to Send Request with Retry Mechanism
# ------------------------------------------------------------------------
async def send_request_openai(model, messages, max_tokens, request_id, results, elapsed_times, semaphore, retries=3):
    """
    Asynchronously send a chat completion request using OpenAI's acreate method with retry mechanism.

    Args:
        model (str): Model name to use for completion.
        messages (list): List of message dictionaries with 'role' and 'content'.
        max_tokens (int): Maximum number of tokens to generate.
        request_id (int): Identifier for the current request.
        results (list): List to store results.
        elapsed_times (list): List to store elapsed times for each request.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.
        retries (int): Number of retry attempts for failed requests.

    Returns:
        bool: True if request succeeded, False otherwise.
    """
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                start_time = time.time()

                # Send the asynchronous request without streaming
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=False  # Disable streaming
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)

                # Extract the generated text
                generated_message = response.choices[0].message['content'].strip()

                # Tokenize and count tokens
                token_count = count_tokens(generated_message)
                tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0

                logging.info(f"Request {request_id}: {token_count} tokens in {elapsed_time:.4f}s (Attempt {attempt})")
                print(f"Request {request_id}: {token_count} tokens in {elapsed_time:.4f}s (Attempt {attempt})")

                # Append to results
                results.append({
                    "index": request_id,
                    "input_prompt": messages[-1]["content"],  # Assuming last message is user prompt
                    "output_text": generated_message,
                    "tokens_count": token_count,
                    "tokens_per_sec": tokens_per_sec,
                })

                # Record metrics
                metrics.end_request(request_id, generated_message)

                return True

            except openai.error.RateLimitError as e:
                logging.warning(f"Request {request_id} encountered RateLimitError: {e}. Attempt {attempt}/{retries}")
                print(f"Request {request_id} encountered RateLimitError: {e}. Attempt {attempt}/{retries}")
            except openai.error.APIConnectionError as e:
                logging.warning(f"Request {request_id} encountered APIConnectionError: {e}. Attempt {attempt}/{retries}")
                print(f"Request {request_id} encountered APIConnectionError: {e}. Attempt {attempt}/{retries}")
            except openai.error.OpenAIError as e:
                logging.error(f"Request {request_id} failed with OpenAI error: {e}. Attempt {attempt}/{retries}")
                print(f"Request {request_id} failed with OpenAI error: {e}. Attempt {attempt}/{retries}")
                break  # Non-retriable error
            except Exception as e:
                logging.error(f"Request {request_id} encountered an exception: {e}. Attempt {attempt}/{retries}")
                print(f"Request {request_id} encountered an exception: {e}. Attempt {attempt}/{retries}")
                break  # Non-retriable error

            # Exponential backoff before retrying
            backoff_time = 2 ** attempt
            logging.info(f"Waiting for {backoff_time} seconds before retrying...")
            print(f"Waiting for {backoff_time} seconds before retrying...")
            await asyncio.sleep(backoff_time)

        # After all retries failed
        logging.error(f"Request {request_id} failed after {retries} attempts.")
        print(f"Request {request_id} failed after {retries} attempts.")
        results.append({
            "index": request_id,
            "input_prompt": messages[-1]["content"],
            "output_text": "",
            "error": f"Failed after {retries} attempts."
        })
        return False

# ------------------------------------------------------------------------
# Token Counting Function
# ------------------------------------------------------------------------
def count_tokens(text):
    """
    Count the number of tokens in a given text using the tokenizer.

    Args:
        text (str): The text to tokenize.

    Returns:
        int: Number of tokens.
    """
    if tokenizer is None:
        # Fallback to character count if no tokenizer is available
        return len(text)
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        print(f"Error counting tokens: {e}")
        return len(text)  # Fallback to character count

# ------------------------------------------------------------------------
# Metrics Tracker Class
# ------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self, tokenizer=None):
        self.start_times = {}
        self.ttfts = {}
        self.end_times = {}
        self.token_counts = {}
        self.first_token_received = False
        self.first_token_time = None
        self.tokenizer = tokenizer

    def count_tokens(self, text):
        return count_tokens(text)

    def start_request(self, request_id):
        self.start_times[request_id] = time.time()

    def record_first_token(self, request_id):
        current_time = time.time()
        if not self.first_token_received:
            self.first_token_time = current_time
            self.first_token_received = True
        self.ttfts[request_id] = current_time - self.start_times[request_id]

    def end_request(self, request_id, response_text):
        self.end_times[request_id] = time.time()
        self.token_counts[request_id] = self.count_tokens(response_text)

    def get_metrics(self):
        if not self.end_times or not self.start_times:
            return {}
        total_time = max(self.end_times.values()) - min(self.start_times.values())
        total_tokens = sum(self.token_counts.values())
        avg_response_time = sum(
            self.end_times[rid] - self.start_times[rid]
            for rid in self.end_times
        ) / len(self.end_times)
        throughput = len(self.end_times) / total_time if total_time > 0 else 0
        avg_ttft = sum(self.ttfts.values()) / len(self.ttfts) if self.ttfts else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        return {
            "total_time_seconds": round(total_time, 3),
            "throughput_rps": round(throughput, 3),
            "average_response_time_seconds": round(avg_response_time, 3),
            "average_ttft_seconds": round(avg_ttft, 3),
            "first_token_time_seconds": round(self.first_token_time - min(self.start_times.values()), 3) if self.first_token_time else None,
            "total_tokens": total_tokens,
            "tokens_per_second": round(tokens_per_second, 3),
            "average_tokens_per_response": round(total_tokens / len(self.token_counts), 3) if self.token_counts else 0,
            "tokenizer_used": type(self.tokenizer).__name__ if self.tokenizer else "character_count"
        }

# Initialize Metrics Tracker
metrics = MetricsTracker(tokenizer)

# ------------------------------------------------------------------------
# Main Asynchronous Function
# ------------------------------------------------------------------------
async def main_async():
    # Initialize OpenAI client with API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    initialize_openai_client(TRITON_SERVER_URL, api_key=api_key)

    print("Loading the 'squad' dataset...")
    logging.info("Loading the 'squad' dataset...")
    dataset = load_dataset("squad", split="validation")
    subset = dataset.select(range(NUM_SAMPLES))

    results = []
    elapsed_times = []  # List to store elapsed time for each request

    # Define maximum number of concurrent requests
    max_concurrent_requests = 2048  # Adjust based on your system's capability and Triton's limits
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = []
    for idx, entry in enumerate(subset):
        question = entry["question"]
        context = entry["context"]

        prompt = build_prompt(question, context)

        # Prepare messages for OpenAI ChatCompletion
        messages = [
            {
                "role": "system",
                "content": "Answer the following question"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Start tracking
        metrics.start_request(idx)

        # Create the asynchronous task
        task = asyncio.create_task(
            send_request_openai(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1,  # Adjust as needed
                request_id=idx,
                results=results,
                elapsed_times=elapsed_times,
                semaphore=semaphore,
                retries=3  # Set number of retries
            )
        )
        tasks.append(task)

    # Await all tasks to complete
    await asyncio.gather(*tasks)

    # Summarize results
    total_elapsed = sum(elapsed_times)
    average_request_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    success_count = sum(1 for r in results if "tokens_count" in r)

    summary = (
        f"\nTotal time for {len(results)} requests: {total_elapsed:.2f}s\n"
        f"Average request time: {average_request_time:.4f}s\n"
        f"Successful requests: {success_count}/{len(results)}"
    )
    logging.info(summary)
    print(summary)

    # Calculate metrics
    total_tokens = sum(r["tokens_count"] for r in results if "tokens_count" in r)
    avg_tokens_per_sec = (total_tokens / total_elapsed) if total_elapsed > 0 else 0

    metrics_summary = metrics.get_metrics()

    # Save metrics
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Metrics:\n{json.dumps(metrics_summary, indent=2)}")
    print(f"Inference metrics saved to: {METRICS_FILE}")

    # Prepare JSON results, extracting text if needed
    json_results = []
    for r in results:
        extracted_text = extract_text_output(r.get("output_text", ""))
        json_results.append({
            "input_text": r.get("input_prompt", ""),
            "output_text": extracted_text
        })

    # Save JSON results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    logging.info(f"Inference results saved to: {OUTPUT_FILE}")
    print(f"Inference results saved to: {OUTPUT_FILE}")

# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Benchmarking interrupted by user.")
        print("Benchmarking interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
