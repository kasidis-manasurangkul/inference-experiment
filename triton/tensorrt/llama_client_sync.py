#!/usr/bin/env python3

import json
import time
from transformers import AutoTokenizer
from datasets import load_dataset  # Import for loading SQuAD dataset
import openai  # Import OpenAI client
import logging  # Import logging for better monitoring
import os

# -------------------
# Configure Logging
# -------------------
logging.basicConfig(
    filename='benchmark_sync.log',
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
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1000))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "llama_inference_results_sync.json")
METRICS_FILE = os.getenv("METRICS_FILE", "llama_inference_metrics_sync.txt")

# ------------------------------------------------------------------------
# Initialize the Qwen tokenizer
# ------------------------------------------------------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    print(f"Error loading tokenizer: {e}")
    exit(1)

# ------------------------------------------------------------------------
# Initialize OpenAI Client
# ------------------------------------------------------------------------
def initialize_openai_client(base_url, api_key="EMPTY"):
    openai.api_base = base_url
    openai.api_key = api_key
    return

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
        return len(text)
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logging.error(f"Error counting tokens: {e}")
        print(f"Error counting tokens: {e}")
        return len(text)

# ------------------------------------------------------------------------
# Metrics Tracker Class
# ------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.token_counts = {}

    def count_tokens(self, text):
        return count_tokens(text)

    def start_request(self, request_id):
        self.start_times[request_id] = time.time()

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

        return {
            "total_time_seconds": round(total_time, 3),
            "throughput_rps": round(throughput, 3),
            "average_response_time_seconds": round(avg_response_time, 3),
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / total_time, 3) if total_time > 0 else 0,
            "average_tokens_per_response": round(total_tokens / len(self.token_counts), 3) if self.token_counts else 0,
        }

# Initialize Metrics Tracker
metrics = MetricsTracker()

# ------------------------------------------------------------------------
# Main Function for Synchronous Requests
# ------------------------------------------------------------------------
def main():
    # Initialize OpenAI client with API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    initialize_openai_client(TRITON_SERVER_URL, api_key=api_key)

    print("Loading the 'squad' dataset...")
    logging.info("Loading the 'squad' dataset...")
    dataset = load_dataset("squad", split="validation")
    subset = dataset.select(range(NUM_SAMPLES))

    results = []
    elapsed_times = []  # List to store elapsed time for each request

    for idx, entry in enumerate(subset):
        question = entry["question"]
        context = entry["context"]

        prompt = build_prompt(question, context)

        messages = [
            {"role": "system", "content": "Answer the following question"},
            {"role": "user", "content": prompt}
        ]

        metrics.start_request(idx)
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1
            )
            end_time = time.time()

            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)

            generated_message = response.choices[0].message['content'].strip()
            token_count = count_tokens(generated_message)
            tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"Request {idx}: {token_count} tokens in {elapsed_time:.4f}s")
            print(f"Request {idx}: {token_count} tokens in {elapsed_time:.4f}s")

            results.append({
                "index": idx,
                "input_prompt": prompt,
                "output_text": generated_message,
                "tokens_count": token_count,
                "tokens_per_sec": tokens_per_sec,
            })

            metrics.end_request(idx, generated_message)

        except openai.error.OpenAIError as e:
            logging.error(f"Request {idx} failed with OpenAI error: {e}")
            print(f"Request {idx} failed with OpenAI error: {e}")
            results.append({
                "index": idx,
                "input_prompt": prompt,
                "output_text": "",
                "error": str(e)
            })

    # Summarize results
    total_elapsed = sum(elapsed_times)
    average_request_time = total_elapsed / len(elapsed_times) if elapsed_times else 0
    success_count = sum(1 for r in results if "tokens_count" in r)

    summary = (
        f"\nTotal time for {len(results)} requests: {total_elapsed:.2f}s\n"
        f"Average request time: {average_request_time:.4f}s\n"
        f"Successful requests: {success_count}/{len(results)}"
    )
    logging.info(summary)
    print(summary)

    metrics_summary = metrics.get_metrics()

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Metrics:\n{json.dumps(metrics_summary, indent=2)}")
    print(f"Inference metrics saved to: {METRICS_FILE}")

    json_results = []
    for r in results:
        extracted_text = extract_text_output(r.get("output_text", ""))
        json_results.append({
            "input_text": r.get("input_prompt", ""),
            "output_text": extracted_text
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    logging.info(f"Inference results saved to: {OUTPUT_FILE}")
    print(f"Inference results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
