#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from openai import OpenAI, AsyncOpenAI

# -----------------------------
# Basic configuration
# -----------------------------
SERVER_URL = os.getenv("TRITON_SERVER_URL", "http://localhost:8080/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "ensemble")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 4000))
OUTPUT_FILE = "result/llama3.2_1b_async.json"
METRICS_FILE = "llama3.2_1b_inference_metrics_async.txt"
MAX_TOKENS = 128
MAX_CONCURRENT_REQUESTS = 32
RETRIES = 1

os.makedirs("result", exist_ok=True)

# Async client
async_client = AsyncOpenAI(
    base_url=SERVER_URL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
)

# Sync client
sync_client = OpenAI(
    base_url=SERVER_URL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

# -----------------------------
# Helper functions
# -----------------------------
def build_prompt(question, context):
    return f"Context: {context}\nQuestion: {question}\nAnswer:"

def count_tokens(text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except:
        return len(text)

# -----------------------------
# Metrics Tracker
# -----------------------------
class MetricsTracker:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.token_counts = {}

    def start_request(self, req_id):
        self.start_times[req_id] = time.time()

    def end_request(self, req_id, output_text):
        self.end_times[req_id] = time.time()
        self.token_counts[req_id] = count_tokens(output_text)

    def summary(self):
        if not self.start_times or not self.end_times:
            return {}
        total_elapsed = max(self.end_times.values()) - min(self.start_times.values())
        total_tokens = sum(self.token_counts.values())
        total_requests = len(self.end_times)
        avg_latency = sum(self.end_times[r] - self.start_times[r] for r in self.end_times) / total_requests
        return {
            "total_requests": total_requests,
            "total_elapsed_seconds": round(total_elapsed, 3),
            "average_latency_seconds": round(avg_latency, 3),
            "total_tokens": total_tokens,
            "tokens_per_second": round(total_tokens / total_elapsed, 3) if total_elapsed else 0,
        }

# -----------------------------
# Asynchronous Implementation
# -----------------------------
async def send_request_openai_async(model, messages, req_id, results, metrics, semaphore):
    async with semaphore:
        metrics.start_request(req_id)
        for _ in range(RETRIES):
            try:
                resp = await async_client.chat.completions.create(
                    model=model, messages=messages, max_tokens=MAX_TOKENS
                )
                text = resp.choices[0].message.content.strip()
                metrics.end_request(req_id, text)
                print(f"[ASYNC] Request {req_id} completed")
                results.append({
                    "index": req_id,
                    "input": messages[-1]["content"],
                    "output": text
                })
                return
            except Exception as e:
                print(f"[ASYNC] Request {req_id} failed: {e}")

        # All retries failed
        results.append({
            "index": req_id,
            "input": messages[-1]["content"],
            "error": "All retries failed"
        })

async def main_async():
    dataset = load_dataset("squad", split="validation").select(range(NUM_SAMPLES))
    metrics = MetricsTracker()
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = []
    for i, row in enumerate(dataset):
        messages = [
            {"role": "system", "content": "Answer the following question in long details"},
            {"role": "user", "content": build_prompt(row["question"], row["context"])}
        ]
        t = asyncio.create_task(send_request_openai_async(MODEL_NAME, messages, i, results, metrics, sem))
        tasks.append(t)

    await asyncio.gather(*tasks)
    return results, metrics


# -----------------------------
# Synchronous Implementation
# -----------------------------
def send_request_openai_sync(model, messages, req_id, results, metrics):
    metrics.start_request(req_id)
    for _ in range(RETRIES):
        try:
            resp = sync_client.chat.completions.create(
                model=model, messages=messages, max_tokens=MAX_TOKENS
            )
            text = resp.choices[0].message.content.strip()
            metrics.end_request(req_id, text)
            print(f"[SYNC] Request {req_id} completed")
            return {
                "index": req_id,
                "input": messages[-1]["content"],
                "output": text
            }
        except Exception as e:
            print(f"[SYNC] Request {req_id} failed: {e}")

    return {
        "index": req_id,
        "input": messages[-1]["content"],
        "error": "All retries failed"
    }

def main_sync():
    dataset = load_dataset("squad", split="validation").select(range(NUM_SAMPLES))
    metrics = MetricsTracker()
    results = []

    for i, row in enumerate(dataset):
        messages = [
            {"role": "system", "content": "Answer the following question in long details"},
            {"role": "user", "content": build_prompt(row["question"], row["context"])}
        ]
        res = send_request_openai_sync(MODEL_NAME, messages, i, results, metrics)
        results.append(res)

    return results, metrics

# -----------------------------
# Common Save/Print
# -----------------------------
def save_and_print(results, metrics):
    summary = metrics.summary()
    print(f"\nRequests: {summary.get('total_requests', 0)}")
    print(f"Elapsed: {summary.get('total_elapsed_seconds', 0)}s, "
          f"Avg Latency: {summary.get('average_latency_seconds', 0)}s, "
          f"Total Tokens: {summary.get('total_tokens', 0)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")

# -----------------------------
# Entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["async", "sync"], default="async", help="Choose run mode")
    args = parser.parse_args()

    try:
        if args.mode == "sync":
            results, metrics = main_sync()
        else:
            results, metrics = asyncio.run(main_async())
        save_and_print(results, metrics)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
