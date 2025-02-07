#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from openai import OpenAI  # For sync mode
from openai import AsyncOpenAI  # For async mode

# Configuration
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1000))
OUTPUT_FILE = "result/llama3.2_1b_async.json"
METRICS_FILE = "llama3.2_1b_inference_metrics_async.txt"
MAX_TOKENS = 2048
MAX_CONCURRENT_REQUESTS = 2048
RETRIES = 1

# Ensure result folder
os.makedirs("result", exist_ok=True)

# Async client
async_client = AsyncOpenAI(
    base_url=VLLM_SERVER_URL, 
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
)

# Sync client
sync_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=VLLM_SERVER_URL,
)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", 
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

def build_prompt(q, c):
    return f"Context: {c}\nQuestion: {q}\nAnswer:"

def count_tokens(text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except:
        # Fallback if tokenizer fails for any reason
        return len(text)

class MetricsTracker:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.token_counts = {}

    def start_request(self, rid):
        # Use perf_counter for high-resolution timing
        self.start_times[rid] = time.perf_counter()

    def end_request(self, rid, output_text):
        # Use perf_counter for high-resolution timing
        self.end_times[rid] = time.perf_counter()
        self.token_counts[rid] = count_tokens(output_text)

    def summary(self):
        if not self.start_times or not self.end_times:
            return {}
        # Calculate total elapsed time using the min start time and max end time
        total_elapsed = max(self.end_times.values()) - min(self.start_times.values())
        total_tokens = sum(self.token_counts.values())
        total_requests = len(self.end_times)
        # Average latency across all requests
        avg_latency = sum(
            self.end_times[r] - self.start_times[r] for r in self.end_times
        ) / total_requests if total_requests > 0 else 0.0

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
                    model=model, 
                    messages=messages, 
                    max_tokens=MAX_TOKENS
                )
                txt = resp.choices[0].message.content.strip()
                metrics.end_request(req_id, txt)
                print(f"[ASYNC] Request {req_id} completed")
                results.append({
                    "index": req_id,
                    "input": messages[-1]["content"],
                    "output": txt
                })
                return
            except Exception as e:
                print(f"[ASYNC] Request {req_id} failed: {e}")
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
            {"role": "system", "content": "Answer the following question in details"},
            {"role": "user", "content": build_prompt(row["question"], row["context"])}
        ]
        task = asyncio.create_task(send_request_openai_async(MODEL_NAME, messages, i, results, metrics, sem))
        tasks.append(task)

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
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            txt = resp.choices[0].message.content.strip()
            metrics.end_request(req_id, txt)
            print(f"[SYNC] Request {req_id} completed")
            return {
                "index": req_id,
                "input": messages[-1]["content"],
                "output": txt
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
            {"role": "system", "content": "Answer the following question in details"},
            {"role": "user",   "content": build_prompt(row["question"], row["context"])}
        ]
        res = send_request_openai_sync(MODEL_NAME, messages, i, results, metrics)
        results.append(res)

    return results, metrics

# -----------------------------
# Shared Entry Point
# -----------------------------
def save_output(results, metrics):
    summary = metrics.summary()
    print(f"\nRequests: {summary.get('total_requests', 0)}")
    print(f"Elapsed: {summary.get('total_elapsed_seconds', 0)}s, "
          f"Avg Latency: {summary.get('average_latency_seconds', 0)}s, "
          f"Total Tokens: {summary.get('total_tokens', 0)}, "
          f"TPS: {summary.get('tokens_per_second', 0)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")

def main():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", 
        choices=["async", "sync"], 
        default="async", 
        help="Choose run mode"
    )
    args = parser.parse_args()

    try:
        if args.mode == "sync":
            results, metrics = main_sync()
        else:
            results, metrics = asyncio.run(main_async())
        save_output(results, metrics)

        # For debugging and detailed timing breakdown
        print(len(list(metrics.start_times.keys())))
        total_time = 0
        total_tokens = 0
        with open("check.txt", "w") as f:
            for i in metrics.start_times:
                duration = metrics.end_times[i] - metrics.start_times[i]
                token_count = metrics.token_counts[i]
                total_time += duration
                total_tokens += token_count
                f.write(f"Request {i}: start: {metrics.start_times[i]} "
                        f"end: {metrics.end_times[i]} "
                        f"diff: {duration} "
                        f"token_count: {token_count}\n")
            f.write(f"Total time: {total_time}\n")
            f.write(f"Total tokens: {total_tokens}\n")
            if total_time > 0:
                f.write(f"Tokens/sec: {total_tokens / total_time}\n")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
