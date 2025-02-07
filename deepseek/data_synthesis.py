#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import re
import time

from transformers import AutoTokenizer
from openai import OpenAI  # For sync mode
from openai import AsyncOpenAI  # For async mode

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "/mnt/data01/models_down/DeepSeek-R1")

REPEATS_PER_KB = 10
OUTPUT_FILE = "result/synthesis_data.json"
VALIDATED_OUTPUT_FILE = "result/synthesis_data_validated.json"
METRICS_FILE = "inference_metrics.txt"

MAX_TOKENS = 1024
MAX_CONCURRENT_REQUESTS = 800
TEMPERATRUE = 0.9
RETRIES = 3

os.makedirs("result", exist_ok=True)

# ---------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------
async_client = AsyncOpenAI(
    base_url=VLLM_SERVER_URL,
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
)
sync_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=VLLM_SERVER_URL
)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# ---------------------------------------------------------------------
# 1) Load system prompts
# ---------------------------------------------------------------------
def load_main_system_prompt():
    file_name = "main_system_prompt.txt"
    if not os.path.exists(file_name):
        print(f"[WARNING] system prompt file not found: {file_name}")
        return "You are a helpful model. Please follow the instructions."  # fallback
    with open(file_name, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_validator_system_prompt():
    file_name = "validator_system_promt.txt"
    if not os.path.exists(file_name):
        print(f"[WARNING] validator prompt file not found: {file_name}")
        return (
            "You are a validator model. Please validate and clean up the conversation."
        )  # fallback
    with open(file_name, "r", encoding="utf-8") as f:
        return f.read().strip()

# ---------------------------------------------------------------------
# 2) Remove chain-of-thought, remove markdown, remove emojis, etc.
# ---------------------------------------------------------------------
def remove_chain_of_thought(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def remove_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold markers
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # Remove single-star
    text = re.sub(r"`+", "", text)                # Remove backticks
    text = re.sub(r"_+", "", text)                # Remove underscores
    return text

def remove_emoji(text: str) -> str:
    # Remove codepoints from U+10000 to U+10FFFF
    return re.sub(r"[\U00010000-\U0010FFFF]+", "", text)

def post_process_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    cleaned_lines = []
    stop_processing = False

    for line in lines:
        if stop_processing:
            break

        stripped = line.strip()

        # Stop if "หมายเหตุ:" or "จุดเด่นของบทสนทนา:"
        if stripped.startswith("หมายเหตุ:") or stripped.startswith("จุดเด่นของบทสนทนา:"):
            stop_processing = True
            continue
        if stripped.startswith("หมายเหตุ :") or stripped.startswith("จุดเด่นของบทสนทนา :"):
            stop_processing = True
            continue

        # Skip lines with special tokens
        if re.match(r"^--- Conversation Record #\d+ \(KB=.*\) ---$", stripped):
            continue
        if stripped == "[Output]:":
            continue
        if stripped.startswith("---") or re.match(r"^=+$", stripped):
            continue

        no_emoji_line = remove_emoji(line)
        no_markdown_line = remove_markdown(no_emoji_line)

        if no_markdown_line.strip():
            cleaned_lines.append(no_markdown_line)

    return "".join(cleaned_lines).rstrip()

# ---------------------------------------------------------------------
# 3) Counting tokens, metrics
# ---------------------------------------------------------------------
def count_tokens(text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text)

class MetricsTracker:
    def __init__(self):
        self.start_times = {}
        self.end_times = {}
        self.token_counts = {}

    def start_request(self, rid):
        self.start_times[rid] = time.perf_counter()

    def end_request(self, rid, output_text):
        self.end_times[rid] = time.perf_counter()
        self.token_counts[rid] = count_tokens(output_text)

    def summary(self):
        if not self.start_times or not self.end_times:
            return {}
        total_elapsed = max(self.end_times.values()) - min(self.start_times.values())
        total_tokens = sum(self.token_counts.values())
        total_requests = len(self.end_times)
        if total_requests > 0:
            avg_latency = sum(
                self.end_times[r] - self.start_times[r]
                for r in self.end_times
            ) / total_requests
        else:
            avg_latency = 0.0
        return {
            "total_requests": total_requests,
            "total_elapsed_seconds": round(total_elapsed, 3),
            "average_latency_seconds": round(avg_latency, 3),
            "total_tokens": total_tokens,
            "tokens_per_second": (
                round(total_tokens / total_elapsed, 3) if total_elapsed else 0
            ),
        }

# ---------------------------------------------------------------------
# 4) Build prompts + load requests
# ---------------------------------------------------------------------
def build_prompt(kb_text):
    return f"หัวข้อสนทนา:\n{kb_text}\nโปรดสร้างบทสนทนาจำลอง 1 ตัวอย่าง"

def load_requests_data():
    kb_folder = "processed_KB"
    kb_files = [f for f in os.listdir(kb_folder) if f.endswith(".txt")]
    if not kb_files:
        print("[WARNING] No KB files found in 'processed_KB'!")
        return []

    requests_data = []
    req_id = 0
    # ตัวอย่าง: จำกัด kb_files[:10] หรือจะนำทั้งหมดก็ได้
    for kb_file in kb_files[:100]:
        kb_path = os.path.join(kb_folder, kb_file)
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_text = f.read()

        for _ in range(REPEATS_PER_KB):
            user_prompt = build_prompt(kb_text)
            requests_data.append((req_id, user_prompt, kb_file))
            req_id += 1

    print(f"[INFO] Prepared {len(requests_data)} total requests.")
    return requests_data

# ---------------------------------------------------------------------
# 5) Async calls for MAIN prompt
# ---------------------------------------------------------------------
async def send_request_openai_async(
    model,
    system_prompt,
    user_content,
    kb_filename,
    req_id,
    results,
    metrics,
    semaphore
):
    async with semaphore:
        metrics.start_request(req_id)
        for _ in range(RETRIES):
            try:
                resp = await async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATRUE
                )
                raw_txt = resp.choices[0].message.content.strip()

                # Remove chain-of-thought
                txt = remove_chain_of_thought(raw_txt)
                # Post-process
                txt = post_process_text(txt)

                metrics.end_request(req_id, txt)
                print(f"[ASYNC] Request {req_id} completed")
                results.append({
                    "index": req_id,
                    "kb_name": kb_filename,
                    "input": user_content,
                    "output": txt
                })
                return
            except Exception as e:
                print(f"[ASYNC] Request {req_id} failed: {e}")

        # If all retries fail
        results.append({
            "index": req_id,
            "kb_name": kb_filename,
            "input": user_content,
            "error": "All retries failed"
        })

async def main_async_inference():
    # 1) Load the MAIN system prompt
    system_prompt = load_main_system_prompt()

    # 2) Prepare requests
    requests_data = load_requests_data()
    metrics = MetricsTracker()
    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = []
    for (req_id, user_prompt, kb_filename) in requests_data:
        t = asyncio.create_task(
            send_request_openai_async(
                MODEL_NAME,
                system_prompt,
                user_prompt,
                kb_filename,
                req_id,
                results,
                metrics,
                sem
            )
        )
        tasks.append(t)

    await asyncio.gather(*tasks)
    return results, metrics

# ---------------------------------------------------------------------
# 6) Sync calls for MAIN prompt
# ---------------------------------------------------------------------
def send_request_openai_sync(
    model,
    system_prompt,
    user_content,
    kb_filename,
    req_id,
    results,
    metrics
):
    metrics.start_request(req_id)
    for _ in range(RETRIES):
        try:
            resp = sync_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATRUE
            )
            raw_txt = resp.choices[0].message.content.strip()

            # Remove chain-of-thought
            txt = remove_chain_of_thought(raw_txt)
            # Post-process
            txt = post_process_text(txt)

            metrics.end_request(req_id, txt)
            print(f"[SYNC] Request {req_id} completed")
            return {
                "index": req_id,
                "kb_name": kb_filename,
                "input": user_content,
                "output": txt
            }
        except Exception as e:
            print(f"[SYNC] Request {req_id} failed: {e}")

    return {
        "index": req_id,
        "kb_name": kb_filename,
        "input": user_content,
        "error": "All retries failed"
    }

def main_sync_inference():
    system_prompt = load_main_system_prompt()
    requests_data = load_requests_data()
    metrics = MetricsTracker()
    results = []

    for (req_id, user_prompt, kb_filename) in requests_data:
        r = send_request_openai_sync(
            MODEL_NAME,
            system_prompt,
            user_prompt,
            kb_filename,
            req_id,
            results,
            metrics
        )
        results.append(r)

    return results, metrics

# ---------------------------------------------------------------------
# 7) Validator pass (ASYNC): use the same model but with a different system prompt
# ---------------------------------------------------------------------
async def validate_responses_async(results):
    """
    เรียก validator prompt สำหรับทุก output ใน results (async)
    แล้วเก็บผลลัพธ์ validated_output ไว้ในแต่ละ item
    """
    validator_prompt = load_validator_system_prompt()
    metrics = MetricsTracker()
    validated_results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def validate_one(item, rid):
        original_text = item.get("output", "")
        if not original_text or "error" in item:
            # ถ้าไม่มี output หรือมี error อยู่แล้ว ก็ข้าม
            item["validated_output"] = original_text
            return item

        metrics.start_request(rid)
        for _ in range(RETRIES):
            try:
                resp = await async_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": validator_prompt},
                        {"role": "user", "content": original_text},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATRUE
                )
                raw_txt = resp.choices[0].message.content.strip()

                # Remove chain-of-thought
                txt = remove_chain_of_thought(raw_txt)
                # Post-process
                txt = post_process_text(txt)

                metrics.end_request(rid, txt)
                item["validated_output"] = txt
                print(f"[ASYNC] Request {rid} completed")
                return item

            except Exception as e:
                print(f"[VALIDATOR] Request {rid} failed: {e}")

        # If all retries fail
        item["validated_output"] = "Validator Error"
        return item

    # สร้าง tasks แบบ async
    tasks = []
    for i, item in enumerate(results):
        # ใช้ semaphore คุมปริมาณ concurrent
        async def wrapped_validate_one(it=item, idx=i):
            async with sem:
                return await validate_one(it, idx)

        tasks.append(asyncio.create_task(wrapped_validate_one()))

    # รอให้ tasks ทั้งหมดเสร็จ
    validated_list = await asyncio.gather(*tasks)
    validated_results = list(validated_list)

    return validated_results, metrics


# ---------------------------------------------------------------------
# 8) Save output
# ---------------------------------------------------------------------
def save_output(results, metrics, outpath):
    summary = metrics.summary()
    print(f"\nRequests: {summary.get('total_requests', 0)}")
    print(
        f"Elapsed: {summary.get('total_elapsed_seconds', 0)}s, "
        f"Avg Latency: {summary.get('average_latency_seconds', 0)}s, "
        f"Total Tokens: {summary.get('total_tokens', 0)}, "
        f"TPS: {summary.get('tokens_per_second', 0)}"
    )

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved to {outpath}")

# ---------------------------------------------------------------------
# 9) Main entry
# ---------------------------------------------------------------------
async def main_validator_async(results):
    """
    ฟังก์ชันช่วยเพื่อเรียก validate_responses_async แบบ async
    แล้ว return (validated_results, metrics_validator)
    """
    validated_results, metrics_validator = await validate_responses_async(results)
    return validated_results, metrics_validator

def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["async", "sync"],
        default="async",
        help="Choose run mode"
    )
    parser.add_argument(
        "--skip_validator",
        action="store_true",
        help="If set, skip the validator pass"
    )
    args = parser.parse_args()

    try:
        # 1) Run main inference
        if args.mode == "sync":
            results, metrics_main = main_sync_inference()
        else:
            # ใช้ asyncio.run(...) เรียก main_async_inference
            results, metrics_main = asyncio.run(main_async_inference())

        print("\n[MAIN] Inference summary:")
        save_output(results, metrics_main, OUTPUT_FILE)

        # 2) เรียก validator pass ถ้าไม่ได้ skip
        if not args.skip_validator:
            print("\n[VALIDATOR] Starting validation pass...")

            # เรียกตัว validator async
            validated_results, metrics_validator = asyncio.run(main_validator_async(results))

            print("\n[VALIDATOR] Validation summary:")
            save_output(validated_results, metrics_validator, VALIDATED_OUTPUT_FILE)
        else:
            validated_results = results  # ถ้าข้าม validator ก็ใช้ results เดิม

        # 3) Debug
        print(f"[DEBUG] Count of start_times = {len(list(metrics_main.start_times.keys()))}")
        total_time = 0
        total_tokens = 0
        with open("check.txt", "w", encoding="utf-8") as f:
            for i in metrics_main.start_times:
                if i not in metrics_main.end_times:
                    f.write(f"Request {i} never completed (no end time)\n")
                    continue
                duration = metrics_main.end_times[i] - metrics_main.start_times[i]
                token_count = metrics_main.token_counts[i]
                total_time += duration
                total_tokens += token_count
                f.write(
                    f"Request {i}: start={metrics_main.start_times[i]}, "
                    f"end={metrics_main.end_times[i]}, diff={duration}, "
                    f"token_count={token_count}\n"
                )
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
