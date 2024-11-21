import time
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Model path or name
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Number of samples to use
NUM_SAMPLES = 1000

# Batch size for inference
BATCH_SIZE = 32  # Adjust based on available VRAM

# Output JSON files
OUTPUT_FILE = "inference_results.json"
RUNTIME_FILE = "runtime_results.json"

# Initialize accelerator
accelerator = Accelerator()

# Initialize model and tokenizer
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Set the padding token to the EOS token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model = model.to(accelerator.device)

# Load the SQuAD dataset
print("Loading the SQuAD dataset...")
dataset = load_dataset("squad_v2", split="validation")
questions = dataset["question"][:NUM_SAMPLES]  # Extract the first NUM_SAMPLES questions

# Split the data across devices
print("Distributing prompts across devices...")
with accelerator.split_between_processes(questions, apply_padding=True) as prompts:
    print(f"Device {accelerator.process_index} received {len(prompts)} prompts.")

    # Perform batched inference
    input_ids_list = []
    output_ids_list = []
    start_time = time.time()

    # Process data in batches
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"Processing on device {accelerator.process_index}"):
        batch = prompts[i : i + BATCH_SIZE]

        # Tokenize the batch
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048, pad_token_id=tokenizer.pad_token_id)

        # Collect input and output tensors
        input_ids_list.extend(inputs["input_ids"])
        output_ids_list.extend(outputs)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / NUM_SAMPLES
    print(f"Device {accelerator.process_index} finished processing in {total_time:.2f} seconds.")
    print(f"Average runtime per question: {avg_time:.4f} seconds.")

# Now, pad sequences to the same length
from torch.nn.utils.rnn import pad_sequence

# Convert lists of tensors to tensors and pad them
input_ids_tensor = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
output_ids_tensor = pad_sequence(output_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)

# Pad tensors across processes to the same length
input_ids_tensor = accelerator.pad_across_processes(input_ids_tensor, dim=1, pad_index=tokenizer.pad_token_id)
output_ids_tensor = accelerator.pad_across_processes(output_ids_tensor, dim=1, pad_index=tokenizer.pad_token_id)

# Gather tensors across devices
print("Gathering results across devices...")
gathered_input_ids = accelerator.gather(input_ids_tensor)
gathered_output_ids = accelerator.gather(output_ids_tensor)

# Decode gathered tensors into strings on the main process
if accelerator.is_main_process:
    print("Decoding gathered results...")
    # Move tensors to CPU
    gathered_input_ids = gathered_input_ids.cpu()
    gathered_output_ids = gathered_output_ids.cpu()

    # Decode the input and output IDs
    input_texts = tokenizer.batch_decode(gathered_input_ids, skip_special_tokens=True)
    output_texts = tokenizer.batch_decode(gathered_output_ids, skip_special_tokens=True)

    # Combine inputs and outputs
    final_results = [{"input": inp, "output": out} for inp, out in zip(input_texts, output_texts)]

    # Save inference results to a JSON file
    print(f"Saving inference results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print("Inference results saved.")

    # Save runtime statistics to a separate JSON file
    print(f"Saving runtime statistics to {RUNTIME_FILE}...")
    runtime_results = {
        "total_time_seconds": total_time,
        "average_time_per_question_seconds": avg_time,
    }
    with open(RUNTIME_FILE, "w", encoding="utf-8") as f:
        json.dump(runtime_results, f, ensure_ascii=False, indent=2)
    print("Runtime statistics saved.")
