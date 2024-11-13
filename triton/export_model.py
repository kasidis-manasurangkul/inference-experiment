import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx

# Define model name and export folder
model_name = "meta-llama/Llama-3.1-8B-Instruct"
export_folder = "llama3.1-8b-tensor-rt"
export_path = os.path.join(export_folder, "llama_8b.onnx")

# Create the folder if it doesn't exist
os.makedirs(export_folder, exist_ok=True)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input for the model
text = "Translate this sentence to Spanish."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Export the model to ONNX
torch.onnx.export(
    model,
    input_ids,
    export_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}}
)

# Verify the ONNX model by directly using the file path
onnx.checker.check_model(export_path)
print("ONNX model exported and verified successfully to", export_path)
