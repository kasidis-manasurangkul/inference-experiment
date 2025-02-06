import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Put the model in evaluation mode
model.eval()

# Define export path
export_path = "onnx/model.onnx"

# Example input for the model
text = "Translate this sentence to Spanish."
inputs = tokenizer(text, return_tensors="pt")

# Prepare inputs as a tuple
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
input_tuple = (input_ids, attention_mask)

torch.onnx.export(
    model,
    input_tuple,
    export_path,
    export_params=True,
    opset_version=14,  # Adjust if needed for compatibility
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    },
    keep_initializers_as_inputs=False  # Embed initializers in the model
)
