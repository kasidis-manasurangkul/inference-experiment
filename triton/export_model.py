from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model in a directory format compatible with Triton
model.save_pretrained("models/llama-3.1-8B/")
tokenizer.save_pretrained("models/llama-3.1-8B/")
