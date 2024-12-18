import datasets
import re

# Removed the OpenAI import since we'll switch to using Optimum-NVIDIA
# from openai import OpenAI

# Import from transformers and optimum-nvidia
from transformers import AutoTokenizer
from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.pipelines import pipeline

def is_numeric_choice(choice):
    # Remove quotes and whitespace
    cleaned = choice.strip('" \'')
    # Check if the choice contains only numbers, dots, and commas
    return bool(re.match(r'^[\d.,]+$', cleaned))

def format_choices_prompt(choices):
    formatted_choices = []
    for i, choice in enumerate(choices):
        if is_numeric_choice(choice):
            # Keep numeric choices as is
            formatted_choices.append(f"ตัวเลือกที่ {i+1}: {choice} (ตัวเลข - คงไว้ตามเดิม)")
        else:
            # Mark non-numeric choices for translation
            formatted_choices.append(f"ตัวเลือกที่ {i+1}: {choice} (แปล)")
    return "\n".join(formatted_choices)

# Load the dataset
dataset = datasets.load_dataset("cais/mmlu", "all")

# Set up the tokenizer and model using Optimum-NVIDIA
# Note: Ensure that your environment has compatible GPU and drivers for FP8 
# and that "Qwen/Qwen2.5-72B-Instruct" model is available locally or from HF Hub.
#
# If FP8 is not supported by your GPU, you can remove the `use_fp8=True` argument.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct", padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    use_fp8=False,            # Remove if not supported on your GPU
    max_prompt_length=1024,
    max_output_length=2048,
    max_batch_size=8,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="auto",  # Adjust if multiple GPUs or CPU
    top_k=40,
    top_p=0.7,
    repetition_penalty=10,
    max_new_tokens=1024
)

for entry in dataset['test']:
    question = entry['question']
    choices = entry['choices']  # Get the choices array

    # Create the prompt with both question and choices
    prompt = f"""คำถาม: {question}

ตัวเลือก:
{format_choices_prompt(choices)}

คำแนะนำ:
1. แปลคำถามเป็นภาษาไทย
2. สำหรับตัวเลือก:
   - ถ้ามีเครื่องหมาย 'ตัวเลข - คงไว้ตามเดิม' ให้คงตัวเลขไว้ตามเดิม
   - ถ้ามีเครื่องหมาย 'แปล' ให้แปลข้อความเป็นภาษาไทย
3. ดูตัวอย่างต่อไปนี้:

ตัวอย่างที่ 1 (เฉพาะตัวเลข):
คำถาม: What is the square root of 16?
ตัวเลือก:
ตัวเลือกที่ 1: "2" (ตัวเลข - คงไว้ตามเดิม)
ตัวเลือกที่ 2: "4" (ตัวเลข - คงไว้ตามเดิม)
ตัวเลือกที่ 3: "8" (ตัวเลข - คงไว้ตามเดิม)
ตัวเลือกที่ 4: "16" (ตัวเลข - คงไว้ตามเดิม)

คำตอบที่ต้องการ:
คำถาม: รากที่สองของ 16 คือเท่าไร?

ตัวเลือก:
1. "2"
2. "4"
3. "8"
4. "16"

ตัวอย่างที่ 2 (ผสมระหว่างข้อความและตัวเลข):
คำถาม: Which mathematical operation should be performed first?
ตัวเลือก:
ตัวเลือกที่ 1: "Addition" (แปล)
ตัวเลือกที่ 2: "2.5" (ตัวเลข - คงไว้ตามเดิม)
ตัวเลือกที่ 3: "Multiplication" (แปล)
ตัวเลือกที่ 4: "Division" (แปล)

คำตอบที่ต้องการ:
คำถาม: ควรทำการดำเนินการทางคณิตศาสตร์ใดก่อน?

ตัวเลือก:
1. "การบวก"
2. "2.5"
3. "การคูณ"
4. "การหาร"

ตัวอย่างที่ 3 (เฉพาะข้อความ):
คำถาม: Which of the following is a prime number?
ตัวเลือก:
ตัวเลือกที่ 1: "First option" (แปล)
ตัวเลือกที่ 2: "Second option" (แปล)
ตัวเลือกที่ 3: "Third option" (แปล)
ตัวเลือกที่ 4: "Fourth option" (แปล)

คำตอบที่ต้องการ:
คำถาม: จำนวนใดต่อไปนี้เป็นจำนวนเฉพาะ?

ตัวเลือก:
1. "ตัวเลือกที่หนึ่ง"
2. "ตัวเลือกที่สอง"
3. "ตัวเลือกที่สาม"
4. "ตัวเลือกที่สี่"

กรุณาแปลโจทย์นี้:
คำถาม: [คำแปลภาษาไทย]

ตัวเลือก:
1. [คำแปลภาษาไทยหรือตัวเลขตามเดิม]
2. [คำแปลภาษาไทยหรือตัวเลขตามเดิม]
3. [คำแปลภาษาไทยหรือตัวเลขตามเดิม]
4. [คำแปลภาษาไทยหรือตัวเลขตามเดิม]
"""

    # Incorporate the system role instructions directly into the prompt
    # Since we're no longer using the chat endpoint, we must include the 
    # "system" role message directly into the prompt.
    full_prompt = "System: คุณคือผู้แปลภาษาอังกฤษเป็นภาษาไทยมืออาชีพ\nUser: " + prompt

    # Generate the completion using the pipeline
    outputs = pipe(full_prompt)
    generated_text = outputs[0]['generated_text']

    print("ต้นฉบับ:")
    print(f"คำถาม: {question}")
    print("ตัวเลือก:", choices)
    print("\nคำแปล:")
    print(generated_text)
    print("\n" + "="*50 + "\n")
