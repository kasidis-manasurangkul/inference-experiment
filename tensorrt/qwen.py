import datasets
import re
import subprocess
import json
import time

def is_numeric_choice(choice):
    cleaned = choice.strip('" \'')
    return bool(re.match(r'^[\d.,]+$', cleaned))

def format_choices_prompt(choices):
    formatted_choices = []
    for i, choice in enumerate(choices):
        if is_numeric_choice(choice):
            formatted_choices.append(f"ตัวเลือกที่ {i+1}: {choice} (ตัวเลข - คงไว้ตามเดิม)")
        else:
            formatted_choices.append(f"ตัวเลือกที่ {i+1}: {choice} (แปล)")
    return "\n".join(formatted_choices)

# Load the dataset
dataset = datasets.load_dataset("cais/mmlu", "all")

# Set directories to your local paths (adjust as needed)
tokenizer_dir = "./tmp/Qwen2.5/72B-Instruct"
engine_dir = "./tmp/Qwen2.5/72B-Instruct/trt_engines/fp16/4-gpu"  # 4-GPU engines

outputs = []
start = time.time()

for entry in dataset['test'].select(range(3)):
    question = entry['question']
    choices = entry['choices']

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
    full_prompt = "System: คุณคือผู้แปลภาษาอังกฤษเป็นภาษาไทยมืออาชีพ\nUser: " + prompt

    # Now, instead of running directly, we'll use mpirun to run with 4 GPUs.
    # Make sure mpirun is available and correctly set up. Also ensure run.py supports multi-GPU inference.
    # Typically, you'd do something like:
    # mpirun -n 4 python ../run.py ...
    # To do this from Python, you can run mpirun as a subprocess:

    command = [
        "mpirun",
        "--allow-run-as-root",
        "-n", "4",
        "python3.10", "TensorRT-LLM/examples/run.py",
        "--input_text", full_prompt,
        "--max_output_len", "512",
        "--tokenizer_dir", tokenizer_dir,
        "--engine_dir", engine_dir
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    generated_text = result.stdout
    outputs.append(
        {
            "question": question,
            "choices": choices,
            "generated_text": generated_text
        }
    )

    print("ต้นฉบับ:")
    print(f"คำถาม: {question}")
    print("ตัวเลือก:", choices)
    print("\nคำแปล:")
    print(generated_text)
    print("\n" + "="*50 + "\n")

end = time.time()
# Save the outputs
with open("qwen_outputs.json", "w") as f:
    json.dump(outputs, f)
print(f"Time taken: {end - start:.2f} seconds")