import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Load tokenizer and model ---
base_model_name = "gpt2-large"  # or "gpt2-xl" if that's what you trained on



tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for GPT-2

print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

model.eval()

# --- Prompt formatting ---
def make_prompt(instruction, input_text=""):
    prompt = f"### Instruction:\n{instruction.strip()}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text.strip()}\n\n"
    prompt += "### Response:\n"
    return prompt

# --- Inference ---
def generate_response(instruction, input_text=""):
    prompt = make_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(prompt):]
    return response.strip()

# --- Terminal interaction ---
if __name__ == "__main__":
    print(f"\n💬 Base model {base_model_name}. Type your instruction below.\nPress Ctrl+C to exit.\n")

    try:
        while True:
            instruction = input("📝 Instruction: ").strip()
            if not instruction:
                continue

            input_text = input("📥 (Optional) Input context: ").strip()

            print("\n🤖 Generating response...\n")
            response = generate_response(instruction, input_text)
            print("🗨️  Response:\n" + response + "\n" + "-"*50 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Exiting. Have a great day!")


'''
📝 Instruction:
Get the current temperature for Tokyo.

📥 (Optional):
Respond in JSON format like:
{"tool": "get_weather", "args": {"location": "CityName"}}

📝 Instruction:
Translate to JSON format:
Get the weather in Paris.

📥 (Optional) Input:
Example:
Instruction: Get the weather in London.
Output: {"tool": "get_weather", "args": {"location": "London"}}

Now:
Instruction: Get the weather in Paris. Generate only one JSON object.

'''