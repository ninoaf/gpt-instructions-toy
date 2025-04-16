import os
import shutil

SOURCE_DIR = "./gpt2xl-lora-alpaca"     # where your checkpoints are saved
DEST_DIR = "./gpt2xl-lora-alpaca-final" # where to export the adapter

# Find latest checkpoint
checkpoints = [
    os.path.join(SOURCE_DIR, d)
    for d in os.listdir(SOURCE_DIR)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(SOURCE_DIR, d))
]

if not checkpoints:
    raise ValueError("❌ No checkpoints found in the directory.")

# Sort by step number
checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
latest_ckpt = checkpoints[-1]
print(f"✅ Found latest checkpoint: {latest_ckpt}")

# Make sure destination exists
os.makedirs(DEST_DIR, exist_ok=True)

# Copy adapter files
for filename in ["adapter_model.bin", "adapter_config.json"]:
    src_file = os.path.join(latest_ckpt, filename)
    dst_file = os.path.join(DEST_DIR, filename)
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"📦 Copied {filename} to {DEST_DIR}")
    else:
        print(f"⚠️ {filename} not found in checkpoint.")

print("✅ Adapter exported. Ready for inference with PEFT!")
