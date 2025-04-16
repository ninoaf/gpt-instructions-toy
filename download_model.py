from transformers import AutoTokenizer, AutoModelForCausalLM

def download_gpt2xl():
    #model_name = "gpt2-xl"
    model_name = "gpt2-large"

    print(f"🔄 Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"⬇️  Downloading model weights for {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("✅ Download complete. Cached locally in HuggingFace cache directory.")
    print(f"📂 Location: ~/.cache/huggingface/ or set HF_HOME to customize")


if __name__ == "__main__":
    download_gpt2xl()

'''
⬇️  Downloading model weights for gpt2-xl...
✅ Download complete. Cached locally in HuggingFace cache directory.
📂 Location: ~/.cache/huggingface/ or set HF_HOME to customize
'''