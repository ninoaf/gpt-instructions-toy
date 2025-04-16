import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
import random

# ---- Load and Format Alpaca Data ----
class AlpacaDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=400):
        with open(path, "r") as f:
            raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for item in raw_data:
            prompt = f"### Instruction:\n{item['instruction']}\n\n"
            if item["input"]:
                prompt += f"### Input:\n{item['input']}\n\n"
            prompt += f"### Response:\n{item['output']}"
            self.data.append(prompt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.data[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.squeeze() for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

def print_rnd_element_dataset(dataset,tokenizer):
    # Choose a random index within the subset
    random_idx = random.randint(0, len(dataset) - 1)

    # Get the sample
    sample = dataset[random_idx]

    # Decode the input_ids back to text
    decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    decoded_labels = tokenizer.decode(sample["labels"], skip_special_tokens=True)

    print(f"🧾 Input:\n{decoded_input}\n")
    print(f"🎯 Label:\n{decoded_labels}")

def main():
    model_name = "gpt2-large"
    dataset_name = "alpaca_data"
    model_name_sufix = "-test"
    alpaca_path = f"data/{dataset_name}.json"  # or full path if needed

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is defined

    print(f"⬇️  Loading {model_name} model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Only if using bitsandbytes
        device_map="auto"
    )

    print("🔧 Applying LoRA adaptation...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("📚 Preparing dataset...")
    dataset_all = AlpacaDataset(alpaca_path, tokenizer)
    dataset = torch.utils.data.Subset(dataset_all, list(range(1000)))
    print_rnd_element_dataset(dataset,tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("⚙️ Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=f"./finetuned_models/{model_name}-lora-{dataset_name}{model_name_sufix}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none",  # Set to "wandb" if using Weights & Biases
    )

    print("🚀 Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"✅ Training complete on instruction following dataset {dataset_name}{model_name_sufix} .")

    #=========== contine training on structured_alpaca_1000=================
    dataset_name2 = "structured_alpaca_1000"
    alpaca_path = f"data/{dataset_name2}.json"  # or full path if needed

    print("📚 Preparing dataset...")
    dataset_all = AlpacaDataset(alpaca_path, tokenizer)
    dataset = torch.utils.data.Subset(dataset_all, list(range(100)))
    print_rnd_element_dataset(dataset,tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("⚙️ Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=f"./finetuned_models/{model_name}-lora-{dataset_name2}{model_name_sufix}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none",  # Set to "wandb" if using Weights & Biases
    )

    print("🚀 Contine training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"✅ Training complete on instruction following dataset {dataset_name2}{model_name_sufix}.")

    # Save final adapter weights after training
    print("💾 Saving final LoRA adapter...")
    model.save_pretrained(f"./finetuned_models/{model_name}-lora-{dataset_name}-{dataset_name2}{model_name_sufix}-final")
    tokenizer.save_pretrained(f"./finetuned_models/{model_name}-lora-{dataset_name}-{dataset_name2}{model_name_sufix}-final")

if __name__ == "__main__":
    main()
