# 02_finetuning.ipynb
# Fine-tuning open-source LLM using QLoRA on synthetic domain name dataset

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# ─── 1. Load Dataset ──────────────────────────────────────────────────────

df = pd.read_csv("data/synthetic_domains_v1.csv")

# Format for language modeling
df["prompt"] = "Suggest 3 domain names for the following business: " + df["business_description"]
df["completion"] = df["suggested_domain"]

# Combine into training text format
df["text"] = df["prompt"] + "\nDomain Suggestion: " + df["completion"]

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["text"]])

# ─── 2. Model & Tokenizer ─────────────────────────────────────────────────

MODEL_NAME = "tiiuae/falcon-7b-instruct"  # or try mistralai/Mistral-7B-Instruct
from huggingface_hub import login, Repository
login(token="xxhxxfxx_NDsxUseBjiSxbsDzziXFCpeRcItmKGQBgm")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ─── 3. LoRA Config (QLoRA/PEFT) ──────────────────────────────────────────

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─── 4. Tokenization & Data Prep ──────────────────────────────────────────

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ─── 5. Training Arguments ────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir="./finetuned_domain_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_dir="./logs",
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# ─── 6. Trainer Setup ─────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# ─── 7. Save Final Model ──────────────────────────────────────────────────

model.save_pretrained("finetuned_domain_model")
tokenizer.save_pretrained("finetuned_domain_model")

print("✅ Fine-tuning complete. Model saved to ./finetuned_domain_model")
