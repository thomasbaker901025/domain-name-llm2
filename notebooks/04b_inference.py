# 04_inference.ipynb
# Generate domain name suggestions using the fine-tuned LLM

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ─── 1. Load Fine-Tuned Model ─────────────────────────────────────────────

model_path = "./finetuned_domain_model"  # Path to your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# ─── 2. Load Descriptions for Inference ───────────────────────────────────

# Use your combined dataset and sample from it
df = pd.read_csv("synthetic_domains_combined.csv")
sample_prompts = df["business_description"].drop_duplicates().sample(n=5, random_state=42)

# Create prompt text for generation
prompt_df = pd.DataFrame({
    "business_description": sample_prompts,
    "prompt": sample_prompts.apply(lambda d: f"Suggest 3 domain names for the following business: {d}")
})

# ─── 3. Inference Function ────────────────────────────────────────────────

def generate_domains(prompt):
    try:
        output = generator(
            prompt,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )
        # Extract domain suggestion from model output
        generated_text = output[0]["generated_text"]
        return generated_text.split("Domain Suggestion:")[-1].strip()
    except Exception as e:
        print("Error generating for:", prompt)
        return "generation_failed"

# ─── 4. Generate All Predictions ──────────────────────────────────────────

prompt_df["generated_domain"] = prompt_df["prompt"].apply(generate_domains)

# ─── 5. Save to CSV ───────────────────────────────────────────────────────

prompt_df[["business_description", "generated_domain"]].to_csv("inference_results.csv", index=False)
print("✅ Inference complete. Saved to inference_results.csv")

# Preview
prompt_df.head(3)
