# notebooks/04_inference.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional: reduce fragmentation if using GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load sample business descriptions
df = pd.read_csv("data/synthetic_domains_combined.csv")
sample_prompts = df["business_description"].drop_duplicates().sample(n=50, random_state=42)
prompt_df = pd.DataFrame({
    "business_description": sample_prompts,
    "prompt": sample_prompts.apply(lambda d: f"Suggest 3 domain names for the following business: {d}")
})

# Load model and tokenizer
model_path = "./finetuned_domain_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Try GPU → fallback to CPU
try:
    print("🔍 Trying to load model on CUDA...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
    device = 0
except torch.cuda.OutOfMemoryError:
    print("⚠️ CUDA out of memory. Falling back to CPU.")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cpu")
    device = -1

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# Inference function
def generate_domains(prompt):
    try:
        outputs = generator(
            prompt,
            max_length=48,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1
        )
        return outputs[0]["generated_text"].split("Domain Suggestion:")[-1].strip()
    except Exception as e:
        print(f"Error on prompt: {prompt} → {e}")
        return "generation_failed"

# Generate predictions
prompt_df["generated_domain"] = prompt_df["prompt"].apply(generate_domains)

# Save result
prompt_df[["business_description", "generated_domain"]].to_csv("inference_results.csv", index=False)
print("✅ inference_results.csv saved.")
