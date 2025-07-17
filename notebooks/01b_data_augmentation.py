# 01b_data_augmentation.ipynb
# Multi-method augmentation for domain name generation dataset

import pandas as pd
import random

# Load original dataset
df = pd.read_csv("data/synthetic_domains_v1.csv")  # Make sure this file exists in the same directory

# Define category-specific keywords
business_keywords = {
    "Coffee Shop": ["organic", "brew", "beans", "espresso", "latte"],
    "Tech Startup": ["AI", "cloud", "data", "cyber", "quantum"],
    "Pet Grooming": ["pet", "fur", "paw", "groom", "tail"],
    "Online Education": ["learn", "academy", "course", "skill", "edu"],
    "Fitness Studio": ["fit", "gym", "core", "strength", "motion"],
    "Legal Services": ["law", "legal", "justice", "firm", "counsel"],
    "Sustainable Products": ["green", "eco", "earth", "sustain", "zero"],
    "Non-Profit Org": ["care", "impact", "hope", "community", "give"],
    "Bakery": ["bake", "bread", "cake", "oven", "sweet"],
    "Financial Advisor": ["invest", "wealth", "capital", "fund", "secure"]
}

# ─── AUGMENTATION METHODS ─────────────────────────────────────────────────

# 1. Hugging Face LLM-based Paraphrasing (placeholder simulation)
def hf_paraphrase(description):
    return [f"Paraphrased (HF): {description}"]

# 2. Keyword Swapping within same category
def keyword_swap(description, keywords):
    swapped = description
    for word in keywords:
        if word in description:
            new_word = random.choice([k for k in keywords if k != word])
            swapped = swapped.replace(word, new_word, 1)
            break
    return [f"Swapped: {swapped}"]

# 3. Light Noise Injection (e.g., filler words)
def inject_noise(text):
    words = text.split()
    insert_pos = random.randint(0, len(words)-1)
    words.insert(insert_pos, random.choice(["really", "absolutely", "totally"]))
    return ["Noised: " + " ".join(words)]

# 4. Template-based Rewrite
def rewrite_template(category, keywords):
    city = "Metropolis"
    return [f"A {category} company offering {keywords[0]} and {keywords[1]} services in {city}."]

# ─── APPLY AUGMENTATIONS ─────────────────────────────────────────────────

augmented_rows = []
sample_df = df.sample(n=100, random_state=42)  # You can increase n as needed

for _, row in sample_df.iterrows():
    desc = row["business_description"]
    domain = row["suggested_domain"]
    cat = row["category"]
    keywords = business_keywords.get(cat, ["value", "service"])

    # Apply all methods
    all_augmented = (
        hf_paraphrase(desc)
        + keyword_swap(desc, keywords)
        + inject_noise(desc)
        + rewrite_template(cat, random.sample(keywords, 2))
    )

    for aug in all_augmented:
        augmented_rows.append({
            "business_description": aug,
            "suggested_domain": domain,
            "category": cat
        })

# Convert to DataFrame
augmented_df = pd.DataFrame(augmented_rows)

# Save to CSV
augmented_df.to_csv("data/augmented_domains_multi_method.csv", index=False)

# Preview
print(f"✅ Augmentation complete. {len(augmented_df)} samples")
augmented_df.head()
