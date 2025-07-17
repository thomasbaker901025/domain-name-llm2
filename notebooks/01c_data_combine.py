import pandas as pd

# Load original and augmented datasets
df_real = pd.read_csv("data/real_synthetic_dataset_v1.csv")
df_original = pd.read_csv("data/synthetic_domains_v1.csv")
df_augmented = pd.read_csv("data/augmented_domains_multi_method.csv")

# Combine the two DataFrames
df_combined = pd.concat([df_original, df_augmented, df_real], ignore_index=True)

# (Optional) Shuffle the data
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
df_combined.to_csv("data/synthetic_domains_combined.csv", index=False)

# Preview
print(f"✅ Combined dataset saved with {len(df_combined)} rows.")
df_combined.head()
