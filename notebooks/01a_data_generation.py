# 01_data_generation.ipynb
# Synthetic Dataset Creation for Domain Name Generation LLM

import pandas as pd
import random
from faker import Faker

# Initialize faker
fake = Faker()

# Define business categories and sample keywords for domain generation
business_categories = {
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

# Generate synthetic business description
def generate_business_description(category, keywords):
    company = fake.company()
    phrase = fake.catch_phrase()
    services = ", ".join(random.sample(keywords, 2))
    location = fake.city()
    description = (
        f"{company} is a {phrase.lower()} business that specializes in {services} services in {location}."
    )
    return description

# Generate plausible domain name
def generate_domain_name(keywords):
    name = "".join(random.sample(keywords, 2))
    tld = random.choice([".com", ".net", ".org"])
    return (name + tld).lower()

# Generate dataset
data = []
samples_per_category = 100

for category, keywords in business_categories.items():
    for _ in range(samples_per_category):
        desc = generate_business_description(category, keywords)
        domain = generate_domain_name(keywords)
        data.append({
            "business_description": desc,
            "suggested_domain": domain,
            "category": category
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = "data/synthetic_domains_v1.csv"
df.to_csv(output_path, index=False)

# Preview
print(f"✅ Generated {len(df)} samples across {len(business_categories)} categories.")
df.head(10)
