import pandas as pd
import os

# Load merged dataset
df = pd.read_csv("data/processed/merged_news_dataset.csv")

# Normalize labels
df["label"] = df["label"].str.lower().str.strip()

# Separate classes
fake_df = df[df["label"] == "fake"]
real_df = df[df["label"] == "real"]

print("Before balancing:")
print(df["label"].value_counts(), "\n")

# Upsample REAL to match FAKE
real_upsampled = real_df.sample(
    n=len(fake_df), 
    replace=True, 
    random_state=42
)

# Combine
balanced_df = pd.concat([fake_df, real_upsampled], ignore_index=True)

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("After balancing:")
print(balanced_df["label"].value_counts(), "\n")

# Save balanced dataset
output_path = "data/processed/balanced_news_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
balanced_df.to_csv(output_path, index=False)

print(f"✅ Balanced dataset saved at: {output_path}")
print(f"📊 Balanced dataset size: {len(balanced_df)} rows")
