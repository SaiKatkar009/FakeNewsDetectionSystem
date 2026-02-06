import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
generated_path = "real_generated_1211.csv"  # your downloaded file
main_path = "data/processed/final_merged_dataset.csv"
save_path = "data/processed/super_final_merged_dataset.csv"

print("📥 Loading datasets...")

# ------------------------------------------------
# Load datasets
# ------------------------------------------------
df_gen = pd.read_csv(generated_path)
df_main = pd.read_csv(main_path)

print(f"Generated dataset: {len(df_gen)} rows")
print(f"Main dataset: {len(df_main)} rows")

# ------------------------------------------------
# Ensure correct labels (generate_real already has label='real')
# ------------------------------------------------
df_gen["label"] = df_gen["label"].astype(str)
df_main["label"] = df_main["label"].astype(str)

# ------------------------------------------------
# Merge
# ------------------------------------------------
print("🔄 Merging datasets...")
merged = pd.concat([df_main, df_gen], ignore_index=True)

# ------------------------------------------------
# Drop duplicates on title + text
# ------------------------------------------------
before = len(merged)
merged = merged.drop_duplicates(subset=["title", "text"], keep="first")
after = len(merged)

print(f"🧹 Removed {before - after} duplicate rows")

# ------------------------------------------------
# Save final dataset
# ------------------------------------------------
merged.to_csv(save_path, index=False, encoding="utf-8")

print(f"✅ Saved SUPER FINAL dataset → {save_path}")
print(f"📊 Total rows: {len(merged)}\n")

# ------------------------------------------------
# Class distribution
# ------------------------------------------------
print("🔍 Label Distribution:")
print(merged["label"].value_counts())
