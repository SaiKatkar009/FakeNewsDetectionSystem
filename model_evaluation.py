import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/cleaned_final_dataset.csv")

print("\n📊 DATASET SUMMARY")
print(f"Total rows: {len(df)}")
print("Label distribution:\n", df["label"].value_counts())

# Convert labels
df["label"] = df["label"].map({"fake": 0, "real": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=42,
    stratify=df["label"]
)

print(f"\n📌 Train size: {len(X_train)}")
print(f"📌 Test size: {len(X_test)}")

# -----------------------------
# 2. Load Model + Tokenizer
# -----------------------------
MODEL_PATH = "model/bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"\n🖥 Using device: {device}")

# -----------------------------
# 3. Tokenize Test Data
# -----------------------------
encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# -----------------------------
# 4. Predict
# -----------------------------
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# -----------------------------
# 5. Metrics
# -----------------------------
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print("\n=====================")
print("📌 MODEL EVALUATION")
print("=====================")
print(f"✔ Accuracy:  {accuracy:.4f}")
print(f"✔ Precision: {precision:.4f}")
print(f"✔ Recall:    {recall:.4f}")
print(f"✔ F1-score:  {f1:.4f}")

print("\n📌 Classification Report:\n")
print(classification_report(y_test, preds, target_names=["fake", "real"]))

print("\n📌 Confusion Matrix:")
print(cm)

# -----------------------------
# 6. Additional Info
# -----------------------------
print("\n📘 TRAINING DETAILS")
print("Epochs used for fine-tuning: 6")   # (Your final training used 6 epochs)
print(f"Total dataset size: {len(df)}")
print(f"Fake samples: {df['label'].value_counts()[0]}")
print(f"Real samples: {df['label'].value_counts()[1]}")

print("\n✅ Evaluation Complete!")
