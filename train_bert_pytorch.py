import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch

# ---------------------------------------------
# 1. LOAD & CLEAN DATASET
# ---------------------------------------------
df = pd.read_csv("data/processed/cleaned_final_dataset.csv")

# Keep only necessary columns
df = df[["text", "label"]].dropna()

# Map labels
df["label"] = df["label"].map({"fake": 0, "real": 1})

# SHUFFLE BEFORE SPLIT (Fix for 100% accuracy)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset loaded and shuffled.")
print(df["label"].value_counts())

# ---------------------------------------------
# 2. TRAIN/TEST SPLIT (STRATIFIED)
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------
# 3. TOKENIZER
# ---------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize full lists directly
train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=256
)

# ---------------------------------------------
# 4. TORCH DATASET
# ---------------------------------------------
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_ds = TorchDataset(train_encodings, y_train)
test_ds = TorchDataset(test_encodings, y_test)

print("Tokenization & dataset conversion complete.")

# ---------------------------------------------
# 5. LOAD MODEL
# ---------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "fake", 1: "real"},
    label2id={"fake": 0, "real": 1}
)

# ---------------------------------------------
# 6. TRAINING ARGUMENTS
# ---------------------------------------------
training_args = TrainingArguments(
    output_dir="model/bert",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    lr_scheduler_type="linear",
    save_total_limit=2,
)

# ---------------------------------------------
# 7. DATA COLLATOR
# ---------------------------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------------------------
# 8. TRAINER
# ---------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
)

# ---------------------------------------------
# 9. TRAIN
# ---------------------------------------------
trainer.train()

# ---------------------------------------------
# 10. SAVE MODEL & TOKENIZER
# ---------------------------------------------
trainer.save_model("model/bert")
tokenizer.save_pretrained("model/bert")

print("\n🔥 BERT successfully trained & saved using GPU with proper shuffled split!")
