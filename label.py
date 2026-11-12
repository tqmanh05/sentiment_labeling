from pathlib import Path

import pandas as pd
import torch
from pyvi import ViTokenizer
from transformers import AutoTokenizer, RobertaForSequenceClassification

IN_PATH = "data/processed/merged_reviews.csv"
OUT = Path("data/labeled")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "wonrax/phobert-base-vietnamese-sentiment"
THRESH = 0.80      
BATCH = 32
MAX_LEN = 256

df = pd.read_csv(IN_PATH)
print("Số review:", len(df))

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = RobertaForSequenceClassification.from_pretrained(MODEL)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
id2label = model.config.id2label
print("Nhãn của mô hình:", id2label, "| thiết bị:", device)

# Tách từ theo yêu cầu của PhoBERT
texts = [ViTokenizer.tokenize(str(t)) for t in df["text"]]

probs = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        p = model(**enc).logits.softmax(dim=-1).cpu().tolist()
        probs.extend(p)
        if (i // BATCH) % 20 == 0:
            print(f"  đã xử lý {min(i + BATCH, len(texts))}/{len(texts)}")

names = [id2label[i] for i in range(len(id2label))]   
pdf = pd.DataFrame(probs, columns=names)
res = pd.concat([df[["id", "category", "text"]], pdf], axis=1)

def decide(row):
    if row["POS"] >= THRESH:
        return 1
    if row["NEG"] >= THRESH:
        return -1
    return None   

res["label"] = res.apply(decide, axis=1)
res["confidence"] = res[["POS", "NEG"]].max(axis=1)

res.to_csv(OUT / "all_predictions.csv", index=False, encoding="utf-8-sig")
good = res[res["label"].notna()].copy()
good["label"] = good["label"].astype(int)
good[["id", "category", "text", "label", "confidence"]].to_csv(
    OUT / "dataset_labeled.csv", index=False, encoding="utf-8-sig")
res[res["label"].isna()][["id", "category", "text"] + names].to_csv(
    OUT / "uncertain.csv", index=False, encoding="utf-8-sig")

print("\nĐã gán nhãn:", len(good), good["label"].value_counts().to_dict())
print("Loại (trung tính/không chắc):", len(res) - len(good))