import re
import unicodedata
from pathlib import Path

import pandas as pd

CSV_PATH = "data/raw/dmx_full_product_reviews.csv"
JSONL_PATH = "data/raw/reviews.jsonl"
OUT_DIR = Path("data/processed")
MIN_WORDS = 2  
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Đọc & đưa về cùng schema
a = pd.read_csv(CSV_PATH).rename(columns={"review_text": "text", "stars": "star"})
a["source"] = "csv"
b = pd.read_json(JSONL_PATH, lines=True, dtype={"product_id": str})
b["source"] = "jsonl"
b["author"] = None
COLS = ["source", "category", "product_url", "author", "text", "star"]
df = pd.concat([a[COLS], b[COLS]], ignore_index=True)
df["text_raw"] = df["text"]
df["reason"] = ""
n_raw = len(df)

# 2. Làm sạch
ADMIN = re.compile(
    r"\s*(?:Bộ phận \w+(?: \w+)?|Chăm sóc khách hàng|Hỗ trợ kỹ thuật)?\s*đã liên hệ hỗ trợ ngày \d{1,2}/\d{1,2}/\d{4}"
)
LIKE_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def clean(t):
    t = unicodedata.normalize("NFC", str(t))
    t = ADMIN.sub("", t)
    t = LIKE_SUFFIX.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


junk = (df["author"] == "Danh mục") | df["text"].str.contains(
    "Đăng nhập Voucher|Tất cả đánh giá|Tổng đài hỗ trợ", na=False
)
df.loc[junk, "reason"] = "rác (footer web)"
df["text"] = df["text"].map(clean)

# 3. Loại review không dùng được
m = df["reason"] == ""
df.loc[m & ~df["text"].str.contains(r"[^\W\d_]", regex=True), "reason"] = "không có chữ (rỗng/ký tự/emoji)"
m = df["reason"] == ""
df.loc[m & (df["text"].str.split().str.len() < MIN_WORDS), "reason"] = f"quá ngắn (<{MIN_WORDS} từ)"


def key(t):
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

df["_key"] = df["text"].map(key)
valid = df[df["reason"] == ""]
dup_idx = valid.index[valid.duplicated(subset="_key", keep="first")]
df.loc[dup_idx, "reason"] = "trùng lặp"

# 5. Xuất file
removed = df[df["reason"] != ""]
good = df[df["reason"] == ""].copy()
good["n_words"] = good["text"].str.split().str.len()
good = good.reset_index(drop=True)
good.insert(0, "id", range(1, len(good) + 1))
good["label"] = ""  
out = good[["id", "source", "category", "product_url", "text", "n_words", "label"]]
out.to_csv(OUT_DIR / "merged_reviews.csv", index=False, encoding="utf-8-sig")
removed[["source", "category", "text_raw", "reason"]].to_csv(
    OUT_DIR / "removed_reviews.csv", index=False, encoding="utf-8-sig"
)

report = f"""Tổng dòng 2 file : {n_raw}
Bị loại          : {len(removed)}
{removed['reason'].value_counts().to_string()}
DÙNG ĐƯỢC        : {len(out)}

Theo nguồn:
{out['source'].value_counts().to_string()}

Theo loại sản phẩm:
{out['category'].value_counts().to_string()}
"""
(OUT_DIR / "merge_report.txt").write_text(report, encoding="utf-8")
print(report)