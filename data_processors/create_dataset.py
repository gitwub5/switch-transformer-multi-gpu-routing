import re
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict

# ✅ 사용자 설정
MAX_SAMPLES = 5000
MIN_TOKEN_LENGTH = 1
MAX_TOKEN_LENGTH = 512
BUCKET_WIDTH = 10  # 10단위 버킷
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# ✅ 버킷 세팅
num_buckets = (MAX_TOKEN_LENGTH - MIN_TOKEN_LENGTH + 1) // BUCKET_WIDTH
samples_per_bucket = MAX_SAMPLES // num_buckets
buckets = defaultdict(list)

# ✅ 전처리 함수
def process_dataset(dataset, text_key):
    for ex in dataset:
        text = ex[text_key].strip().replace("\n", " ")
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_len = len(tokens)
        if MIN_TOKEN_LENGTH <= token_len <= MAX_TOKEN_LENGTH:
            bucket_id = (token_len - MIN_TOKEN_LENGTH) // BUCKET_WIDTH
            if len(buckets[bucket_id]) < samples_per_bucket:
                buckets[bucket_id].append({
                    "text": text,
                    "token_length": token_len
                })
        if sum(len(lst) for lst in buckets.values()) >= MAX_SAMPLES:
            break

# ✅ 데이터 수집
print("📦 AG News 수집 중...")
ag_dataset = load_dataset("ag_news", split="train[:100000]")
process_dataset(ag_dataset, "text")

print("📦 OpenWebText 수집 중...")
owt_dataset = load_dataset("openwebtext", split="train[:100000]")
process_dataset(owt_dataset, "text")

# ✅ 결과 정리
texts = []
for bucket in buckets.values():
    texts.extend(bucket)

random.shuffle(texts)
texts = texts[:MAX_SAMPLES]
texts = [{"text_id": f"S{i+1:04d}", "text": entry["text"], "token_length": entry["token_length"]} for i, entry in enumerate(texts)]

# ✅ 평균 토큰 수
avg_tokens = sum(t["token_length"] for t in texts) / len(texts)
print(f"✅ 총 문장 수: {len(texts)} | 평균 토큰 수: {avg_tokens:.2f}")

# ✅ 저장
os.makedirs("data", exist_ok=True)
output_path = f"data/structured_texts_balanced_{MAX_SAMPLES}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)
print(f"✅ 저장 완료: {output_path}")

# ✅ 시각화
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
token_lengths = [t["token_length"] for t in texts]
sns.histplot(token_lengths, bins=50, kde=True, color="skyblue")
plt.title("Token Length Distribution (Balanced Sample)")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/token_length_distribution.png")