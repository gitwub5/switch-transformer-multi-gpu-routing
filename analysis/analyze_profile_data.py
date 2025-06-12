import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
sns.set(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False

# 1. 데이터 로드
df = pd.read_csv("../outputs/merged_latency_comparison.csv")
print(f"🔍 전체 행 수: {len(df)}\n")

# 2. Expert 수 확인
df["layer_expert"] = df["layer_type"].astype(str) + "_" + df["layer_index"].astype(str) + "_" + df["expert_id"].astype(str)
print(f"🔍 Expert 수: {len(df['layer_expert'].unique())}")

# 3. token_count 분포 저장 (범위 1~512)
token_min = df["token_count"].min()
token_max = df["token_count"].max()
print(f"🔍 토큰 최소값: {token_min}, 최대값: {token_max}")

plt.figure(figsize=(10, 6))
sns.histplot(df["token_count"], bins=token_max, kde=True)
plt.title("Token Count Distribution")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.savefig("token_count_distribution.png")


# 4. layer_expert별 token_count 분포 확인