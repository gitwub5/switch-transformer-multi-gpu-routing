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

# 2. latency_1080Ti, latency_A6000 분포 확인
def plot_latency_distribution(df, gpu_type):
    max_latency = df[f"latency_{gpu_type}"].max()
    min_latency = df[f"latency_{gpu_type}"].min()
    print(f"🔍 {gpu_type} 최대 latency: {max_latency}, 최소 latency: {min_latency}\n")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[f"latency_{gpu_type}"], bins=50, kde=True) 
    plt.title(f"Latency Distribution for {gpu_type}")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.show()

plot_latency_distribution(df, "1080Ti")
plot_latency_distribution(df, "A6000")

# 3. Expert 수 확인
df["layer_expert"] = df["layer_type"].astype(str) + "_" + df["layer_index"].astype(str) + "_" + df["expert_id"].astype(str)
print(f"🔍 Expert 수: {len(df['layer_expert'].unique())}")
