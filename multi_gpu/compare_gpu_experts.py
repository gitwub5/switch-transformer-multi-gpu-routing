import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

def compute_average_latency(expert, layer_type, token_range=range(1, 200, 2)):
    result = {
        "expert": expert,
        "layer_type": layer_type
    }
    token_log = np.log1p(np.array(token_range).reshape(-1, 1))

    for gpu in ["1080Ti", "A6000"]:
        # ✅ layer_type 하위 폴더까지 반영
        model_path = os.path.join(layer_type, gpu, f"{expert}_model.joblib")
        scaler_path = os.path.join(layer_type, gpu, f"{expert}_scaler.joblib")

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print(f"❌ {expert} model for {gpu} not found at {model_path}. Skipping...")
            result[gpu] = np.nan
            continue

        pred_scaled = model.predict(token_log)
        pred_log10 = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        latency = np.power(10, pred_log10)
        average_latency = latency.mean()
        result[gpu] = average_latency

    if np.isnan(result["1080Ti"]) or np.isnan(result["A6000"]):
        result["better_gpu"] = "N/A"
    else:
        result["better_gpu"] = "1080Ti" if result["1080Ti"] < result["A6000"] else "A6000"

    return result

# 평균 레이턴시 비교
df = pd.read_csv("../outputs/merged_latency_comparison.csv")
df["layer_expert_full"] = df["layer_type"] + "_" + df["layer_index"].astype(str) + "_" + df["expert_id"].astype(str)
unique_experts = df[["layer_type", "layer_expert_full"]].drop_duplicates()

results = []
for _, row in unique_experts.iterrows():
    res = compute_average_latency(row["layer_expert_full"], row["layer_type"])
    results.append(res)

# 평균 레이턴시 비교 결과 저장
df_result = pd.DataFrame(results)
df_result.to_csv("results/gpu_comparison_avg.csv", index=False)

# 그래프 시각화 1
sns.countplot(data=df_result, x="better_gpu", order=["1080Ti", "A6000"])
plt.title("GPU Preference by Average Latency (Token 1–100)")
plt.xlabel("Better GPU")
plt.ylabel("Number of Experts")
plt.grid(True)
plt.savefig("results/gpu_comparison_avg.png")
plt.close()

# 그래프 시각화 2
sns.set(style="whitegrid")
plt.figure(figsize=(18, 6))  # 넓은 가로 그래프
df_result["latency_diff"] = df_result["1080Ti"] - df_result["A6000"]
for i, layer_type in enumerate(["encoder", "decoder"]):
    plt.subplot(1, 2, i + 1)

    df_sub = df_result[df_result["layer_type"] == layer_type].copy()
    df_sub = df_sub.sort_values(by="latency_diff", ascending=False)

    colors = df_sub["better_gpu"].map({
        "1080Ti": "red",
        "A6000": "blue",
        "N/A": "gray"
    })

    # latency 차이를 bar height로 표현
    bar_heights = df_sub["latency_diff"].fillna(0).abs()

    plt.bar(df_sub["expert"], bar_heights, color=colors)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"{layer_type.capitalize()} Experts - Latency Difference")
    plt.ylabel("|Latency 1080Ti - A6000| (ms)")
    plt.xlabel("Expert")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    if i == 0:
        plt.legend(
            handles=[
                plt.Line2D([0], [0], color='red', lw=8, label='1080Ti better'),
                plt.Line2D([0], [0], color='blue', lw=8, label='A6000 better'),
                plt.Line2D([0], [0], color='gray', lw=8, label='Model missing')
            ],
            loc='upper right'
        )

plt.suptitle("GPU Preference by Latency Difference per Expert", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("results/gpu_latency_diff_barplot.png")
plt.close()