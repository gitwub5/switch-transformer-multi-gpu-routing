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

# 시각화 스타일 설정
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 1. GPU 선호도 분포 시각화
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df_result, x="better_gpu", order=["1080Ti", "A6000"],
                  palette={"1080Ti": "#1f77b4", "A6000": "#ff7f0e"})
plt.title("GPU Preference Distribution by Average Latency", pad=20)
plt.xlabel("Preferred GPU")
plt.ylabel("Number of Experts")

plt.tight_layout()
plt.savefig("results/gpu_comparison_avg.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. 레이어별 레이턴시 차이 시각화
plt.figure(figsize=(15, 6))
df_result["latency_diff"] = df_result["1080Ti"] - df_result["A6000"]

for i, layer_type in enumerate(["encoder", "decoder"]):
    plt.subplot(1, 2, i + 1)
    
    df_sub = df_result[df_result["layer_type"] == layer_type].copy()
    df_sub = df_sub.sort_values(by="latency_diff", ascending=False)
    
    # 색상 매핑
    colors = df_sub["better_gpu"].map({
        "1080Ti": "#1f77b4",  # 파란색
        "A6000": "#ff7f0e",   # 주황색
        "N/A": "#7f7f7f"      # 회색
    })
    
    # 레이턴시 차이를 막대 높이로 표현 (절대값 사용)
    bar_heights = df_sub["latency_diff"].fillna(0).abs()
    
    # 막대 그래프 그리기
    bars = plt.bar(df_sub["expert"], bar_heights, color=colors, alpha=0.8)
    
    # x축 레이블 회전 및 크기 조정
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    # 제목 및 레이블
    plt.title(f"{layer_type.capitalize()} Layer Experts", pad=20)
    plt.ylabel("|Latency Difference| (ms)")
    plt.xlabel("Expert ID")
    
    # y축을 로그 스케일로 설정
    plt.yscale('log')
    
    # 그리드 설정
    plt.grid(axis="y", linestyle="--", alpha=0.3)

# 범례 추가
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor="#1f77b4", alpha=0.8, label='1080Ti better'),
    plt.Rectangle((0,0),1,1, facecolor="#ff7f0e", alpha=0.8, label='A6000 better'),
    plt.Rectangle((0,0),1,1, facecolor="#7f7f7f", alpha=0.8, label='Model missing')
]
plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=3, frameon=True, fontsize=10)

plt.suptitle("GPU Performance Comparison by Expert", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig("results/gpu_latency_diff_barplot.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. 레이턴시 분포 히트맵 시각화
plt.figure(figsize=(12, 8))

# 데이터 전처리
df_pivot = df_result.pivot_table(
    values='latency_diff',
    index='layer_type',
    columns='expert',
    aggfunc='mean'
)

# NaN 값을 0으로 대체
df_pivot = df_pivot.fillna(0)

# 히트맵 그리기
sns.heatmap(df_pivot, cmap='RdBu_r', center=0,
            annot=True, fmt='.1f', cbar_kws={'label': 'Latency Difference (ms)'},
            square=True, vmin=-5, vmax=5)  # 값의 범위를 -5ms에서 5ms로 제한

plt.title("Latency Difference Distribution Across Experts and Layers", pad=20)
plt.xlabel("Expert ID")
plt.ylabel("Layer Type")
plt.tight_layout()
plt.savefig("results/gpu_latency_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()