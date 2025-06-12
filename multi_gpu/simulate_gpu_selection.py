import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model_and_predict(expert, layer_type, token_count, gpu):
    model_path = os.path.join(layer_type, gpu, f"{expert}_model.joblib")
    scaler_path = os.path.join(layer_type, gpu, f"{expert}_scaler.joblib")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"❌ {expert} model for {gpu} not found. Skipping...")
        return np.nan
    
    token_log = np.log1p(token_count).reshape(-1, 1)
    pred_scaled = model.predict(token_log)
    pred_log10 = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    latency = np.power(10, pred_log10)
    return latency[0]

def simulate_execution(df, token_counts):
    results = []
    
    # 각 expert별로 시뮬레이션
    for _, row in df.iterrows():
        expert = row["layer_expert_full"]
        layer_type = row["layer_type"]
        
        # 각 GPU에서의 예측 레이턴시 계산
        latencies = {}
        for gpu in ["1080Ti", "A6000"]:
            latency = load_model_and_predict(expert, layer_type, token_counts, gpu)
            latencies[gpu] = latency
        
        # 최적의 GPU 선택
        best_gpu = min(latencies.items(), key=lambda x: x[1])[0] if not any(np.isnan(v) for v in latencies.values()) else "N/A"
        
        results.append({
            "expert": expert,
            "layer_type": layer_type,
            "1080Ti_latency": latencies["1080Ti"],
            "A6000_latency": latencies["A6000"],
            "best_gpu": best_gpu,
            "best_latency": min(latencies.values()) if not any(np.isnan(v) for v in latencies.values()) else np.nan
        })
    
    return pd.DataFrame(results)

def visualize_results(df_results, token_counts):
    # 1. 전체 실행 시간 비교 (비율)
    total_time_1080ti = df_results["1080Ti_latency"].sum()
    total_time_a6000 = df_results["A6000_latency"].sum()
    total_time_with_selection = df_results["best_latency"].sum()

    ratio_1080ti = 100
    ratio_a6000 = (total_time_a6000 / total_time_1080ti) * 100 if total_time_1080ti else 0
    ratio_optimal = (total_time_with_selection / total_time_1080ti) * 100 if total_time_1080ti else 0

    plt.figure(figsize=(10, 6))
    plt.bar(["1080Ti Only", "A6000 Only", "Optimal Selection"],
            [ratio_1080ti, ratio_a6000, ratio_optimal],
            color=["red", "blue", "green"])
    plt.title("Total Execution Time Comparison (Relative to 1080Ti, %)")
    plt.ylabel("Relative Execution Time (%)")
    plt.ylim(0, max(110, ratio_1080ti, ratio_a6000, ratio_optimal))
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/total_execution_time_comparison_ratio.png")
    plt.close()
    
    # 2. GPU 선택 분포
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_results, x="best_gpu", order=["1080Ti", "A6000", "N/A"])
    plt.title("GPU Selection Distribution")
    plt.xlabel("Selected GPU")
    plt.ylabel("Number of Experts")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/gpu_selection_distribution.png")
    plt.close()
    
    # 3. 레이어 타입별 GPU 선택 분포
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_results, x="layer_type", hue="best_gpu", 
                 order=["encoder", "decoder"], hue_order=["1080Ti", "A6000", "N/A"])
    plt.title("GPU Selection by Layer Type")
    plt.xlabel("Layer Type")
    plt.ylabel("Number of Experts")
    plt.legend(title="Selected GPU")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/gpu_selection_by_layer.png")
    plt.close()
    
    # 4. 레이턴시 개선률 (1080Ti 대비)
    improvement = (df_results["1080Ti_latency"] - df_results["best_latency"]) / df_results["1080Ti_latency"] * 100
    plt.figure(figsize=(10, 6))
    sns.histplot(improvement.dropna(), bins=20)
    plt.title("Latency Improvement Distribution (vs 1080Ti)")
    plt.xlabel("Improvement Percentage (%)")
    plt.ylabel("Number of Experts")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/latency_improvement_distribution.png")
    plt.close()
    
    # 5. 레이턴시 비교 산점도 (전체 범위)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["1080Ti_latency"], df_results["A6000_latency"], alpha=0.5)
    plt.plot([0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             [0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             'r--', alpha=0.3)
    plt.title("Latency Comparison: 1080Ti vs A6000 (Full Range)")
    plt.xlabel("1080Ti Latency (ms)")
    plt.ylabel("A6000 Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/latency_comparison_scatter_full.png")
    plt.close()
    
    # 6. 레이턴시 비교 산점도 (확대 범위)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["1080Ti_latency"], df_results["A6000_latency"], alpha=0.5)
    plt.plot([0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             [0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             'r--', alpha=0.3)
    plt.title("Latency Comparison: 1080Ti vs A6000 (Zoomed)")
    plt.xlabel("1080Ti Latency (ms)")
    plt.ylabel("A6000 Latency (ms)")
    
    # 데이터의 95% 범위로 제한
    x_95 = np.percentile(df_results["1080Ti_latency"], 95)
    y_95 = np.percentile(df_results["A6000_latency"], 95)
    plt.xlim(0, x_95)
    plt.ylim(0, y_95)
    
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/latency_comparison_scatter_zoomed.png")
    plt.close()
    
    # 7. 레이턴시 비교 산점도 (히트맵)
    plt.figure(figsize=(10, 6))
    plt.hist2d(df_results["1080Ti_latency"], df_results["A6000_latency"], 
               bins=50, cmap='viridis')
    plt.colorbar(label='Count')
    plt.plot([0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             [0, max(df_results["1080Ti_latency"].max(), df_results["A6000_latency"].max())], 
             'r--', alpha=0.3)
    plt.title("Latency Comparison: 1080Ti vs A6000 (Heatmap)")
    plt.xlabel("1080Ti Latency (ms)")
    plt.ylabel("A6000 Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/latency_comparison_heatmap.png")
    plt.close()
    
    # 8. 레이턴시 비교 산점도 (0.015~0.020 확대)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_results["1080Ti_latency"], df_results["A6000_latency"], alpha=0.5)
    plt.plot([0.015, 0.020], [0.015, 0.020], 'r--', alpha=0.3)
    plt.title("Latency Comparison: 1080Ti vs A6000 (0.015~0.020 Zoomed)")
    plt.xlabel("1080Ti Latency (ms)")
    plt.ylabel("A6000 Latency (ms)")
    plt.xlim(0.015, 0.020)
    plt.ylim(0.015, 0.020)
    plt.grid(True, alpha=0.3)
    plt.savefig("results/simulation/latency_comparison_scatter_0015_0020.png")
    plt.close()

def main():
    # 데이터 로드
    df = pd.read_csv("../outputs/merged_latency_comparison.csv")
    df["layer_expert_full"] = df["layer_type"] + "_" + df["layer_index"].astype(str) + "_" + df["expert_id"].astype(str)
    unique_experts = df[["layer_type", "layer_expert_full"]].drop_duplicates()
    
    # 토큰 수 범위 설정 (1부터 200까지 5개씩)
    token_counts = np.arange(1, 201, 5)
    
    # 시뮬레이션 실행
    results = simulate_execution(unique_experts, token_counts)
    
    # 결과 저장
    results.to_csv("results/simulation/gpu_selection_simulation.csv", index=False)
    
    # 시각화
    visualize_results(results, token_counts)
    
    # 결과 요약 출력
    print("\n=== 시뮬레이션 결과 요약 ===")
    print(f"전체 expert 수: {len(results)}")
    print(f"1080Ti 선택된 expert 수: {len(results[results['best_gpu'] == '1080Ti'])}")
    print(f"A6000 선택된 expert 수: {len(results[results['best_gpu'] == 'A6000'])}")
    print(f"모델 누락된 expert 수: {len(results[results['best_gpu'] == 'N/A'])}")
    print(f"\n전체 실행 시간 (1080Ti): {results['1080Ti_latency'].sum():.2f} ms")
    print(f"최적 GPU 선택 시 실행 시간: {results['best_latency'].sum():.2f} ms")
    print(f"개선률: {((results['1080Ti_latency'].sum() - results['best_latency'].sum()) / results['1080Ti_latency'].sum() * 100):.2f}%")

if __name__ == "__main__":
    main() 