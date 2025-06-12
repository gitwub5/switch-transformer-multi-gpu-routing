import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from collections import defaultdict, deque

# ============================
# 모델 불러오기
# ============================
def load_all_models(model_base_dir):
    models = {}

    for layer_type in ["encoder", "decoder"]:
        type_path = os.path.join(model_base_dir, layer_type)
        if not os.path.exists(type_path):
            continue

        for fname in os.listdir(type_path):
            if fname.endswith(".joblib"):
                # 파일 이름 형식: layer_{layer_index}.joblib
                layer_index = int(fname.replace("layer_", "").replace(".joblib", ""))
                model = joblib.load(os.path.join(type_path, fname))
                models[(layer_type, layer_index)] = model

    return models

# ============================
# 예측 기반 Top1 라우팅
# ============================
def latency_aware_top1_router(token_feature: dict, models: dict):
    # 현재 layer -> 다음 routing 가능한 layer 정의
    NEXT_ROUTING_LAYER = {
        1: 1,  # layer 1의 expert 선택
        3: 3,  # layer 3의 expert 선택
        5: 5,  # layer 5의 expert 선택
        7: 7,  # layer 7의 expert 선택
        9: 9,  # layer 9의 expert 선택
        11: 11,  # layer 11의 expert 선택
    }

    # 현재 layer 기준 다음 layer 설정
    current_layer = token_feature["layer_index"]
    current_type = "encoder" if token_feature["layer_type_encoded"] == 0 else "decoder"
    
    # encoder layer 11에서 decoder layer 1로 넘어가는 경우
    if current_type == "encoder" and current_layer == 11:
        next_layer_index = 1
        next_type = "decoder"
    else:
        next_layer_index = NEXT_ROUTING_LAYER.get(current_layer)
        next_type = current_type

    # 다음 라우팅 layer가 없다면 None 반환
    if next_layer_index is None:
        print(f"⛔️ layer {current_layer}에서는 더 이상 routing이 없습니다.")
        return None

    best_latency = float("inf")
    best_expert = None

    # 현재 layer의 모델 가져오기
    model_key = (next_type, next_layer_index)
    if model_key not in models:
        return None

    model = models[model_key]

    # 각 expert와 GPU 조합에 대해 예측
    for expert_id in range(8):  # 8개의 expert
        for gpu_id in [0, 1]:  # 2개의 GPU
            # 특성 벡터 생성 (token_id 제외)
            feature_vec = [
                token_feature["router_entropy"],
                token_feature["router_score"],
                token_feature["layer_token_count"],
                expert_id,  # expert_id
                gpu_id  # target GPU
            ]
            
            pred_latency = model.predict([feature_vec])[0]
            pred_latency = max(0.0, pred_latency)
            
            if pred_latency < best_latency:
                best_latency = pred_latency
                best_expert = (next_layer_index, expert_id, gpu_id)

    return best_expert, best_latency

# ============================
# 시뮬레이션
# ============================
def simulate_routing(df, models, n_samples=300):
    results = []
    rr_queues = defaultdict(lambda: deque())  # RoundRobin 큐: (layer_type, layer_index, gpu_id) -> expert_id 순서

    for key, group in df.groupby(["layer_type", "layer_index", "gpu_id_encoded"]):
        expert_ids = group["expert_id"].unique().tolist()
        rr_queues[key] = deque(sorted(expert_ids))  # 고정된 순서로 초기화

    for _, row in df.sample(n=n_samples, random_state=42).iterrows():
        token_feature = {
            "layer_index": row["layer_index"],  # layer_index는 라우팅 로직에만 사용
            "router_entropy": row["router_entropy"],
            "router_score": row["router_score"],
            "layer_token_count": row["layer_token_count"],
            "layer_type_encoded": 0 if row["layer_type"] == "encoder" else 1,
            "gpu_id_encoded": row["gpu_id_encoded"],
        }

        # Latency-Aware 라우팅
        latency_pred_result = latency_aware_top1_router(token_feature, models)
        if latency_pred_result is not None:
            _, latency_pred = latency_pred_result
        else:
            latency_pred = None

        # Random 라우팅
        rand_candidates = df[
            (df["layer_type"] == row["layer_type"]) &
            (df["layer_index"] == row["layer_index"]) &
            (df["gpu_id_encoded"] == row["gpu_id_encoded"])
        ]
        latency_random = rand_candidates.sample(1)["latency_ms"].values[0] if not rand_candidates.empty else None

        # Round-Robin 라우팅
        rr_key = (row["layer_type"], row["layer_index"], row["gpu_id_encoded"])
        rr_queue = rr_queues[rr_key]
        if rr_queue:
            rr_expert = rr_queue[0]
            rr_queues[rr_key].rotate(-1)
            rr_latency_row = rand_candidates[rand_candidates["expert_id"] == rr_expert]
            latency_rr = rr_latency_row.sample(1)["latency_ms"].values[0] if not rr_latency_row.empty else None
        else:
            latency_rr = None

        # Uniform Expert 선택 (샘플 개수 고려 없이 무작위 expert 선택)
        unique_experts = rand_candidates["expert_id"].unique()
        if len(unique_experts) > 0:
            u_expert = np.random.choice(unique_experts)
            u_latency_row = rand_candidates[rand_candidates["expert_id"] == u_expert]
            latency_uniform = u_latency_row.sample(1)["latency_ms"].values[0] if not u_latency_row.empty else None
        else:
            latency_uniform = None

        results.append({
            "layer_index": row["layer_index"],
            "layer_type": row["layer_type"],
            "gpu_id": row["gpu_id"],
            "LatencyAware": latency_pred,
            "Random": latency_random,
            "RoundRobin": latency_rr,
            "Uniform": latency_uniform,
        })

    return pd.DataFrame(results)

# ============================
# 시각화
# ============================
def visualize_comparison(result_df):
    # 그래프 스타일 설정
    sns.set(style="whitegrid", font_scale=1.2)
    # 데이터 준비
    melted_df = result_df.melt(
        id_vars=["layer_index"], 
        value_vars=["LatencyAware", "Random", "RoundRobin", "Uniform"],
        var_name="Router", 
        value_name="Latency"
    )
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    # 박스플롯 생성 (이상치 표시 제거)
    ax = sns.boxplot(
        data=melted_df, 
        x="Router", 
        y="Latency",
        showfliers=False,  # 이상치 표시 제거
    )
    # y축 범위 설정
    plt.ylim(0.010, 0.025)
    
    # 제목과 레이블 설정
    plt.title("Latency Comparison of Different Routing Algorithms", 
              fontsize=14, 
              pad=15)
    plt.xlabel("Routing Strategy", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    
    # 그리드 설정
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # x축 레이블 회전
    plt.xticks(rotation=0)
    
    # 테두리 제거
    sns.despine()
    
    # 여백 조정
    plt.tight_layout()
    
    # 저장 및 표시
    plt.savefig("results/routing_simulation_result.png", 
                dpi=300,  # 고해상도
                bbox_inches='tight',
                pad_inches=0.2)
    plt.show()

def visualize_more(result_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 히스토그램
    plt.figure(figsize=(10,6))
    for router in ["LatencyAware", "Random", "RoundRobin", "Uniform"]:
        sns.histplot(result_df[router], label=router, kde=True, stat="density", element="step", fill=False)
    plt.legend()
    plt.title("Latency Distribution by Routing Algorithm")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Density")
    plt.savefig("results/latency_histogram.png")
    plt.close()

    # 2. 누적분포(CDF)
    plt.figure(figsize=(10,6))
    for router in ["LatencyAware", "Random", "RoundRobin", "Uniform"]:
        sns.ecdfplot(result_df[router], label=router)
    plt.legend()
    plt.title("Latency CDF by Routing Algorithm")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.savefig("results/latency_cdf.png")
    plt.close()

    # 4. 개선률 바플롯
    improvement = {
        "Random": 100 * (result_df["Random"].mean() - result_df["LatencyAware"].mean()) / result_df["Random"].mean(),
        "RoundRobin": 100 * (result_df["RoundRobin"].mean() - result_df["LatencyAware"].mean()) / result_df["RoundRobin"].mean(),
        "Uniform": 100 * (result_df["Uniform"].mean() - result_df["LatencyAware"].mean()) / result_df["Uniform"].mean(),
    }
    plt.figure(figsize=(8,6))
    sns.barplot(x=list(improvement.keys()), y=list(improvement.values()))
    plt.title("Latency Improvement of LatencyAware (%)")
    plt.ylabel("Improvement (%)")
    plt.xlabel("Compared to")
    plt.savefig("results/latency_improvement_bar.png")
    plt.close()

# ============================
# 실행
# ============================
if __name__ == "__main__":
    CSV_PATH = "../outputs/router_dataset.csv"
    MODEL_DIR = "models"

    df = pd.read_csv(CSV_PATH)
    df["gpu_id_encoded"] = df["gpu_id"].map({"1080Ti": 0, "A6000": 1})

    models = load_all_models(MODEL_DIR)

    result_df = simulate_routing(df, models, n_samples=300)
    result_df.to_csv("results/routing_simulation_result.csv", index=False)
    print(result_df.describe())

    result_df = pd.read_csv("results/routing_simulation_result.csv")
    visualize_comparison(result_df)
    visualize_more(result_df)