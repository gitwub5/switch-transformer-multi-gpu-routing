import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import torch

# =========================
# 학습 결과 저장
# =========================
def save_model_summary(summary_records, output_base_dir):
    """
    학습된 모든 모델의 성능 요약을 CSV로 저장하는 함수
    """
    summary_df = pd.DataFrame(summary_records)
    summary_dir = os.path.join(output_base_dir, "results")
    os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"📄 전체 학습 요약 저장 완료: {summary_path}")

# =========================
# 데이터 로딩 및 전처리
# =========================
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df["gpu_id_encoded"] = df["gpu_id"].map({"1080Ti": 0, "A6000": 1})
    
    # 이상치 제거를 위한 IQR 방법 적용
    def remove_outliers(group):
        Q1 = group['latency_ms'].quantile(0.25)
        Q3 = group['latency_ms'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group['latency_ms'] >= lower_bound) & (group['latency_ms'] <= upper_bound)]
    
    # layer_type, layer_index, expert_id, gpu_id_encoded 기준으로 그룹화하여 이상치 제거
    df = df.groupby(['layer_type', 'layer_index', 'expert_id', 'gpu_id_encoded']).apply(remove_outliers).reset_index(drop=True)
    
    print(f"📊 이상치 제거 후 데이터 크기: {len(df)} rows")
    return df


# ============================
# layer_type, layer_index 기준 학습
# ============================
def train_and_save_models(df, output_base_dir):
    summary_records = []
    
    for layer_type in df["layer_type"].unique():
        type_df = df[df["layer_type"] == layer_type]
        type_str = "encoder" if layer_type == "encoder" else "decoder"

        for layer_index in sorted(type_df["layer_index"].unique()):
            print(f"🔍 {type_str} layer {layer_index} 학습 시작")
            layer_df = type_df[type_df["layer_index"] == layer_index]
            layer_dir = os.path.join(output_base_dir, type_str)
            os.makedirs(layer_dir, exist_ok=True)

            # 입력 특성 (token_id 제외)
            features = [
                "router_entropy", "router_score",
                "layer_token_count", "gpu_id_encoded", "expert_id"
            ]
            
            X = layer_df[features]
            y = layer_df["latency_ms"]
            
            model = XGBRegressor(device="cuda", n_estimators=100, learning_rate=0.01, max_depth=3)
            model.fit(X, y)

            # 성능 평가
            preds = model.predict(X)
            mae = np.mean(np.abs(preds - y))
            rmse = np.sqrt(np.mean((preds - y) ** 2))
            r2 = model.score(X, y)

            summary_records.append({
                "layer_type": type_str,
                "layer_index": layer_index,
                "sample_count": len(layer_df),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            })

            # layer별로 하나의 모델 저장
            fname = f"layer_{layer_index}.joblib"
            joblib.dump(model, os.path.join(layer_dir, fname))
            print(f"✅ {type_str} layer {layer_index} 학습 완료")
            
    save_model_summary(summary_records, output_base_dir)

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
# 추론: 최적 GPU 선택
# ============================
def latency_aware_gpu_selector(token_feature: dict, expert_id: int, models: dict):
    """
    주어진 expert_id에 대해 latency가 가장 낮은 GPU를 선택하는 함수
    
    Args:
        token_feature (dict): 토큰의 특성 정보
            - router_entropy: 라우터 엔트로피
            - router_score: 라우터 점수
            - layer_token_count: layer의 토큰 수
            - layer_type_encoded: layer 타입 (0: encoder, 1: decoder)
            - gpu_id_encoded: 현재 토큰이 위치한 GPU (0: 1080Ti, 1: A6000)
        expert_id (int): 기존 라우터가 선택한 expert ID
        models (dict): 학습된 모델들
    
    Returns:
        dict: 선택된 GPU 정보와 예상 latency
    """
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

    # 각 GPU에 대해 예측
    best_latency = float("inf")
    best_gpu = None

    for target_gpu_id in [0, 1]:  # 2개의 GPU
        # 특성 벡터 생성 (token_id 제외)
        feature_vec = [
            token_feature["router_entropy"],
            token_feature["router_score"],
            token_feature["layer_token_count"],
            expert_id,  # expert_id
            target_gpu_id  # target GPU
        ]
        
        model_key = (next_type, next_layer_index)
        if model_key in models:
            pred_latency = models[model_key].predict([feature_vec])[0]
            pred_latency = max(0.0, pred_latency)
            
            if pred_latency < best_latency:
                best_latency = pred_latency
                best_gpu = target_gpu_id

    if best_gpu is None:
        return None
    
    return {
        "layer_index": next_layer_index,
        "expert_id": expert_id,
        "gpu_id": best_gpu,
        "latency": float(best_latency)
    }

# ============================
# 실행 예시
# ============================
if __name__ == "__main__":
    CSV_PATH = "outputs/router_dataset.csv"
    MODEL_DIR = "models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 사용 가능한 장치: {device}")

    df = load_and_preprocess_data(CSV_PATH)
    print(f"📊 데이터 로딩 완료: {len(df)} rows")
    train_and_save_models(df, MODEL_DIR)

    models = load_all_models(MODEL_DIR)
    print(models.keys())

    example_input = {
        "router_entropy": 0.32,
        "router_score": 0.5,
        "layer_token_count": 128,
        "layer_type_encoded": 0,  # encoder
        "gpu_id_encoded": 0,  # 현재 1080Ti에 위치
        "layer_index": 3
    }

    # 예시: expert_id가 3으로 결정된 경우
    selected_expert_id = 3
    result = latency_aware_gpu_selector(example_input, selected_expert_id, models)
    
    if result:
        print(f"🔀 GPU 선택 결과:")
        print(f"  - Layer: {result['layer_index']}")
        print(f"  - Expert: {result['expert_id']}")
        print(f"  - GPU: {result['gpu_id']} ({'1080Ti' if result['gpu_id'] == 0 else 'A6000'})")
        print(f"  - 예상 지연시간: {result['latency']:.6f} ms")
    else:
        print("❌ GPU 선택 결과가 없습니다.")
