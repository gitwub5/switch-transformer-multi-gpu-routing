import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
    
    # 이상치 제거 (선택적으로 latency_ms 기준)
    def remove_outliers(group):
        Q1 = group['latency_ms'].quantile(0.25)
        Q3 = group['latency_ms'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return group[(group['latency_ms'] >= lower) & (group['latency_ms'] <= upper)]
    
    df = df.groupby(['layer_type', 'layer_index', 'expert_id', 'gpu_id_encoded']).apply(remove_outliers).reset_index(drop=True)

    # standard scaling
    scaler = StandardScaler()
    df[["router_score", "router_entropy", "layer_token_count"]] = scaler.fit_transform(
        df[["router_score", "router_entropy", "layer_token_count"]]
    )

    print(f"이상치 제거 후 데이터 수: {len(df)} rows")
    return df


# ============================
# GPU 분류기 학습
# ============================
def train_and_save_gpu_classifiers(df, output_base_dir):
    summary_records = []
    os.makedirs(output_base_dir, exist_ok=True)

    base_features = [
        "router_score",
        "router_entropy",
        "layer_token_count"
    ]

    grouped = df.groupby(["layer_type", "layer_index"])
    for (layer_type, layer_index), sub_df in grouped:
        if sub_df["gpu_id_encoded"].nunique() < 2:
            print(f"⚠️ layer {layer_index}에 대해 GPU 클래스 부족 → 스킵")
            continue

        # expert_id → one-hot
        one_hot = pd.get_dummies(sub_df["expert_id"], prefix="expert")
        X = pd.concat([sub_df[base_features].reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)
        y = sub_df["gpu_id_encoded"].reset_index(drop=True)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            eval_metric='logloss'
        )
        model.fit(X, y)

        preds = model.predict(X)
        acc = (preds == y).mean()

        # 저장
        layer_type_str = "encoder" if layer_type == "encoder" else "decoder"
        model_dir = os.path.join(output_base_dir, layer_type_str, f"layer_{layer_index}")
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(model_dir, f"gpu_classifier.joblib"))
        joblib.dump(X.columns.tolist(), os.path.join(model_dir, f"gpu_classifier_features.joblib"))

        summary_records.append({
            "layer_type": layer_type_str,
            "layer_index": layer_index,
            "sample_count": len(sub_df),
            "accuracy": round(acc, 4)
        })
        print(f"✅ Trained layer {layer_index} ({layer_type_str}) with acc {acc:.4f}")

    pd.DataFrame(summary_records).to_csv(os.path.join(output_base_dir, "gpu_classifier_summary.csv"), index=False)
    
# ============================
# 모델 불러오기
# ============================
def load_gpu_classifiers(model_base_dir):
    models = {}
    for layer_type in ["encoder", "decoder"]:
        layer_type_dir = os.path.join(model_base_dir, layer_type)
        if not os.path.exists(layer_type_dir):
            continue

        for layer_folder in os.listdir(layer_type_dir):
            layer_path = os.path.join(layer_type_dir, layer_folder)
            if not os.path.isdir(layer_path):
                continue

            layer_index = int(layer_folder.replace("layer_", ""))
            for fname in os.listdir(layer_path):
                if fname.endswith("_gpu_classifier.joblib"):
                    expert_id = int(fname.split("_")[1])
                    model_path = os.path.join(layer_path, fname)
                    feature_path = model_path.replace("_gpu_classifier.joblib", "_features.joblib")

                    model = joblib.load(model_path)
                    features = joblib.load(feature_path)
                    models[(layer_type, layer_index, expert_id)] = (model, features)

    return models
    
# ============================
# 실행 예시
# ============================
if __name__ == "__main__":
    CSV_PATH = "outputs/router_dataset.csv"
    MODEL_SAVE_DIR = "router_classifiers"

    df = load_and_preprocess_data(CSV_PATH)
    train_and_save_gpu_classifiers(df, MODEL_SAVE_DIR)