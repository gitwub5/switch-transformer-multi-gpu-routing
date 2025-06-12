import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# 💡 MLP 모델 정의
class LatencyMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # latency 예측
        )

    def forward(self, x):
        return self.model(x)

# ✅ 통합 MLP 학습 함수
def train_unified_mlp_from_xgb(model_base_dir, df_csv, output_model_path, use_actual_latency=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(df_csv)
    df["gpu_id_encoded"] = df["gpu_id"].map({"1080Ti": 0, "A6000": 1})
    df["layer_type_encoded"] = df["layer_type"].map({"encoder": 0, "decoder": 1})

    # 전체 입력 특성 (token_id 제외)
    features = [
        "router_entropy", "router_score",
        "layer_token_count", "gpu_id_encoded", "expert_id",
        "layer_index", "layer_type_encoded"
    ]

    all_X, all_y = [], []

    for layer_type in ["encoder", "decoder"]:
        type_path = os.path.join(model_base_dir, layer_type)
        if not os.path.exists(type_path):
            continue

        for fname in os.listdir(type_path):
            if fname.endswith(".joblib"):
                layer_index = int(fname.replace("layer_", "").replace(".joblib", ""))
                model_path = os.path.join(type_path, fname)
                xgb_model = joblib.load(model_path)

                sub_df = df[
                    (df["layer_type"] == layer_type) &
                    (df["layer_index"] == layer_index)
                ]

                if sub_df.empty:
                    continue

                X = sub_df[features].values.astype(np.float32)

                if use_actual_latency:
                    y = sub_df["latency_ms"].values.astype(np.float32)
                    print(f"📊 실제 latency 사용: {layer_type} layer {layer_index}")
                else:
                    y = xgb_model.predict(X).astype(np.float32)
                    print(f"📊 XGBoost 예측 latency 사용: {layer_type} layer {layer_index}")

                all_X.append(X)
                all_y.append(y)

    # 전체 통합
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0).reshape(-1, 1)

    # Torch Dataset
    X_tensor = torch.tensor(X_all)
    y_tensor = torch.tensor(y_all)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = LatencyMLP(input_dim=X_all.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 학습
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"✅ MLP 모델 저장 완료: {output_model_path}")

# 🔧 실행 예시
if __name__ == "__main__":
    MODEL_DIR = "dynamic_routing/models"
    CSV_PATH = "outputs/router_dataset.csv"
    OUTPUT_PT_PATH = "dynamic_routing/pt_models/latency_predictor.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 사용 가능한 장치: {device}")

    train_unified_mlp_from_xgb(
        model_base_dir=MODEL_DIR,
        df_csv=CSV_PATH,
        output_model_path=OUTPUT_PT_PATH,
        use_actual_latency=False,  # ❗XGB 예측 latency를 타겟으로 사용
        device=device
    )