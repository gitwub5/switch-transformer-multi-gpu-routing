import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# 모델 정의
# ---------------------------
class LatencyAwareTop1Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)  # 각 expert별 latency 예측
        )

    def forward(self, x):
        # hidden_states: [batch_size, tokens_per_group, hidden_dim]
        pooled = hidden_states.mean(dim=1)  # 토큰 평균 → feature로 사용

        # 우리가 학습한 latency 예측 모델 (PyTorch MLP or 외부 latency regressor)
        predicted_latency = self.latency_model(pooled)  # [batch_size, num_experts]

        # 라우팅 확률 계산: latency 낮을수록 높게 → softmax(-latency)
        router_logits = -predicted_latency
        router_probs = F.softmax(router_logits, dim=-1)

        # 기존과 동일한 방식으로 expert 선택
        top_expert = torch.argmax(router_probs, dim=-1)
        return self.router(x)  # [batch_size, num_experts]

def preprocess_router_dataset(df):
    df["gpu_id_encoded"] = df["gpu_id"].map({"1080Ti": 0, "A6000": 1})
    features = ["token_id", "layer_index", "router_entropy", "layer_token_count", "gpu_id_encoded"]

    # 토큰별로 latency가 최소인 expert를 label로 지정
    def get_best_expert(group):
        min_row = group.loc[group["latency_ms"].idxmin()]
        return pd.Series({
            **min_row[features].to_dict(),
            "label": min_row["expert_id"]
        })

    grouped = df.groupby(["text_id", "token_id", "layer_index"]).apply(get_best_expert).reset_index(drop=True)

    X = grouped[features].values
    y = grouped["label"].values
    return X, y


def train_all_latency_routers(df, num_experts=8, device='cuda'):
    summary = []
    models = {}

    df["gpu_id_encoded"] = df["gpu_id"].map({"1080Ti": 0, "A6000": 1})
    features = ["token_id", "router_entropy", "layer_token_count", "gpu_id_encoded"]

    grouped = df.groupby(["layer_type", "layer_index"])
    for (layer_type, layer_index), layer_df in grouped:
        print(f"📦 Training Router for {layer_type} layer {layer_index}...")

        def get_best_expert(group):
            min_row = group.loc[group["latency_ms"].idxmin()]
            return pd.Series({**min_row[features].to_dict(), "label": min_row["expert_id"]})

        train_df = layer_df.groupby(["text_id", "token_id"]).apply(get_best_expert).reset_index(drop=True)

        if len(train_df) < 50:
            print(f"⚠️ 데이터 부족: {layer_type} {layer_index}, 건너뜀")
            continue

        X = train_df[features].values
        y = train_df["label"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = LatencyAwareTop1Router(input_dim=X.shape[1], num_experts=num_experts).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(30):
            model.train()
            total_loss = 0
            for xb, yb in dataloader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        models[(layer_type, layer_index)] = {
            "model": model,
            "scaler": scaler
        }

        # 성능 측정
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).argmax(dim=1).cpu().numpy()
            acc = np.mean(preds == y)
            summary.append({
                "layer_type": layer_type,
                "layer_index": layer_index,
                "accuracy": round(acc, 4),
                "samples": len(train_df)
            })

    return models, pd.DataFrame(summary)

if __name__ == "__main__":
    df = pd.read_csv("../outputs/router_dataset.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, summary = train_all_latency_routers(df, num_experts=8, device=device)

    #모델 저장
    for (layer_type, layer_index), model_data in models.items():
        model = model_data["model"]
        scaler = model_data["scaler"]
        model_path = f"torch_models/{layer_type}_{layer_index}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장: {model_path}")

    summary.to_csv("torch_models/results/router_training_summary.csv", index=False)