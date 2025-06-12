import pandas as pd
import ast
import os

def preprocess_router_tokens(file_path: str, gpu_name: str):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["routed_token_ids", "routed_token_strs", "router_entropies", "latency_ms"])

    rows = []
    for _, row in df.iterrows():
        try:
            token_ids = ast.literal_eval(row["routed_token_ids"])
            token_strs = ast.literal_eval(row["routed_token_strs"])
            token_scores = ast.literal_eval(row["router_scores"])
            router_entropies = ast.literal_eval(row["router_entropies"]) if "router_entropies" in row else [None] * len(token_ids)
        except Exception as e:
            print(f"⚠️ Skipping row due to parsing error: {e}")
            continue

        for token_id, token_str, score, entropy in zip(token_ids, token_strs, token_scores, router_entropies):
            rows.append({
                "text_id": row["text_id"],
                "layer_type": row["layer_type"],
                "layer_index": row["layer_index"],
                "expert_id": row["expert_id"],
                "token_id": token_id,
                "token_str": token_str,
                "router_entropy": entropy,
                "router_score": score,
                "latency_ms": row["latency_ms"],  # ✅ latency 추가
                "gpu_id": gpu_name,
            })

    return pd.DataFrame(rows)

# 현재 파일의 절대 경로를 기준으로 outputs 디렉토리 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
data_dir = os.path.join(os.path.dirname(current_dir), "outputs")  # 상위 디렉토리의 outputs 폴더

# 파일 경로 설정
files = {
    "A6000": os.path.join(data_dir, "merged_profile_NVIDIA RTX A6000.csv"),
    "1080Ti": os.path.join(data_dir, "merged_profile_NVIDIA GeForce GTX 1080 Ti.csv")
}

# ✅ 각 GPU별 데이터 전처리
processed_dfs = []
for gpu_name, path in files.items():
    if os.path.exists(path):
        df = preprocess_router_tokens(path, gpu_name=gpu_name)
        processed_dfs.append(df)
    else:
        print(f"❌ 파일 없음: {path}")

# ✅ 전체 병합
token_level_df = pd.concat(processed_dfs, ignore_index=True)

# ✅ layer별 token_count 계산 후 병합
layer_token_counts = (
    token_level_df.groupby(["text_id", "layer_type", "layer_index"])
    .size()
    .reset_index(name="layer_token_count")
)

token_level_df = pd.merge(
    token_level_df,
    layer_token_counts,
    on=["text_id", "layer_type", "layer_index"],
    how="left"
)

# ✅ 컬럼 순서 정리
cols_order = [
    "text_id", "layer_type", "layer_index", "expert_id",
    "token_id", "token_str", "router_entropy", 
    "router_score", "layer_token_count",
    "latency_ms", "gpu_id", 
]
token_level_df = token_level_df[cols_order]

# ✅ 저장
output_path = os.path.join(data_dir, "router_dataset.csv")
token_level_df.to_csv(output_path, index=False)
print("✅ 저장 완료:", output_path)