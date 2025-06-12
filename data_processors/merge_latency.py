import pandas as pd

# ✅ CSV 파일 경로
path_1080ti = "../outputs/merged_profile_NVIDIA GeForce GTX 1080 Ti.csv"
path_a6000 = "../outputs/merged_profile_NVIDIA RTX A6000.csv"

# ✅ 데이터 불러오기
df_1080ti = pd.read_csv(path_1080ti)
df_a6000 = pd.read_csv(path_a6000)

# ✅ 필요한 컬럼만 선택
cols = ["text_id", "layer_type", "layer_index", "expert_id", "flops", "latency_ms"]
df_1080ti = df_1080ti[cols].copy()
df_a6000 = df_a6000[cols].copy()

# ✅ 컬럼 이름 변경 (latency_ms → latency_XXX)
df_1080ti = df_1080ti.rename(columns={"latency_ms": "latency_1080Ti"})
df_a6000 = df_a6000.rename(columns={"latency_ms": "latency_A6000"})

# ✅ 병합 키: expert를 유일하게 식별할 수 있는 컬럼
merge_keys = ["text_id", "layer_type", "layer_index", "expert_id", "flops"]

# ✅ 내부 조인 (같은 expert + 같은 문장에 대해 비교)
df_merged = pd.merge(df_1080ti, df_a6000, on=merge_keys, how="inner")

# ✅ 결과 확인
print(f"병합된 데이터 수: {len(df_merged)}")
print(df_merged.head())

# ✅ 저장
df_merged.to_csv("../outputs/merged_latency_comparison.csv", index=False)