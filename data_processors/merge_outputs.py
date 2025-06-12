import pandas as pd
import os

# ✅ 파일 경로
data_dir = "../outputs"
device_suffix = "NVIDIA  GeForce GTX 1080 Ti"
# device_suffix = "NVIDIA RTX A6000"

dense_path = os.path.join(data_dir, f"dense_profile_{device_suffix}.csv")
expert_path = os.path.join(data_dir, f"expert_profile_{device_suffix}.csv")
router_path = os.path.join(data_dir, f"router_profile_{device_suffix}.csv")

# ✅ 파일 존재 확인
for path in [dense_path, expert_path, router_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

# ✅ CSV 파일 읽기
dense_df = pd.read_csv(dense_path)
expert_df = pd.read_csv(expert_path)
router_df = pd.read_csv(router_path)

# ✅ 병합 기준 키
merge_keys = ["text_id", "layer_type", "layer_index", "expert_id"]

# ✅ 병합 수행
merged_df = pd.merge(dense_df, expert_df, how="outer", on=merge_keys)
merged_df = pd.merge(merged_df, router_df, how="outer", on=merge_keys)

# ✅ 정렬: text_id → step 순
if "step" in merged_df.columns:
    merged_df = merged_df.sort_values(by=["text_id", "step"])
else:
    merged_df = merged_df.sort_values(by=["text_id"])

# ✅ 출력
print(f"총 병합된 row 수: {len(merged_df)}")
print(merged_df.head())

# ✅ 저장
output_path = os.path.join(data_dir, f"merged_profile_{device_suffix}.csv")
merged_df.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}")