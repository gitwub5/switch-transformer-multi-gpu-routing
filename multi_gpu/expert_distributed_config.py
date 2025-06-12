import pandas as pd
import json
from collections import defaultdict

def create_gpu_distribution_config():
    # CSV 파일 읽기
    df = pd.read_csv("results/gpu_comparison_avg.csv")
    
    # GPU별 배치 구성을 저장할 딕셔너리
    gpu_config = {
        "1080Ti": {
            "encoder": defaultdict(list),
            "decoder": defaultdict(list)
        },
        "A6000": {
            "encoder": defaultdict(list),
            "decoder": defaultdict(list)
        }
    }
    
    # 각 행을 순회하면서 better_gpu에 따라 배치
    for _, row in df.iterrows():
        expert = row['expert']
        layer_type = row['layer_type']
        better_gpu = row['better_gpu']
        
        # expert 문자열에서 layer_index와 expert_id 추출
        # 예: "encoder_5_6" -> layer_index=5, expert_id=6
        parts = expert.split('_')
        layer_index = int(parts[1])
        expert_id = int(parts[2])
        
        # 해당 GPU의 layer_type에 expert 추가
        gpu_config[better_gpu][layer_type][layer_index].append(expert_id)
    
    # 각 GPU의 layer_type별로 expert_id 리스트 정렬
    for gpu in gpu_config:
        for layer_type in gpu_config[gpu]:
            for layer_index in gpu_config[gpu][layer_type]:
                gpu_config[gpu][layer_type][layer_index].sort()
    
    # JSON 파일로 저장
    with open("results/expert_distribution_config.json", "w") as f:
        json.dump(gpu_config, f, indent=4)
    
    # 배치 결과 출력
    print("\n=== GPU 배치 구성 ===")
    for gpu in gpu_config:
        print(f"\n[{gpu}]")
        for layer_type in gpu_config[gpu]:
            print(f"\n{layer_type.upper()}:")
            for layer_index in sorted(gpu_config[gpu][layer_type].keys()):
                experts = gpu_config[gpu][layer_type][layer_index]
                print(f"  Layer {layer_index}: {experts}")

if __name__ == "__main__":
    create_gpu_distribution_config()
