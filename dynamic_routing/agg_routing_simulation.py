import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from train_routing import load_all_models, latency_aware_gpu_selector

def generate_random_router_outputs(num_tokens):
    """
    랜덤한 라우터 출력값 생성
    """
    # 랜덤한 logits 생성
    router_logits = torch.randn(num_tokens, 8)  # 8개의 expert에 대한 logits
    
    # softmax를 통한 확률 계산
    router_probs = torch.softmax(router_logits, dim=-1)
    
    # entropy 계산
    router_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
    
    # 각 토큰별 최대 확률값 (router_score)
    router_score = torch.max(router_probs, dim=-1)[0]
    
    return router_logits, router_probs, router_entropy, router_score

def simulate_token_routing(num_tokens, models):
    """
    하나의 토큰에 대한 전체 라우팅 과정 시뮬레이션
    """
    routing_history = []
    
    # Encoder layers (1, 3, 5, 7, 9, 11)
    for layer_idx in [1, 3, 5, 7, 9, 11]:
        # 랜덤한 라우터 출력값 생성
        router_logits, router_probs, router_entropy, router_score = generate_random_router_outputs(num_tokens)
        
        # 각 토큰에 대해 라우팅 수행
        for token_idx in range(num_tokens):
            # 현재 토큰의 expert 선택 확률
            token_probs = router_probs[token_idx]
            
            # top-1 routing: 가장 높은 확률을 가진 expert 선택
            selected_expert = torch.argmax(token_probs).item()
            
            token_feature = {
                "router_entropy": router_entropy[token_idx].item(),
                "router_score": router_score[token_idx].item(),
                "layer_token_count": num_tokens,
                "layer_type_encoded": 0,  # encoder
                "gpu_id_encoded": 0,  # 초기 GPU는 1080Ti
                "layer_index": layer_idx
            }
            
            # 선택된 expert에 대해 최적의 GPU 선택
            result = latency_aware_gpu_selector(token_feature, selected_expert, models)
            
            if result:
                routing_history.append({
                    "layer": f"encoder_{layer_idx}",
                    "token_idx": token_idx,
                    "expert_id": selected_expert,
                    "gpu_id": result["gpu_id"],
                    "latency": result["latency"]
                })
    
    # Decoder layers (1, 3, 5, 7, 9, 11)
    for layer_idx in [1, 3, 5, 7, 9, 11]:
        # 랜덤한 라우터 출력값 생성
        router_logits, router_probs, router_entropy, router_score = generate_random_router_outputs(num_tokens)
        
        # 각 토큰에 대해 라우팅 수행
        for token_idx in range(num_tokens):
            # 현재 토큰의 expert 선택 확률
            token_probs = router_probs[token_idx]
            
            # top-1 routing: 가장 높은 확률을 가진 expert 선택
            selected_expert = torch.argmax(token_probs).item()
            
            token_feature = {
                "router_entropy": router_entropy[token_idx].item(),
                "router_score": router_score[token_idx].item(),
                "layer_token_count": num_tokens,
                "layer_type_encoded": 1,  # decoder
                "gpu_id_encoded": 0,  # 초기 GPU는 1080Ti
                "layer_index": layer_idx
            }
            
            # 선택된 expert에 대해 최적의 GPU 선택
            result = latency_aware_gpu_selector(token_feature, selected_expert, models)
            
            if result:
                routing_history.append({
                    "layer": f"decoder_{layer_idx}",
                    "token_idx": token_idx,
                    "expert_id": selected_expert,
                    "gpu_id": result["gpu_id"],
                    "latency": result["latency"]
                })
    
    return routing_history

def analyze_routing_results(routing_history):
    """
    라우팅 결과 분석
    """
    # GPU 선택 분포
    gpu_selections = [r["gpu_id"] for r in routing_history]
    gpu_0_count = gpu_selections.count(0)  # 1080Ti
    gpu_1_count = gpu_selections.count(1)  # A6000
    
    # Expert 선택 분포
    expert_selections = [r["expert_id"] for r in routing_history]
    expert_distribution = {i: expert_selections.count(i) for i in range(8)}
    
    # 레이어별 평균 레이턴시
    layer_latencies = {}
    for r in routing_history:
        layer = r["layer"]
        if layer not in layer_latencies:
            layer_latencies[layer] = []
        layer_latencies[layer].append(r["latency"])
    
    layer_avg_latencies = {layer: np.mean(latencies) for layer, latencies in layer_latencies.items()}
    
    return {
        "gpu_distribution": {
            "1080Ti": gpu_0_count,
            "A6000": gpu_1_count
        },
        "expert_distribution": expert_distribution,
        "layer_avg_latencies": layer_avg_latencies
    }

def visualize_routing_path(routing_history, sim_idx):
    """
    단일 시뮬레이션의 라우팅 경로를 시각화
    """
    # 레이어 순서 정의
    layers = [f"encoder_{i}" for i in [1,3,5,7,9,11]] + [f"decoder_{i}" for i in [1,3,5,7,9,11]]
    
    # GPU별로 분리된 Expert 매트릭스 생성
    gpu0_matrix = np.zeros((len(layers), 8))  # 1080Ti
    gpu1_matrix = np.zeros((len(layers), 8))  # A6000
    
    # 라우팅 히스토리에서 데이터 채우기
    for r in routing_history:
        layer_idx = layers.index(r["layer"])
        expert_idx = r["expert_id"]
        if r["gpu_id"] == 0:  # 1080Ti
            gpu0_matrix[layer_idx, expert_idx] += 1
        else:  # A6000
            gpu1_matrix[layer_idx, expert_idx] += 1
    
    # 최대 토큰 수 계산 (색상 스케일링을 위해)
    max_tokens = max(np.max(gpu0_matrix), np.max(gpu1_matrix))
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    
    # 1080Ti 히트맵
    sns.heatmap(gpu0_matrix, ax=ax1, cmap="Blues", 
                xticklabels=[f"E{i}" for i in range(8)],
                yticklabels=layers,
                annot=True, fmt=".0f",
                vmin=0, vmax=max_tokens)  # 색상 스케일 통일
    ax1.set_title("1080Ti Expert Routing")
    ax1.set_xlabel("Expert ID")
    ax1.set_ylabel("Layer")
    
    # A6000 히트맵
    sns.heatmap(gpu1_matrix, ax=ax2, cmap="Reds",
                xticklabels=[f"E{i}" for i in range(8)],
                yticklabels=layers,
                annot=True, fmt=".0f",
                vmin=0, vmax=max_tokens)  # 색상 스케일 통일
    ax2.set_title("A6000 Expert Routing")
    ax2.set_xlabel("Expert ID")
    ax2.set_ylabel("Layer")
    
    plt.suptitle(f"Routing Path Visualization (Simulation {sim_idx})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/routing_path_sim_{sim_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_aggregated_routing(num_simulations, all_routing_histories):
    """
    300번 시뮬레이션의 결과를 종합하여 시각화
    """
    # 레이어 순서 정의
    layers = [f"encoder_{i}" for i in [1,3,5,7,9,11]] + [f"decoder_{i}" for i in [1,3,5,7,9,11]]
    
    # GPU별로 분리된 Expert 매트릭스 생성
    gpu0_matrix = np.zeros((len(layers), 8))  # 1080Ti
    gpu1_matrix = np.zeros((len(layers), 8))  # A6000
    
    # 모든 시뮬레이션 결과 집계
    for routing_history in all_routing_histories:
        for r in routing_history:
            layer_idx = layers.index(r["layer"])
            expert_idx = r["expert_id"]
            if r["gpu_id"] == 0:  # 1080Ti
                gpu0_matrix[layer_idx, expert_idx] += 1
            else:  # A6000
                gpu1_matrix[layer_idx, expert_idx] += 1
    
    # 최대 토큰 수 계산 (색상 스케일링을 위해)
    max_tokens = max(np.max(gpu0_matrix), np.max(gpu1_matrix))
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    
    # 1080Ti 히트맵
    sns.heatmap(gpu0_matrix, ax=ax1, cmap="Blues", 
                xticklabels=[f"E{i}" for i in range(8)],
                yticklabels=layers,
                annot=True, fmt=".0f",
                vmin=0, vmax=max_tokens)  # 색상 스케일 통일
    ax1.set_title("1080Ti Expert Routing (Total Tokens)")
    ax1.set_xlabel("Expert ID")
    ax1.set_ylabel("Layer")
    
    # A6000 히트맵
    sns.heatmap(gpu1_matrix, ax=ax2, cmap="Reds",
                xticklabels=[f"E{i}" for i in range(8)],
                yticklabels=layers,
                annot=True, fmt=".0f",
                vmin=0, vmax=max_tokens)  # 색상 스케일 통일
    ax2.set_title("A6000 Expert Routing (Total Tokens)")
    ax2.set_xlabel("Expert ID")
    ax2.set_ylabel("Layer")
    
    plt.suptitle(f"Aggregated Routing Path (300 Simulations)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/aggregated_routing_path.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 모델 로드
    MODEL_DIR = "models"  # train_routing.py에서 저장한 모델 디렉토리
    models = load_all_models(MODEL_DIR)
    
    # 시뮬레이션 파라미터
    num_tokens = 128  # 한 번에 처리할 토큰 수
    num_simulations = 300  # 시뮬레이션 횟수
    
    # 결과 저장을 위한 리스트
    all_routing_histories = []
    
    # 시뮬레이션 실행
    for sim_idx in range(num_simulations):
        print(f"Running simulation {sim_idx + 1}/{num_simulations}")
        routing_history = simulate_token_routing(num_tokens, models)
        all_routing_histories.append(routing_history)
        
        # # 각 시뮬레이션의 라우팅 경로 시각화
        # visualize_routing_path(routing_history, sim_idx)
        
        # # 결과 분석
        # results = analyze_routing_results(routing_history)
        
        # # 결과 출력
        print(f"\n=== 시뮬레이션 {sim_idx + 1} 결과 ===")
        # print("\n1. GPU 선택 분포:")
        # for gpu, count in results["gpu_distribution"].items():
        #     print(f"  - {gpu}: {count}회 선택")
        
        # print("\n2. Expert 선택 분포:")
        # for expert, count in results["expert_distribution"].items():
        #     print(f"  - Expert {expert}: {count}회 선택")
        
        # print("\n3. 레이어별 평균 레이턴시 (ms):")
        # for layer, latency in results["layer_avg_latencies"].items():
        #     print(f"  - {layer}: {latency:.6f}")
    
    # 전체 시뮬레이션 결과 종합 시각화
    visualize_aggregated_routing(num_simulations, all_routing_histories)

if __name__ == "__main__":
    main() 