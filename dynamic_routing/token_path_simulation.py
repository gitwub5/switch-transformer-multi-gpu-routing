import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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
                layer_index = int(fname.replace("layer_", "").replace(".joblib", ""))
                model = joblib.load(os.path.join(type_path, fname))
                models[(layer_type, layer_index)] = model

    return models

# ============================
# 라우팅 경로 시뮬레이션
# ============================
def simulate_routing_path(token_feature: dict, models: dict):
    """
    단일 토큰의 전체 라우팅 경로를 시뮬레이션
    
    Returns:
        list: [(layer_index, expert_id, gpu_id, latency), ...]
    """
    path = []
    current_layer = token_feature["layer_index"]
    current_type = "encoder" if token_feature["layer_type_encoded"] == 0 else "decoder"
    
    # 라우팅 가능한 layer 정의
    ROUTING_LAYERS = {
        "encoder": [1, 3, 5, 7, 9, 11],
        "decoder": [1, 3, 5, 7, 9, 11]
    }
    
    # 현재 layer가 라우팅 가능한 layer인지 확인
    if current_layer not in ROUTING_LAYERS[current_type]:
        return path
    
    # 현재 layer의 모델 가져오기
    model_key = (current_type, current_layer)
    if model_key not in models:
        return path
    
    model = models[model_key]
    
    # 랜덤하게 expert 선택
    expert_id = np.random.randint(0, 8)  # 0부터 7까지 랜덤 선택
    gpu_id = np.random.randint(0, 2)  # 0 또는 1 랜덤 선택
    
    # 특성 벡터 생성 (token_id 제외)
    feature_vec = [
        token_feature["router_entropy"],
        token_feature["router_score"],
        token_feature["layer_token_count"],
        expert_id,
        gpu_id
    ]
    
    pred_latency = model.predict([feature_vec])[0]
    pred_latency = max(0.0, pred_latency)
    
    path.append((current_layer, expert_id, gpu_id, pred_latency))
    
    return path

# ============================
# 경로 시각화
# ============================
def visualize_routing_path(path, output_path=None):
    """
    라우팅 경로를 시각화
    
    Args:
        path: [(layer_index, expert_id, gpu_id, latency), ...]
        output_path: 저장할 파일 경로
    """
    plt.figure(figsize=(15, 8))
    
    # Encoder와 Decoder 경로 분리
    encoder_path = path[:6]  # 처음 6개는 encoder
    decoder_path = path[6:]  # 나머지는 decoder
    
    # 노드 위치 계산
    def get_node_positions(path, start_x, y_offset=0):
        positions = {}
        for i, (layer, expert, gpu, _) in enumerate(path):
            x = start_x + i * 2
            y = y_offset
            positions[f"L{layer}_E{expert}_G{gpu}"] = (x, y)
        return positions
    
    # Encoder와 Decoder 노드 위치 계산
    encoder_pos = get_node_positions(encoder_path, 0, 1)
    decoder_pos = get_node_positions(decoder_path, 0, 0)
    pos = {**encoder_pos, **decoder_pos}
    
    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가
    for i, (layer, expert, gpu, latency) in enumerate(path):
        node_id = f"L{layer}_E{expert}_G{gpu}"
        is_encoder = i < 6
        G.add_node(node_id, 
                  layer=layer,
                  expert=expert,
                  gpu="1080Ti" if gpu == 0 else "A6000",
                  latency=f"{latency:.3f}ms",
                  type="Encoder" if is_encoder else "Decoder")
    
    # 엣지 추가
    for i in range(len(path)-1):
        G.add_edge(f"L{path[i][0]}_E{path[i][1]}_G{path[i][2]}",
                  f"L{path[i+1][0]}_E{path[i+1][1]}_G{path[i+1][2]}")
    
    # 노드 그리기
    encoder_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == "Encoder"]
    decoder_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == "Decoder"]
    
    nx.draw_networkx_nodes(G, pos, nodelist=encoder_nodes, 
                          node_color='lightblue', node_size=2000, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=decoder_nodes, 
                          node_color='lightgreen', node_size=2000, alpha=0.7)
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # 레이블 그리기
    labels = {node: f"{G.nodes[node]['layer']}\nExpert {G.nodes[node]['expert']}\n{G.nodes[node]['gpu']}\n{G.nodes[node]['latency']}"
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # 제목과 범례 추가
    plt.title("Token Routing Path", pad=20)
    plt.text(0.02, 0.98, "Encoder", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(0.02, 0.02, "Decoder", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================
# 전체 라우팅 경로 시뮬레이션 (순차 연결)
# ============================
def simulate_full_routing_path(token_feature: dict, models: dict):
    """
    Encoder 1~11 → Decoder 1~11을 순차적으로 라우팅하며 전체 경로를 반환
    """
    path = []
    token = token_feature.copy()
    encoder_layers = [1, 3, 5, 7, 9, 11]
    decoder_layers = [1, 3, 5, 7, 9, 11]

    # Encoder
    for layer in encoder_layers:
        token["layer_index"] = layer
        token["layer_type_encoded"] = 0
        step = simulate_routing_path(token, models)
        if step:
            path.extend(step)
    # Decoder
    for layer in decoder_layers:
        token["layer_index"] = layer
        token["layer_type_encoded"] = 1
        step = simulate_routing_path(token, models)
        if step:
            path.extend(step)
    return path

# ============================
# 실행
# ============================
if __name__ == "__main__":
    MODEL_DIR = "models"
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 모델 로드
    models = load_all_models(MODEL_DIR)
    
    # 전체 경로 시뮬레이션을 위한 토큰
    token = {
        "layer_index": 1,
        "router_entropy": 0.32,
        "router_score": 0.5,
        "layer_token_count": 128,
        "layer_type_encoded": 0,  # encoder
        "gpu_id_encoded": 0  # 1080Ti
    }
    
    # 전체 경로 시뮬레이션 (순차 연결)
    full_path = simulate_full_routing_path(token, models)
    
    # 결과 출력
    print("\n🔀 전체 라우팅 경로:")
    encoder_layers = [1, 3, 5, 7, 9, 11]
    for i, (layer, expert, gpu, latency) in enumerate(full_path):
        layer_type = "Encoder" if i < len(encoder_layers) else "Decoder"
        print(f"  {layer_type} Layer {layer} -> Expert {expert} -> GPU {gpu} ({'1080Ti' if gpu == 0 else 'A6000'})")
        print(f"  예상 지연시간: {latency:.3f}ms")
    
    # 경로 시각화
    visualize_routing_path(full_path, os.path.join(OUTPUT_DIR, "full_routing_path.png")) 