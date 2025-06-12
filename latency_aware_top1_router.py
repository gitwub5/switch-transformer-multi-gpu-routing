import torch
import torch.nn as nn
import torch.nn.functional as F

class LatencyMLP(nn.Module):
    def __init__(self, input_dim=8):  # 입력 차원 주의
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
    
class LatencyAwareTop1Router(SwitchTransformersTop1Router):
    def __init__(self, input_dim, latency_model_path, gpu_mapping_dict, **kwargs):
        super().__init__(**kwargs)
        self.latency_model = LatencyMLP(input_dim)
        self.latency_model.load_state_dict(torch.load(latency_model_path))
        self.latency_model.eval().cuda()
        self.gpu_mapping_dict = gpu_mapping_dict  # {expert_id: gpu_id}

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # ✅ 1. 기존 라우팅 로직 유지
        expert_index, router_probs, router_logits = super().forward(hidden_states)

        # ✅ 2. 라우팅 정보 기반 토큰별 latency-aware GPU 선택 로직
        batch_size, seq_len, _ = hidden_states.shape
        selected_gpu_ids = torch.zeros(batch_size, seq_len, dtype=torch.int32)

        # 현재 GPU ID 확인
        device_name = torch.cuda.get_device_name(0)  # 현재 사용 중인 GPU 이름
        if "1080" in device_name:
            gpu_id_encoded = 0  # 1080Ti
        else:
            gpu_id_encoded = 1  # A6000

        for i in range(batch_size):
            for j in range(seq_len):
                # 어떤 expert로 라우팅되었는지 확인
                expert_id = torch.argmax(expert_index[i, j]).item()
                router_score = router_probs[i, j].item()
                entropy = -torch.sum(router_logits[i, j] * torch.log_softmax(router_logits[i, j], dim=-1)).item()

                # 🔧 feature vector 구성
                features = torch.tensor([
                    0.0,  # token_id 자리 (token_id값 받아와야함)
                    float(expert_id), # expert_id
                    float(kwargs.get("layer_index", 0)), # layer_index
                    float(entropy), # router_entropy
                    float(router_score), # router_score
                    float(kwargs.get("layer_token_count", 128)), # layer_token_count
                    float(kwargs.get("layer_type_encoded", 0)), # layer_type_encoded
                    float(gpu_id_encoded) # gpu_id_encoded (현재 gpu 위치)
                ]).unsqueeze(0).cuda()

                with torch.no_grad():
                    predicted_latency = self.latency_model(features).item()

                # 🔁 예시: expert-to-GPU mapping 사용
                selected_gpu_ids[i, j] = self.gpu_mapping_dict.get(expert_id, 0)

        # 🟡 선택된 GPU ID는 별도 반환 (또는 routing table에서 활용)
        return expert_index, router_probs, router_logits, selected_gpu_ids