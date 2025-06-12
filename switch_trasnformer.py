# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # === MLP Latency Predictor ===
# class LatencyMLP(nn.Module):
#     def __init__(self, input_dim=7):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, x):
#         return self.model(x)


# # === Switch Router ===
# class SwitchTransformersTop1Router(nn.Module):
#     def __init__(self, input_dim, num_experts, expert_capacity):
#         super().__init__()
#         self.num_experts = num_experts
#         self.expert_capacity = expert_capacity
#         self.classifier = nn.Linear(input_dim, num_experts, bias=False)

#     def forward(self, hidden_states):
#         logits = self.classifier(hidden_states)
#         probs = F.softmax(logits, dim=-1)
#         top1 = torch.argmax(probs, dim=-1)
#         expert_index = F.one_hot(top1, num_classes=self.num_experts).float()

#         # Capacity 제한 적용
#         token_priority = torch.cumsum(expert_index, dim=-2)
#         expert_capacity_mask = token_priority <= self.expert_capacity
#         expert_index = expert_index * expert_capacity_mask

#         max_probs = torch.max(probs, dim=-1).values.unsqueeze(-1)
#         return expert_index, max_probs, logits


# # === Latency-Aware Router ===
# class LatencyAwareTop1Router(SwitchTransformersTop1Router):
#     def __init__(self, input_dim, num_experts, expert_capacity, latency_model_path, gpu_mapping_dict):
#         super().__init__(input_dim, num_experts, expert_capacity)
#         self.latency_model = LatencyMLP(input_dim=7)
#         self.latency_model.load_state_dict(torch.load(latency_model_path))
#         self.latency_model.eval().cuda()
#         self.gpu_mapping_dict = gpu_mapping_dict

#     def forward(self, hidden_states, **kwargs):
#         expert_index, router_probs, router_logits = super().forward(hidden_states)

#         batch_size, seq_len, _ = hidden_states.shape
#         selected_gpu_ids = torch.zeros(batch_size, seq_len, dtype=torch.int32)

#         layer_index = kwargs["layer_index"]
#         layer_type_encoded = kwargs["layer_type_encoded"]
#         layer_token_count = kwargs.get("layer_token_count", 128)
#         gpu_id_encoded = kwargs.get("gpu_id_encoded", 0)

#         for i in range(batch_size):
#             for j in range(seq_len):
#                 expert_id = torch.argmax(expert_index[i, j]).item()
#                 router_score = router_probs[i, j].item()
#                 entropy = -torch.sum(F.softmax(router_logits[i, j], dim=-1) * F.log_softmax(router_logits[i, j], dim=-1)).item()

#                 features = torch.tensor([
#                     float(expert_id),
#                     float(layer_index),
#                     float(entropy),
#                     float(router_score),
#                     float(layer_token_count),
#                     float(layer_type_encoded),
#                     float(gpu_id_encoded)
#                 ]).unsqueeze(0).cuda()

#                 with torch.no_grad():
#                     predicted_latency = self.latency_model(features).item()

#                 selected_gpu_ids[i, j] = self.gpu_mapping_dict.get(expert_id, 0)

#         return expert_index, router_probs, router_logits, selected_gpu_ids


# # === Example Switch Transformer Block ===
# class SwitchTransformerLayer(nn.Module):
#     def __init__(self, input_dim, num_experts, expert_capacity, latency_model_path, gpu_mapping_dict):
#         super().__init__()
#         self.router = LatencyAwareTop1Router(
#             input_dim=input_dim,
#             num_experts=num_experts,
#             expert_capacity=expert_capacity,
#             latency_model_path=latency_model_path,
#             gpu_mapping_dict=gpu_mapping_dict
#         )
#         # Expert module 예시 (공통 레이어)
#         self.experts = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts)])

#     def forward(self, hidden_states, **kwargs):
#         expert_index, router_probs, router_logits, selected_gpu_ids = self.router(hidden_states, **kwargs)
#         output = torch.zeros_like(hidden_states)

#         for i in range(self.router.num_experts):
#             mask = expert_index[:, :, i].unsqueeze(-1)
#             expert_input = hidden_states * mask
#             expert_output = self.experts[i](expert_input)
#             output += expert_output * mask

#         return output, selected_gpu_ids


# # === Full Transformer Model 예시 ===
# class SimpleSwitchTransformer(nn.Module):
#     def __init__(self, num_layers, input_dim, num_experts, expert_capacity, latency_model_path, gpu_mapping_dict):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             SwitchTransformerLayer(input_dim, num_experts, expert_capacity, latency_model_path, gpu_mapping_dict)
#             for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x, _ = layer(x, layer_index=i, layer_type_encoded=0, gpu_id_encoded=0)
#         return x
