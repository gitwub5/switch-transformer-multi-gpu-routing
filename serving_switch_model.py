import torch
from transformers import AutoModelForSeq2SeqLM
from latency_aware_top1_router import LatencyAwareTop1Router

# 🧠 모델 로드 (Switch Transformer Base)
model_name = "google/switch-base-8"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 각 expert를 gpu에 분리하여 배치


# 🔧 latency 예측 모델 및 expert → gpu 매핑 설정
latency_model_path = "dynamic_routing/pt_models/latency_predictor.pt"
gpu_mapping_dict = {
    0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1
}  # 예시로 expert 0,2,4,6은 GPU 0으로, 1,3,5,7은 GPU 1로

# ✅ Encoder의 MoE block 탐색 및 교체
for i, block in enumerate(model.encoder.block):
    try:
        ff = block.layer[1].feed_forward
        if hasattr(ff, "router"):
            old_router = ff.router
            input_dim = old_router.classifier.in_features
            num_experts = old_router.classifier.out_features

            # 새로운 라우터 생성
            new_router = LatencyAwareTop1Router(
                input_dim=input_dim,
                num_experts=num_experts,
                latency_model_path=latency_model_path,
                gpu_mapping_dict=gpu_mapping_dict,
            )

            # 교체
            model.encoder.block[i].layer[1].feed_forward.router = new_router
            print(f"✅ Router replaced in encoder block {i}")

    except Exception as e:
        print(f"⚠️ Encoder block {i} skipped: {e}")

# ✅ Decoder의 MoE block 탐색 및 교체
for i, block in enumerate(model.decoder.block):
    try:
        ff = block.layer[1].feed_forward
        if hasattr(ff, "router"):
            old_router = ff.router
            input_dim = old_router.classifier.in_features
            num_experts = old_router.classifier.out_features

            new_router = LatencyAwareTop1Router(
                input_dim=input_dim,
                num_experts=num_experts,
                latency_model_path=latency_model_path,
                gpu_mapping_dict=gpu_mapping_dict,
            )

            model.decoder.block[i].layer[1].feed_forward.router = new_router
            print(f"✅ Router replaced in decoder block {i}")

    except Exception as e:
        print(f"⚠️ Decoder block {i} skipped: {e}")

# 🎉 완료
print("🔁 모든 Router 교체 작업 완료")
