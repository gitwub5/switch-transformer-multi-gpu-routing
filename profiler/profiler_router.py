import re
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.gpu_utils import select_gpu
from collections import defaultdict
from utils.load_dataset import load_profiling_dataset

# GPU 설정
device = select_gpu()
torch.cuda.set_device(device)
device_name = torch.cuda.get_device_name(device)

# 모델 및 토크나이저 로드
model_name = "google/switch-base-8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

# ✅ 처리할 문장 리스트 (JSON 형식)
dataset_path = "../data/structured_texts_balanced_5000.json"
texts = load_profiling_dataset(dataset_path)

# 결과 저장 리스트
records = []

# Hook 함수 정의
def get_router_output_hook(layer_name):
    def hook(module, input, output):
        match = re.search(r"(encoder|decoder)\.block\.(\d+)", layer_name)
        if not match:
            return
        layer_type = match.group(1)
        layer_index = int(match.group(2))

        logits = output  # [B, T, E]
        softmax_scores = torch.softmax(logits, dim=-1)  # [B, T, E]
        expert_ids = torch.argmax(logits, dim=-1)  # [B, T]

        routed_tokens_by_expert = defaultdict(list)

        for token_idx, expert_id in enumerate(expert_ids[0].tolist()):
            routed_tokens_by_expert[expert_id].append({
                "token_id": current_token_ids[token_idx],
                "token_str": current_token_strs[token_idx],
                "router_score": softmax_scores[0, token_idx, expert_id].item(),
                "router_entropy": -torch.sum(
                    softmax_scores[0, token_idx] * torch.log(softmax_scores[0, token_idx] + 1e-9)
                ).item()
            })

        # 저장
        for expert_id, tokens in routed_tokens_by_expert.items():
            records.append({
                "text_id": current_text_id,
                "layer_type": layer_type,
                "layer_index": layer_index,
                "expert_id": expert_id,
                "token_count": len(tokens),
                "routed_token_ids": [t["token_id"] for t in tokens],
                "routed_token_strs": [t["token_str"] for t in tokens],
                "router_scores": [t["router_score"] for t in tokens],
                "router_entropies": [t["router_entropy"] for t in tokens],
            })
    return hook

# Hook 등록
for name, module in model.named_modules():
    if "router.classifier" in name:
        module.register_forward_hook(get_router_output_hook(name))


# ✅ 문장별로 forward 실행
for entry in texts:
    current_text_id = entry["text_id"]
    text = entry["text"]

    inputs = tokenizer(text, return_tensors="pt").to(device)
    decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.pad_token_id
    # inputs["decoder_input_ids"] = torch.tensor([[decoder_start_token_id]], device=device)
    inputs["decoder_input_ids"] = inputs["input_ids"].clone() # encoder와 decoder가 다루는 토큰 수를 동일하게 설정

    # 현재 토큰 정보 저장 (hook에서 참조 가능하도록 전역 변수로)
    current_token_ids = inputs["input_ids"][0].tolist()
    current_token_strs = tokenizer.convert_ids_to_tokens(current_token_ids)

    with torch.no_grad():
        model(**inputs)

# ✅ 결과 저장
df = pd.DataFrame(records)
df.to_csv(f"../outputs/router_profile_{device_name}.csv", index=False)
print(df.head())