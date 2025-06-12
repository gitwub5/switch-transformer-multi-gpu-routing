import re
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.gpu_utils import select_gpu
from utils.load_dataset import load_profiling_dataset

# ✅ GPU 설정
device = select_gpu()
torch.cuda.set_device(device)
device_name = torch.cuda.get_device_name(device)

# ✅ 모델 및 토크나이저 로드
model_name = "google/switch-base-8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

# ✅ 입력 텍스트 (JSON 형식)
dataset_path = "../data/structured_texts_balanced_5000.json"
texts = load_profiling_dataset(dataset_path)

# ✅ 결과 저장 리스트
expert_profiles = []

# ✅ Hook 함수 정의
def get_expert_profile_hook(layer_name):
    def hook(module, input, output):
        match = re.search(r"(encoder|decoder)\.block\.(\d+).*expert_(\d+)", layer_name)
        if not match:
            return
        layer_type = match.group(1)
        layer_index = int(match.group(2))
        expert_id = int(match.group(3))

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        mem_bytes = torch.cuda.memory_allocated(device)

        expert_profiles.append({
            "text_id": current_text_id,
            "layer_type": layer_type,
            "layer_index": layer_index,
            "expert_id": expert_id,
            "latency_ms": elapsed_ms,
            "memory_allocated_bytes": mem_bytes,
        })
    return hook

# ✅ 모든 expert 모듈에 hook 등록
for name, module in model.named_modules():
    if re.match(r".*experts\.expert_\d+$", name):
        module.register_forward_hook(get_expert_profile_hook(name))

# ✅ 문장별 실행
for entry in texts:
    current_text_id = entry["text_id"]
    text = entry["text"]

    # 입력 토큰화
    inputs = tokenizer(text, return_tensors="pt").to(device)
    decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.pad_token_id
    # inputs["decoder_input_ids"] = torch.tensor([[decoder_start_token_id]], device=device)
    inputs["decoder_input_ids"] = inputs["input_ids"].clone() # encoder와 decoder가 다루는 토큰 수를 동일하게 설정

    # 실행 전 메모리 초기화
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        model(**inputs)

# ✅ 결과 저장
df = pd.DataFrame(expert_profiles)
df.to_csv(f"../outputs/expert_profile_{device_name}.csv", index=False)
print(df.head())