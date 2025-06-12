import re
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

# ✅ 결과 저장 딕셔너리
flops_profiles_dict = {}

def get_flops_hook(layer_name):
    def hook(module, input, output):
        match = re.search(r"(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\.mlp\.experts\.expert_(\d+)\.(wi|wo)", layer_name)
        if not match:
            print(f"[SKIP] Unmatched: {layer_name}")
            return

        layer_type = match.group(1)
        layer_index = int(match.group(2))
        expert_id = int(match.group(4))
        sub_layer = match.group(5)  # wi or wo

        key = (current_text_id, layer_type, layer_index, expert_id)

        input_tensor = input[0] if input else None
        flops = 0
        if isinstance(module, torch.nn.Linear) and input_tensor is not None:
            batch_size = input_tensor.shape[0]
            flops = 2 * batch_size * module.in_features * module.out_features

        # ✅ dict에 누적 저장
        if key not in flops_profiles_dict:
            flops_profiles_dict[key] = {
                "text_id": current_text_id,
                "layer_type": layer_type,
                "layer_index": layer_index,
                "expert_id": expert_id,
                "flops": 0,
                "input_shape": None,
                "output_shape": None,
            }

        flops_profiles_dict[key]["flops"] += flops
        if sub_layer == "wi":
            flops_profiles_dict[key]["input_shape"] = list(input_tensor.shape) if input_tensor is not None else None
        elif sub_layer == "wo":
            flops_profiles_dict[key]["output_shape"] = list(output.shape) if isinstance(output, torch.Tensor) else None

    return hook
        
# ✅ Hook 등록 (wi, wo만 대상)
for name, module in model.named_modules():
    if re.search(r"(encoder|decoder)\.block\.\d+\.layer\.\d+\.mlp\.experts\.expert_\d+\.(wi|wo)$", name):
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(get_flops_hook(name))

# ✅ 문장별 실행
for entry in texts:
    current_text_id = entry["text_id"]
    text = entry["text"]

    inputs = tokenizer(text, return_tensors="pt").to(device)
    decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.pad_token_id
    # inputs["decoder_input_ids"] = torch.tensor([[decoder_start_token_id]], device=device)
    inputs["decoder_input_ids"] = inputs["input_ids"].clone() # encoder와 decoder가 다루는 토큰 수를 동일하게 설정

    with torch.no_grad():
        model(**inputs)

# ✅ 결과 저장
df = pd.DataFrame(list(flops_profiles_dict.values()))
df.to_csv(f"../outputs/dense_profile_{device_name}.csv", index=False)
print(df.head())
