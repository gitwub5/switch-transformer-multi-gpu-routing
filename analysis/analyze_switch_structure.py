import torch
from transformers import AutoModelForSeq2SeqLM
from collections import defaultdict
import json
from datetime import datetime

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

def analyze_model_structure(model, prefix=''):
    structure_info = []

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        num_params = count_parameters(module)
        structure_info.append({
            'layer_name': full_name,
            'layer_type': type(module).__name__,
            'num_params': num_params
        })

        # 재귀 호출로 서브모듈 탐색
        structure_info.extend(analyze_model_structure(module, prefix=full_name))

    return structure_info

if __name__ == "__main__":
    model_name = "google/switch-base-8"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    structure_info = analyze_model_structure(model)
    
    # 결과를 JSON 파일로 저장
    output_file = f"../outputs/model_structure.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'total_parameters': sum(item['num_params'] for item in structure_info),
            'structure': structure_info
        }, f, indent=2, ensure_ascii=False)
    
    print(f"분석 결과가 {output_file}에 저장되었습니다.")
    
    # 콘솔에도 출력
    print(f"\n{'Layer Name':<60} {'Layer Type':<40} {'#Params'}")
    print("-" * 110)
    for item in structure_info:
        print(f"{item['layer_name']:<60} {item['layer_type']:<40} {item['num_params']}")