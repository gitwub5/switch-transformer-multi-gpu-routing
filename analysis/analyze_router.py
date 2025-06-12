import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import inspect

def load_switch_model(model_name="google/switch-base-8"):
    """Switch-Base-8 모델을 직접 로드하는 함수"""
    print("모델 로딩 중...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def analyze_router_structure(model):
    """Router 구조 분석"""
    router_info = []
    
    # Encoder 분석
    for i, encoder_layer in enumerate(model.encoder.block):
        try:
            if hasattr(encoder_layer.layer[1], 'mlp'):
                mlp = encoder_layer.layer[1].mlp
                if hasattr(mlp, 'router') and hasattr(mlp, 'experts'):
                    router = mlp.router
                    # Router의 상세 구조 분석
                    router_details = {
                        'layer_type': 'encoder',
                        'block': i,
                        'router_type': 'encoder_router',
                        'router_params': sum(p.numel() for p in router.parameters()),
                        'num_experts': router.classifier.out_features,
                        'router_input_dim': router.classifier.in_features,
                        'router_output_dim': router.classifier.out_features,
                        'router_weights_shape': str(router.classifier.weight.shape),
                        'router_bias': router.classifier.bias is not None,
                        'router_forward_signature': str(inspect.signature(router.forward)),
                        'router_class': type(router).__name__,
                        'router_module_path': '.'.join([type(router).__module__, type(router).__name__])
                    }
                    
                    # Router의 forward 메소드 분석
                    router_forward = router.forward
                    router_forward_source = inspect.getsource(router_forward)
                    router_details['router_forward_source'] = router_forward_source
                    
                    router_info.append(router_details)
                    print(f"\nEncoder Layer {i} Router 상세 구조:")
                    print(f"Input Dimension: {router_details['router_input_dim']}")
                    print(f"Output Dimension (Number of Experts): {router_details['router_output_dim']}")
                    print(f"Weights Shape: {router_details['router_weights_shape']}")
                    print(f"Has Bias: {router_details['router_bias']}")
                    print(f"\nRouter 구현:")
                    print(f"Class: {router_details['router_class']}")
                    print(f"Module Path: {router_details['router_module_path']}")
                    print(f"Forward Signature: {router_details['router_forward_signature']}")
                    print("\nRouter Forward 구현:")
                    print(router_forward_source)
        except Exception as e:
            print(f"Encoder Layer {i} 분석 중 오류 발생: {str(e)}")
            continue
    
    # Decoder 분석
    for i, decoder_layer in enumerate(model.decoder.block):
        try:
            if hasattr(decoder_layer.layer[2], 'mlp'):
                mlp = decoder_layer.layer[2].mlp
                if hasattr(mlp, 'router') and hasattr(mlp, 'experts'):
                    router = mlp.router
                    # Router의 상세 구조 분석
                    router_details = {
                        'layer_type': 'decoder',
                        'block': i,
                        'router_type': 'decoder_router',
                        'router_params': sum(p.numel() for p in router.parameters()),
                        'num_experts': router.classifier.out_features,
                        'router_input_dim': router.classifier.in_features,
                        'router_output_dim': router.classifier.out_features,
                        'router_weights_shape': str(router.classifier.weight.shape),
                        'router_bias': router.classifier.bias is not None,
                        'router_forward_signature': str(inspect.signature(router.forward)),
                        'router_class': type(router).__name__,
                        'router_module_path': '.'.join([type(router).__module__, type(router).__name__])
                    }
                    
                    # Router의 forward 메소드 분석
                    router_forward = router.forward
                    router_forward_source = inspect.getsource(router_forward)
                    router_details['router_forward_source'] = router_forward_source
                    
                    router_info.append(router_details)
                    print(f"\nDecoder Layer {i} Router 상세 구조:")
                    print(f"Input Dimension: {router_details['router_input_dim']}")
                    print(f"Output Dimension (Number of Experts): {router_details['router_output_dim']}")
                    print(f"Weights Shape: {router_details['router_weights_shape']}")
                    print(f"Has Bias: {router_details['router_bias']}")
                    print(f"\nRouter 구현:")
                    print(f"Class: {router_details['router_class']}")
                    print(f"Module Path: {router_details['router_module_path']}")
                    print(f"Forward Signature: {router_details['router_forward_signature']}")
                    print("\nRouter Forward 구현:")
                    print(router_forward_source)
        except Exception as e:
            print(f"Decoder Layer {i} 분석 중 오류 발생: {str(e)}")
            continue
    
    if not router_info:
        print("\n경고: Router 구조를 찾을 수 없습니다.")
        print("모델의 구조를 확인해보니 예상과 다른 구조를 가지고 있습니다.")
        print("모델 구조를 분석하여 정확한 경로를 찾아야 합니다.")
    
    return pd.DataFrame(router_info)

def visualize_router_analysis(df):
    """Router 분석 결과 시각화"""
    # 결과 저장 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    
    # 1. Router 파라미터 분포
    plt.figure(figsize=(15, 10))
    
    # 1-1. Encoder/Decoder별 Router 파라미터
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='layer_type', y='router_params')
    plt.title('Router Parameters by Layer Type')
    plt.xlabel('Layer Type')
    plt.ylabel('Number of Parameters')
    
    # 1-2. Layer별 Router 파라미터
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='block', y='router_params', hue='layer_type')
    plt.title('Router Parameters by Layer')
    plt.xlabel('Layer Block')
    plt.ylabel('Number of Parameters')
    
    # 2. Expert vs Router 파라미터 비교
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='router_params', y='router_params', hue='layer_type')
    plt.title('Expert vs Router Parameters')
    plt.xlabel('Router Parameters')
    plt.ylabel('Router Parameters')
    
    # 3. 전체 파라미터 중 Router 비율
    plt.subplot(2, 2, 4)
    df['router_ratio'] = df['router_params'] / df['router_params'].sum() * 100
    sns.barplot(data=df, x='layer_type', y='router_ratio')
    plt.title('Router Parameters Ratio')
    plt.xlabel('Layer Type')
    plt.ylabel('Percentage of Total Parameters')
    
    plt.tight_layout()
    plt.savefig('router_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_router_summary(df):
    """Router 구조 요약 정보 출력"""
    print("\n=== Router Structure Summary ===")
    print(f"\nTotal Layers: {len(df)}")
    print(f"Encoder Layers: {len(df[df['layer_type'] == 'encoder'])}")
    print(f"Decoder Layers: {len(df[df['layer_type'] == 'decoder'])}")
    
    print("\nRouter Parameters:")
    print(f"Average Router Parameters: {df['router_params'].mean():.2f}")
    print(f"Total Router Parameters: {df['router_params'].sum():,}")
    
    print("\nRouter Architecture Details:")
    print(f"Input Dimension: {df['router_input_dim'].iloc[0]}")
    print(f"Output Dimension (Number of Experts): {df['router_output_dim'].iloc[0]}")
    print(f"Weights Shape: {df['router_weights_shape'].iloc[0]}")
    print(f"Has Bias: {df['router_bias'].iloc[0]}")
    
    print("\nRouter Implementation Details:")
    print(f"Class: {df['router_class'].iloc[0]}")
    print(f"Module Path: {df['router_module_path'].iloc[0]}")
    print(f"Forward Signature: {df['router_forward_signature'].iloc[0]}")
    
    print("\nLayer-wise Router Parameters:")
    layer_summary = df.groupby(['layer_type', 'block'])['router_params'].mean().unstack()
    print(layer_summary)
    
    # 결과를 CSV로 저장
    df.to_csv('router_analysis_summary.csv', index=False)

def main():
    # Switch-Base-8 모델 로드
    print("모델 로딩 중...")
    model, tokenizer = load_switch_model()
    
    # Router 구조 분석
    print("\nRouter 구조 분석 중...")
    router_df = analyze_router_structure(model)
    
    if not router_df.empty:
        # 분석 결과 출력
        print_router_summary(router_df)
        
        # 시각화
        # visualize_router_analysis(router_df)
    else:
        print("\nRouter 구조를 찾을 수 없어 분석을 중단합니다.")
        print("모델 구조를 확인하고 코드를 수정해야 합니다.")

if __name__ == "__main__":
    main() 