import torch
import subprocess
import platform

def get_gpu_info():
    print("\n=== GPU 정보 ===")
    
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {'✅' if cuda_available else '❌'}")
    
    if cuda_available:
        # GPU 개수
        gpu_count = torch.cuda.device_count()
        print(f"\n사용 가능한 GPU 개수: {gpu_count}")
        
        # 각 GPU의 상세 정보
        print("\n=== GPU 상세 정보 ===")
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  - 이름: {torch.cuda.get_device_name(i)}")
            print(f"  - 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 현재 사용 중인 메모리
            if torch.cuda.is_available():
                print(f"  - 현재 사용 중인 메모리: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"  - 캐시된 메모리: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        
        # CUDA 버전
        print(f"\nCUDA 버전: {torch.version.cuda}")
        
        # cuDNN 버전
        if torch.backends.cudnn.is_available():
            print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"cuDNN 사용 가능: ✅")
        else:
            print("cuDNN 사용 가능: ❌")
    else:
        print("\nGPU가 감지되지 않았습니다.")
    
    # 시스템 정보
    print("\n=== 시스템 정보 ===")
    print(f"운영체제: {platform.system()} {platform.release()}")
    print(f"Python 버전: {platform.python_version()}")
    print(f"PyTorch 버전: {torch.__version__}")

def get_available_gpus():
    """사용 가능한 GPU 목록을 반환합니다."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def select_gpu():
    """사용 가능한 GPU 중 하나를 선택합니다."""
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
        return torch.device("cpu")
    
    print("\n사용 가능한 GPU 목록:")
    for i, gpu in enumerate(available_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"{i}: {device_name} ({gpu})")
    
    while True:
        try:
            choice = int(input("\n사용할 GPU 번호를 선택하세요: "))
            if 0 <= choice < len(available_gpus):
                selected_device = f"cuda:{choice}"
                print(f"\n선택된 디바이스: {selected_device}")
                print(f"디바이스 이름: {torch.cuda.get_device_name(choice)}")
                return torch.device(selected_device)  # ✅ 여기를 변경
            else:
                print("유효하지 않은 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("숫자를 입력해주세요.")
