import json

def load_profiling_dataset(path: str, max_samples: int = None):
    """
    저장된 JSON 형식의 profiling용 텍스트 데이터셋을 불러옵니다.

    Args:
        path (str): JSON 파일 경로
        max_samples (int, optional): 불러올 최대 샘플 수 (None이면 전체 사용)

    Returns:
        List[Dict]: [{"text_id": ..., "text": ...}, ...] 형식의 리스트
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    return data