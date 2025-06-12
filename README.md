# Switch Transformer with Dynamic GPU Routing

이 프로젝트는 Switch Transformer 아키텍처를 기반으로 하여, 다중 GPU 환경에서 효율적인 Expert 분배와 동적 라우팅을 구현한 시스템입니다.

## 주요 기능

### 1. GPU 기반 Expert 분배
- 여러 GPU에 Expert를 효율적으로 분배
- 각 Expert의 특성과 GPU의 성능을 고려한 최적의 배치
- GPU 매핑 딕셔너리를 통한 Expert-GPU 매핑 관리

### 2. 동적 라우팅 시스템
- Latency-Aware Top1 Router 구현
- MLP 기반의 지연 시간 예측 모델
- 토큰별 최적의 GPU 선택을 위한 동적 라우팅

## 핵심 컴포넌트

### LatencyAwareTop1Router
- 토큰별 지연 시간을 고려한 라우팅 결정
- Expert 선택과 GPU 할당을 동시에 수행
- 라우팅 점수와 엔트로피를 고려한 의사결정

### LatencyMLP
- 지연 시간 예측을 위한 MLP 모델
- 8차원 입력 특성을 사용하여 지연 시간 예측
- 3개의 레이어로 구성된 신경망 구조

### SwitchTransformerLayer
- Expert 모듈과 라우터를 포함한 기본 레이어
- 토큰별 Expert 할당 및 처리
- GPU 간 효율적인 데이터 이동 관리

## 사용된 주요 기술
- PyTorch
- CUDA
- MLP 기반 지연 시간 예측
- 동적 라우팅 알고리즘

## 프로젝트 구조
```
.
├── switch_trasnformer.py      # Switch Transformer 구현
├── latency_aware_top1_router.py  # 동적 라우터 구현
├── multi_gpu/                 # 다중 GPU 관련 코드
├── dynamic_routing/          # 동적 라우팅 관련 코드
└── utils/                    # 유틸리티 함수
