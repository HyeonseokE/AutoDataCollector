# Server LLM Implementation Plan

SSH 서버의 vLLM을 활용한 LLM 원격 추론 시스템 구축 계획

---

## 1. 개요

### 1.1 목표

OpenAI API 호출을 **vLLM 서버 호출로 대체**하여 비용 절감

| 항목 | 기존 | 변경 |
|------|------|------|
| LLM | OpenAI API (유료) | vLLM + 오픈소스 모델 (무료) |
| 코드 변경 | - | `base_url` 1줄 추가 |

### 1.2 핵심 아이디어

vLLM은 **OpenAI API와 100% 호환되는 HTTP API**를 제공합니다.
→ 기존 코드에서 `base_url`만 변경하면 됩니다.

```python
# 기존 (OpenAI API - 유료)
client = OpenAI(api_key="sk-xxx")

# 변경 (vLLM - 무료)
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
```

---

## 2. 아키텍처

### 2.1 단순화된 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         vLLM 기반 LLM 서버                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [로컬 우분투 PC]                          [SSH 서버 (RTX 3090)]          │
│  (로봇 제어)                               (GPU 추론)                     │
│                                                                         │
│  ┌─────────────────────┐                 ┌─────────────────────┐        │
│  │                     │                 │                     │        │
│  │  파이프라인 실행     │    SSH 터널     │  vLLM Server        │        │
│  │                     │   :8001 ───►    │                     │        │
│  │  ┌───────────────┐  │                 │  ┌───────────────┐  │        │
│  │  │ OpenAI Client │  │ ─────────────►  │  │ 7B Coding LLM │  │        │
│  │  │ (base_url=    │  │                 │  │               │  │        │
│  │  │  localhost:   │  │ ◄─────────────  │  │ Qwen2.5-Coder │  │        │
│  │  │  8001)        │  │   생성된 코드    │  │ -7B-Instruct  │  │        │
│  │  └───────────────┘  │                 │  └───────────────┘  │        │
│  │                     │                 │                     │        │
│  │  로봇 코드 실행      │                 │  :8001              │        │
│  │                     │                 │                     │        │
│  └─────────────────────┘                 └─────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 vLLM의 역할

| 역할 | 설명 |
|------|------|
| 모델 로딩 | HuggingFace에서 모델 다운로드 후 GPU 메모리에 로드 |
| 토큰화 | 텍스트 프롬프트 → 토큰 ID 변환 |
| GPU 추론 | PagedAttention으로 효율적인 추론 수행 |
| 디토큰화 | 토큰 ID → 텍스트 변환 |
| API 제공 | OpenAI-compatible HTTP API (`/v1/chat/completions`) |

### 2.3 불필요한 것들

| 항목 | 필요 여부 | 이유 |
|------|----------|------|
| gRPC 서버 (llm_server.py) | ❌ 불필요 | vLLM이 직접 HTTP API 제공 |
| 별도 클라이언트 (client.py) | ❌ 불필요 | 기존 OpenAI 클라이언트 사용 |
| 프롬프트 전송 코드 | ❌ 불필요 | 기존 코드 그대로 사용 |

---

## 3. 구현 방법

### 3.1 서버 측 (SSH 서버, RTX 3090)

#### Step 1: vLLM 설치

```bash
# 가상환경 생성 (권장)
conda create -n vllm python=3.11
conda activate vllm

# vLLM 설치
pip install vllm
```

#### Step 2: vLLM 서버 실행

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --host 0.0.0.0 \
    --port 8001
```

- 최초 실행 시 HuggingFace에서 모델 자동 다운로드 (~14GB)
- 이후 캐시되어 빠르게 로드

#### Step 3: 서버 실행 확인

```bash
# 테스트 요청
curl http://localhost:8001/v1/models
```

### 3.2 로컬 측 (우분투 로봇 PC)

#### Step 1: SSH 터널 설정

```bash
# 터미널 1: SSH 터널 (백그라운드 실행)
ssh -L 8001:localhost:8001 -N user@server-address &
```

또는 SSH config 설정:

```
# ~/.ssh/config
Host llm-server
    HostName server-address
    User hscho
    LocalForward 8001 localhost:8001
```

```bash
ssh llm-server
```

#### Step 2: 코드 변경

```python
# code_gen_lerobot/llm_utils/openai_utils.py

# 기존
openai_client = OpenAI(api_key=_api_keys.get("openai_api_key"))

# 변경 (USE_LLM_SERVER 환경변수로 제어)
import os

if os.getenv("USE_LLM_SERVER", "").lower() in ("1", "true"):
    openai_client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )
else:
    openai_client = OpenAI(api_key=_api_keys.get("openai_api_key"))
```

#### Step 3: 파이프라인 실행

```bash
# 서버 모드로 실행
export USE_LLM_SERVER=1
./run_forward_and_reset.sh
```

---

## 4. 선택 가능한 모델

### 4.1 Coding LLM 모델 목록 (RTX 3090 24GB 기준)

| 모델 | 크기 | VRAM | 특징 | 추천 |
|------|------|------|------|------|
| **Qwen/Qwen2.5-Coder-7B-Instruct** | 7B | ~16GB | 코드 생성 SOTA급, 한국어 지원 | ⭐⭐⭐ |
| **deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct** | 16B (MoE) | ~14GB | MoE 구조로 효율적 | ⭐⭐⭐ |
| meta-llama/Llama-3.1-8B-Instruct | 8B | ~18GB | 범용, 안정적 | ⭐⭐ |
| codellama/CodeLlama-7b-Instruct-hf | 7B | ~16GB | Meta 코드 특화 (구버전) | ⭐ |

### 4.2 선택된 모델

**✅ Qwen/Qwen2.5-Coder-7B-Instruct**

| 항목 | 값 |
|------|------|
| 모델명 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 크기 | 7B 파라미터 |
| VRAM | ~16GB (RTX 3090에서 여유있게 실행) |
| 선택 이유 | 코드 생성 성능 SOTA급, 한국어 지원, 최신 모델 |

---

## 5. 통신 흐름

```
[로컬 우분투 PC]                              [SSH 서버]
      │                                           │
      │  1. SSH 터널 연결                          │
      │  (localhost:8001 → server:8001)           │
      ├──────────────────────────────────────────►│
      │                                           │
      │  2. 파이프라인 실행                         │
      │  (run_forward_and_reset.sh)               │
      │                                           │
      │  3. LLM 코드 생성 요청                      │
      │  POST /v1/chat/completions                │
      │  {                                        │
      │    "model": "Qwen2.5-Coder-7B",           │
      │    "messages": [{"role": "user",          │
      │                  "content": prompt}]      │
      │  }                                        │
      ├──────────────────────────────────────────►│
      │                                           │  4. vLLM 추론
      │                                           │  (GPU에서 코드 생성)
      │  5. 응답 수신                              │
      │  {                                        │
      │    "choices": [{                          │
      │      "message": {"content": code}         │
      │    }]                                     │
      │  }                                        │
      │◄──────────────────────────────────────────┤
      │                                           │
      │  6. 생성된 코드로 로봇 실행                  │
      ▼                                           │
```

---

## 6. 체크리스트

### Phase 1: 서버 구축 ✅

- [x] vLLM 설치 (`pip install vllm`)
- [x] Qwen2.5-Coder-7B-Instruct 모델 다운로드 (자동)
- [x] vLLM 서버 실행 테스트
- [x] curl로 API 호출 테스트

### Phase 2: 로컬 연동 ✅

- [x] SSH 터널 설정 (문서화 완료)
- [x] `llm.py` 코드 수정 (`_call_vllm_server()` 함수 추가)
- [x] `USE_LLM_SERVER=1`로 파이프라인 테스트

### Phase 3: 정리 ✅

- [x] 서버 실행 스크립트 작성 (`run_vllm_server.py`, `run_vllm_server.sh`)
- [x] 불필요한 gRPC 코드 삭제 (client.py, llm_server.py, vlm_server.py, proto/)

---

## 7. 참고

### 7.1 vLLM 서버 옵션

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 4096 \        # 최대 컨텍스트 길이
    --gpu-memory-utilization 0.9  # GPU 메모리 사용률
```

### 7.2 현재 서버 GPU 정보

```
GPU: NVIDIA GeForce RTX 3090
VRAM: 24GB
CUDA: 13.0
→ 7B 모델 여유있게 실행 가능
```

### 7.3 현재 server_inference 디렉토리 구조

```
server_inference/
├── __init__.py                    # 패키지 초기화
├── run_vllm_server.py            # vLLM 서버 실행 (Python)
└── scripts/
    └── run_vllm_server.sh        # vLLM 서버 실행 (Shell)
```

**삭제된 파일 (2025-12-29):**
- `llm_server.py`, `vlm_server.py`, `client.py` - gRPC 서버/클라이언트
- `config.py`, `server_configs.py` - 설정 파일
- `proto/` - gRPC 프로토콜 정의
