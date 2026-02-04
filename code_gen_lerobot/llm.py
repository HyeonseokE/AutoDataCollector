"""
LLM Response Module

모델 이름을 기반으로 자동으로 provider를 감지하여 응답을 생성합니다.

지원 모델:
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o1-pro 등
- Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash 등
- DeepSeek: deepseek-chat, deepseek-reasoner 등
- Llama: llama-3.1-70b, llama-3.1-8b 등

vLLM 서버 추론 모드:
- use_server=True로 설정하거나 환경변수 USE_LLM_SERVER=1 설정
- 환경변수 VLLM_SERVER_URL로 서버 주소 설정 가능 (기본: http://localhost:8001/v1)
- 환경변수 VLLM_MODEL_NAME로 모델명 설정 가능 (기본: Qwen/Qwen2.5-Coder-7B-Instruct)

클라우드 서버 추론 모드:
- 환경변수 USE_CLOUD_SERVER=1 설정
- 환경변수 CLOUD_SERVER_URL로 서버 주소 설정 (예: http://115.71.36.55/deployment/xxx)
"""

import os
import time
from typing import Optional


# vLLM 서버 설정 (환경변수로 override 가능)
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8001/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")


def _should_use_server(use_server: bool = None) -> bool:
    """서버 사용 여부 결정 (파라미터 > 환경변수)"""
    if use_server is not None:
        return use_server
    return os.getenv("USE_LLM_SERVER", "").lower() in ("1", "true", "yes")


def _should_use_cloud_server() -> bool:
    """클라우드 서버 사용 여부 결정 (환경변수)"""
    return os.getenv("USE_CLOUD_SERVER", "").lower() in ("1", "true", "yes")


def _call_cloud_server(
    prompt: str,
    check_time: bool = True,
) -> Optional[str]:
    """
    클라우드 서버 호출

    Args:
        prompt: 프롬프트
        check_time: 시간 출력 여부

    Returns:
        생성된 텍스트 또는 실패 시 None
    """
    try:
        from cloud_inference2 import cloud_llm_response
        return cloud_llm_response(prompt=prompt, check_time=check_time)
    except ImportError as e:
        print(f"[Cloud] Error: cloud_inference2 module not found: {e}")
        return None
    except Exception as e:
        print(f"[Cloud] Error: {e}")
        return None


def _call_vllm_server(
    prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
) -> Optional[str]:
    """
    vLLM 서버 호출 (OpenAI-compatible API)

    Args:
        prompt: 프롬프트
        max_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        stop_sequences: 정지 시퀀스
        check_time: 시간 출력 여부

    Returns:
        생성된 텍스트 또는 실패 시 None
    """
    from openai import OpenAI
    import httpx

    # 함수 호출 시점에 환경변수 읽기 (모듈 로드 시점이 아닌)
    server_url = os.getenv("VLLM_SERVER_URL", "http://localhost:8001/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")

    # vLLM 서버는 모델 로딩/긴 프롬프트 처리에 시간이 걸릴 수 있음
    client = OpenAI(
        base_url=server_url,
        api_key="not-needed",  # vLLM 로컬 서버는 API 키 불필요
        timeout=httpx.Timeout(120.0, connect=10.0),  # 응답 120초, 연결 10초
    )

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences if stop_sequences else None,
        )

        if check_time:
            elapsed = time.time() - start_time
            print(f"[VLLM] Model: {model_name}, Response time: {elapsed:.2f}s")

        return response.choices[0].message.content

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[VLLM] Error ({elapsed:.2f}s): {e}")
        return None


def detect_provider(model: str) -> str:
    """
    모델 이름으로 provider 자동 감지

    Args:
        model: 모델 이름 (예: "gpt-4o-mini", "gemini-1.5-flash")

    Returns:
        provider 이름 ("openai", "gemini", "deepseek", "llama")
    """
    model_lower = model.lower()

    # OpenAI 모델
    if any(prefix in model_lower for prefix in ["gpt-", "o1", "o3"]):
        return "openai"

    # Gemini 모델
    if "gemini" in model_lower:
        return "gemini"

    # DeepSeek 모델
    if "deepseek" in model_lower:
        return "deepseek"

    # Llama 모델
    if "llama" in model_lower:
        return "llama"

    # Claude 모델 (Anthropic)
    if "claude" in model_lower:
        return "anthropic"

    # 기본값: OpenAI로 시도
    return "openai"


def llm_response(
    model: str,
    prompt: str,
    stop_sequences: list = None,
    check_time: bool = True,
    use_server: bool = None,
) -> Optional[str]:
    """
    LLM 응답 생성

    Args:
        model: 모델 이름 (예: "gpt-4o-mini", "gemini-1.5-flash")
        prompt: 프롬프트
        stop_sequences: 정지 시퀀스
        check_time: 시간 출력 여부
        use_server: True면 SSH 서버로 추론, None이면 환경변수 USE_LLM_SERVER 확인

    Returns:
        LLM 응답 문자열
    """
    # 클라우드 서버 사용 여부 결정 (최우선)
    if _should_use_cloud_server():
        return _call_cloud_server(
            prompt=prompt,
            check_time=check_time,
        )

    # vLLM 서버 사용 여부 결정
    if _should_use_server(use_server):
        return _call_vllm_server(
            prompt=prompt,
            stop_sequences=stop_sequences,
            check_time=check_time,
        )

    # 기존 API 호출
    provider = detect_provider(model)
    stop_sequences = stop_sequences or ['']

    if provider == "openai":
        from .llm_utils.openai_utils import chatgpt_response
        return chatgpt_response(
            prompt=prompt,
            model=model,
            stop_sequences=stop_sequences,
            check_time=check_time,
            source="openai",
        )

    elif provider == "deepseek":
        from .llm_utils.openai_utils import chatgpt_response
        return chatgpt_response(
            prompt=prompt,
            model=model,
            stop_sequences=stop_sequences,
            check_time=check_time,
            source="deepseek",
        )

    elif provider == "gemini":
        from .llm_utils.gemini import gemini_response
        return gemini_response(
            prompt=prompt,
            model=model,
            stop_sequences=stop_sequences,
            check_time=check_time,
        )

    elif provider == "llama":
        from .llm_utils.llama import llama_response
        return llama_response(
            prompt=prompt,
            model=model,
            stop_sequences=stop_sequences,
            check_time=check_time,
        )

    else:
        raise ValueError(f"지원하지 않는 모델: {model} (provider: {provider})")
