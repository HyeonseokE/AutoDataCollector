import os
import time
from typing import Dict, List, Optional, Tuple

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Image
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, NotFound

# Vertex AI 설정 (환경변수로 override 가능)
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "prism-485101")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

MAX_RETRIES = 10
RETRY_DELAY = 60  # seconds (fixed interval)
GEMINI3_DEFAULT_THINKING_BUDGET = 10000  # Gemini 3 thinking 토큰 제한 (기본 10K)

_initialized = False
_initialized_location = None


def _make_gen_config(model: str, temperature: float = 0.0, **kwargs) -> GenerationConfig:
    """GenerationConfig 생성. Gemini 3 모델은 thinking budget 자동 적용."""
    gen_config = GenerationConfig(temperature=temperature, **kwargs)
    if "gemini-3" in model.lower():
        budget = GEMINI3_DEFAULT_THINKING_BUDGET
        gen_config._raw_generation_config.thinking_config.thinking_budget = budget
    return gen_config


def _get_location_for_model(model: str) -> str:
    """모델에 따라 적절한 Vertex AI location 반환.

    Gemini 3.0 preview 모델은 global endpoint만 지원.
    """
    if "gemini-3" in model.lower():
        return "global"
    return os.getenv("VERTEX_LOCATION", "us-central1")


def _ensure_init(location: str = None):
    """Vertex AI 초기화. location이 바뀌면 재초기화."""
    global _initialized, _initialized_location
    target_location = location or LOCATION
    if not _initialized or _initialized_location != target_location:
        vertexai.init(project=PROJECT_ID, location=target_location)
        _initialized = True
        _initialized_location = target_location


def _build_contents(turn: Dict) -> list:
    """턴 dict에서 Gemini API contents 리스트 구성.

    지원하는 키:
        text: 프롬프트 텍스트 (필수)
        image_path: 단일 이미지 경로 (옵션)
        image_paths: 복수 이미지 경로 리스트 (옵션, CAD 등)
    """
    contents = [turn["text"]]
    if turn.get("image_path"):
        contents.append(Part.from_image(Image.load_from_file(turn["image_path"])))
    for img_path in turn.get("image_paths", []):
        contents.append(Part.from_image(Image.load_from_file(img_path)))
    return contents


def _send_with_retry(chat, contents, generation_config,
                     max_retries=MAX_RETRIES):
    """Rate limit (429) / Service Unavailable (503) / NotFound (404, preview 모델 간헐적) 시 exponential backoff 재시도."""
    for attempt in range(max_retries + 1):
        try:
            return chat.send_message(contents,
                                     generation_config=generation_config)
        except (ResourceExhausted, ServiceUnavailable, NotFound) as e:
            if attempt == max_retries:
                raise
            delay = RETRY_DELAY
            if isinstance(e, ResourceExhausted):
                err_type = "Rate limit"
            elif isinstance(e, NotFound):
                err_type = "404 NotFound (preview model intermittent)"
            else:
                err_type = "503 Unavailable"
            print(f"  [{err_type}] Waiting {delay}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(delay)


# ============================================================
# Dynamic chat session API (신규)
# ============================================================

def gemini_chat_start(
    model: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    thinking_budget: int = None,
) -> Tuple:
    """Chat 세션 시작. (chat, gen_config) 튜플 반환.

    이후 gemini_chat_send()로 턴을 하나씩 전송할 수 있음.

    Args:
        model: Gemini 모델 이름
        system_prompt: 시스템 프롬프트
        temperature: 샘플링 온도
        thinking_budget: thinking 토큰 제한 (Gemini 3 전용).
            None이면 Gemini 3 모델은 기본 15000 적용, 그 외 모델은 미적용.

    Returns:
        (chat_session, generation_config) 튜플
    """
    location = _get_location_for_model(model)
    _ensure_init(location)

    gemini_model = GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )
    chat = gemini_model.start_chat()

    # Gemini 3: thinking_budget 파라미터 우선, 없으면 기본값 자동 적용
    if "gemini-3" in model.lower() and thinking_budget is not None:
        gen_config = _make_gen_config(model, temperature=temperature)
        gen_config._raw_generation_config.thinking_config.thinking_budget = thinking_budget
    else:
        gen_config = _make_gen_config(model, temperature=temperature)

    if "gemini-3" in model.lower():
        print(f"[GEMINI] Thinking budget: {gen_config._raw_generation_config.thinking_config.thinking_budget} tokens")

    return chat, gen_config


def gemini_chat_send(
    chat,
    gen_config,
    turn: Dict,
    check_time: bool = True,
    turn_label: str = None,
) -> str:
    """기존 chat 세션에 턴 1개 전송.

    Args:
        chat: gemini_chat_start()에서 반환된 chat 객체
        gen_config: gemini_chat_start()에서 반환된 GenerationConfig
        turn: {"text": str, "image_path": str|None, "image_paths": list|None}
        check_time: 시간 출력 여부
        turn_label: 로그에 표시할 턴 라벨 (예: "Turn 0", "Crop: male hinge")

    Returns:
        LLM 응답 텍스트
    """
    start_time = time.time()

    contents = _build_contents(turn)
    resp = _send_with_retry(chat, contents, gen_config)

    if check_time:
        elapsed = time.time() - start_time
        n_images = (1 if turn.get("image_path") else 0) + len(turn.get("image_paths", []))
        img_str = f" + {n_images} image(s)" if n_images > 0 else ""
        label = f" [{turn_label}]" if turn_label else ""
        # Token usage breakdown
        token_str = ""
        try:
            usage = resp.usage_metadata
            parts = []
            if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                parts.append(f"in={usage.prompt_token_count}")
            if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                parts.append(f"out={usage.candidates_token_count}")
            if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                parts.append(f"think={usage.thoughts_token_count}")
            elif hasattr(usage, 'thinking_token_count') and usage.thinking_token_count:
                parts.append(f"think={usage.thinking_token_count}")
            if hasattr(usage, 'total_token_count') and usage.total_token_count:
                parts.append(f"total={usage.total_token_count}")
            if parts:
                token_str = f" ({', '.join(parts)})"
        except Exception:
            pass
        print(f"[GEMINI/Chat]{label}{img_str}: {elapsed:.2f}s{token_str}")

    return resp.text


# ============================================================
# 기존 API (하위호환)
# ============================================================

def gemini_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    stop_sequences: list = None,
    check_time: bool = True,
    system_prompt: Optional[str] = None,
    image_path: Optional[str] = None,
) -> str:
    _ensure_init(_get_location_for_model(model))

    start_time = time.time()

    gemini_model = GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )

    # 멀티모달 입력 구성: 텍스트 + (옵션) 이미지
    contents = [prompt]
    if image_path:
        contents.append(Part.from_image(Image.load_from_file(image_path)))

    stop = [s for s in stop_sequences if s] or None if stop_sequences else None
    gen_config = _make_gen_config(model, temperature=temperature, stop_sequences=stop)

    response = gemini_model.generate_content(
        contents,
        generation_config=gen_config,
    )

    if check_time:
        elapsed = time.time() - start_time
        has_image = " + image" if image_path else ""
        has_system = " + system_prompt" if system_prompt else ""
        print(f"[GEMINI/VertexAI] Model: {model}{has_system}{has_image}, Response time: {elapsed:.2f}s")

    return response.text


def gemini_chat(
    model: str,
    system_prompt: str,
    turns: List[Dict],
    temperature: float = 0.0,
    check_time: bool = True,
) -> List[str]:
    """
    멀티턴 Gemini chat session

    Args:
        model: Gemini 모델 이름 (예: "gemini-2.0-flash")
        system_prompt: 시스템 프롬프트 (전체 session에 고정)
        turns: 턴 리스트, 각 턴은 {"text": str, "image_path": str|None}
        temperature: 샘플링 온도
        check_time: 시간 출력 여부

    Returns:
        각 턴별 LLM 응답 문자열 리스트
    """
    location = _get_location_for_model(model)
    _ensure_init(location)

    start_time = time.time()

    gemini_model = GenerativeModel(
        model,
        system_instruction=system_prompt if system_prompt else None,
    )
    chat = gemini_model.start_chat()

    generation_config = _make_gen_config(model, temperature=temperature)
    responses = []

    for i, turn in enumerate(turns):
        turn_start = time.time()

        contents = _build_contents(turn)
        resp = _send_with_retry(chat, contents, generation_config)
        responses.append(resp.text)

        if check_time:
            turn_elapsed = time.time() - turn_start
            n_images = (1 if turn.get("image_path") else 0) + len(turn.get("image_paths", []))
            img_str = f" + {n_images} image(s)" if n_images > 0 else ""
            print(f"[GEMINI/Chat] Turn {i+1}/{len(turns)}{img_str}: {turn_elapsed:.2f}s")

    if check_time:
        total_elapsed = time.time() - start_time
        print(f"[GEMINI/Chat] Total ({len(turns)} turns): {total_elapsed:.2f}s")

    return responses
