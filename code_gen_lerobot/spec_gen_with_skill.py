#!/usr/bin/env python3
"""
LeRobot SO-101 Specification Generation Module

자연어 목표를 구조화된 스펙(단계별 primitive 액션)으로 변환합니다.
"""

from .llm import llm_response
from .forward_execution.prompt import lerobot_spec_gen_prompt

import json
import os
from typing import Dict, List, Optional


def lerobot_spec_gen(
    instruction: str,
    object_queries: List[str] = None,
    object_positions: Dict[str, List[float]] = None,
    use_detection: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """
    LeRobot SO-101용 태스크 스펙 생성

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_queries: 디텍션 ON시 찾을 객체 리스트
        object_positions: 디텍션 OFF시 직접 전달할 위치 딕셔너리
        use_detection: True면 object_detection으로 위치 획득
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")

    Returns:
        스펙 딕셔너리:
        {
            "required_skills": ["skill1", "skill2", ...],
            "steps": [
                {"step": 1, "action": "skill_name", ...},
                ...
            ]
        }

    Example:
        spec = lerobot_spec_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_queries=["red cup", "blue box"],
        )
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Spec Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 1) 객체 위치 획득
    if use_detection:
        if object_queries is None or len(object_queries) == 0:
            raise ValueError("use_detection=True requires object_queries (non-empty list)")

        print(f"{YELLOW}[1/3] Detecting objects: {object_queries}{RESET}")

        try:
            from run_detect import run_realtime_detection
            positions = run_realtime_detection(
                queries=object_queries,
                timeout=10.0,
                unit="m",
                visualize=False,
            )
        except ImportError as e:
            print(f"[Error] Could not import object_positions module: {e}")
            positions = {q: None for q in object_queries}
        except Exception as e:
            print(f"[Error] Detection failed: {e}")
            positions = {q: None for q in object_queries}

    else:
        if object_positions is None or len(object_positions) == 0:
            raise ValueError("use_detection=False requires object_positions (non-empty dict)")

        print(f"{YELLOW}[1/3] Using provided positions{RESET}")
        positions = object_positions

    # 검출 결과 출력
    found_count = sum(1 for v in positions.values() if v is not None)
    print(f"  Found: {found_count}/{len(positions)} objects")

    # 2) 프롬프트 생성
    print(f"\n{YELLOW}[2/3] Generating prompt{RESET}")
    prompt = lerobot_spec_gen_prompt(
        instruction=instruction,
        object_positions=positions,
    )

    # 3) LLM 호출
    print(f"\n{YELLOW}[3/3] Calling LLM ({llm_model}){RESET}")
    response = llm_response(llm_model, prompt, check_time=True)

    # 4) 스펙 파싱
    spec = parse_spec_from_response(response)

    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + "Spec generation completed.".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    return spec


def parse_spec_from_response(response: str) -> dict:
    """
    LLM 응답에서 스펙 JSON을 파싱

    Args:
        response: LLM 응답 문자열

    Returns:
        파싱된 스펙 딕셔너리
    """
    import re

    # "Specification: {...}" 패턴 찾기
    spec_pattern = r'Specification:\s*(\{.*\})'
    match = re.search(spec_pattern, response, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # JSON 블록 찾기
    json_pattern = r'\{[^{}]*"required_skills"[^{}]*"steps"[^{}]*\[.*?\]\s*\}'
    match = re.search(json_pattern, response, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 전체에서 JSON 추출 시도
    try:
        # { 와 } 사이의 가장 큰 부분 찾기
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(response[start:end+1])
    except json.JSONDecodeError:
        pass

    # 파싱 실패시 원본 반환
    return {"raw_response": response}


def lerobot_spec_gen_and_save(
    instruction: str,
    object_positions: Dict[str, List[float]],
    save_path: str,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """
    스펙 생성 후 파일로 저장

    Args:
        instruction: 자연어 목표
        object_positions: 객체 위치 딕셔너리
        save_path: 저장 경로
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")

    Returns:
        생성된 스펙 딕셔너리
    """
    spec = lerobot_spec_gen(
        instruction=instruction,
        object_positions=object_positions,
        use_detection=False,
        llm_model=llm_model,
    )

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # JSON 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print(f"[Saved] Spec saved to: {save_path}")

    return spec
