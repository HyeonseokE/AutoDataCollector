"""
Forward Execution Spec Generator for Multi-Robot

Generates structured task specifications for forward execution.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_gen_lerobot.llm import llm_response
from .prompt import lerobot_spec_gen_prompt_multi


def single_robot_spec_gen(
    instruction: str,
    object_positions: Dict,
    robot_id: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> Dict:
    """
    단일 로봇용 스펙 생성

    Args:
        instruction: 자연어 목표
        object_positions: 객체별 위치 (로봇 프레임 기준)
        robot_id: 로봇 번호
        llm_model: LLM 모델

    Returns:
        Dict: 생성된 스펙 (required_skills, steps 포함)
    """

    # ANSI 색상 코드
    YELLOW = "\033[93m"
    LIGHT_GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"{YELLOW}[Robot {robot_id}] Generating spec{RESET}")

    # 프롬프트 생성
    prompt = lerobot_spec_gen_prompt_multi(
        instruction=instruction,
        object_positions=object_positions,
        robot_id=robot_id,
    )

    # LLM 호출
    response = llm_response(llm_model, prompt, check_time=True)

    # 스펙 파싱
    spec = parse_spec_from_response(response)
    spec["robot_id"] = robot_id

    print(f"{LIGHT_GREEN}[Robot {robot_id}] Spec generation completed.{RESET}")

    return spec


def multi_robot_spec_gen(
    instruction: str,
    robot_positions: Dict[int, Dict],
    robot_ids: List[int] = None,
    llm_model: str = "gpt-4o-mini",
    parallel: bool = True,
) -> Dict[int, Dict]:
    """
    멀티 로봇용 스펙 생성 (병렬 처리)

    Args:
        instruction: 자연어 목표
        robot_positions: 로봇별 객체 위치
        robot_ids: 로봇 ID 목록
        llm_model: LLM 모델
        parallel: 병렬 처리 여부

    Returns:
        Dict[int, Dict]: {robot_id: spec}
    """

    # ANSI 색상 코드
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    line_width = 70

    if robot_ids is None:
        robot_ids = list(robot_positions.keys())

    print(CYAN + "=" * line_width + RESET)
    print(CYAN + BOLD + "Multi-Robot Spec Generation".center(line_width) + RESET)
    print(CYAN + "=" * line_width + RESET)

    results = {}

    def generate_for_robot(robot_id: int) -> Tuple[int, Dict]:
        positions = robot_positions.get(robot_id, {})
        spec = single_robot_spec_gen(
            instruction=instruction,
            object_positions=positions,
            robot_id=robot_id,
            llm_model=llm_model,
        )
        return robot_id, spec

    if parallel and len(robot_ids) > 1:
        with ThreadPoolExecutor(max_workers=len(robot_ids)) as executor:
            futures = {
                executor.submit(generate_for_robot, rid): rid
                for rid in robot_ids
            }

            for future in as_completed(futures):
                robot_id = futures[future]
                try:
                    rid, spec = future.result()
                    results[rid] = spec
                except Exception as e:
                    print(f"[Robot {robot_id}] Spec generation failed: {e}")
                    results[robot_id] = {"error": str(e)}
    else:
        for robot_id in robot_ids:
            try:
                rid, spec = generate_for_robot(robot_id)
                results[rid] = spec
            except Exception as e:
                print(f"[Robot {robot_id}] Spec generation failed: {e}")
                results[robot_id] = {"error": str(e)}

    print(CYAN + "=" * line_width + RESET)

    return results


def parse_spec_from_response(response: str) -> Dict:
    """
    LLM 응답에서 스펙(JSON)을 파싱

    Args:
        response: LLM 응답 문자열

    Returns:
        Dict: 파싱된 스펙
    """
    if not response:
        return {"required_skills": [], "steps": []}

    # JSON 블록 찾기
    json_pattern = r'\{[\s\S]*?"required_skills"[\s\S]*?"steps"[\s\S]*?\}'

    # Specification: { ... } 패턴
    spec_pattern = r'Specification:\s*(\{[\s\S]*\})'
    spec_match = re.search(spec_pattern, response)

    if spec_match:
        try:
            return json.loads(spec_match.group(1))
        except json.JSONDecodeError:
            pass

    # 일반 JSON 패턴
    matches = re.findall(json_pattern, response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # 코드 블록 내 JSON
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    code_matches = re.findall(code_block_pattern, response)
    for match in code_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # 파싱 실패 시 기본값
    return {"required_skills": [], "steps": [], "parse_error": True, "raw_response": response[:500]}
