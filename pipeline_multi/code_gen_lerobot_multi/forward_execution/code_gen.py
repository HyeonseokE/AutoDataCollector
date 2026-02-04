"""
Forward Execution Code Generator for Multi-Robot

Generates executable Python code for forward task execution
with support for multiple robots operating in parallel.
"""

import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_gen_lerobot.llm import llm_response
from ..prompt import ForwardTemplatePrompt, TowelFoldingPrompt, PickAndPlacePrompt, StackingPrompt, DoorHingeAssemblyPrompt

# Task prompts mapping
_TASK_PROMPTS = {
    "towel_folding": TowelFoldingPrompt(),
    "pick_and_place": PickAndPlacePrompt(),
    "stacking": StackingPrompt(),
    "door_hinge_assembly": DoorHingeAssemblyPrompt(),
}


def _get_task_prompt(instruction: str = None, task_type: str = None):
    """Get matching task prompt based on instruction or task_type"""
    # Explicit task_type
    if task_type and task_type in _TASK_PROMPTS:
        return _TASK_PROMPTS[task_type]

    # Match by instruction keywords
    if instruction:
        for task in _TASK_PROMPTS.values():
            if task.matches(instruction):
                return task

    return None


def single_robot_code_gen(
    instruction: str,
    object_positions: Dict,
    robot_id: int = 3,
    llm_model: str = "gpt-4o-mini",
    spec: str = None,
    task_type: str = None,
    assigned_object: str = None,
    assigned_role: str = None,
) -> Tuple[str, Dict]:
    """
    단일 로봇용 코드 생성 (멀티 로봇 환경에서 호출됨)

    Args:
        instruction: 자연어 목표
        object_positions: 객체별 위치 (로봇 프레임 기준, 이미 변환됨)
        robot_id: 로봇 번호 (2 또는 3)
        llm_model: LLM 모델
        spec: 코드 생성 가이드라인/스펙
        task_type: 태스크 유형 ("towel_folding" 등)
        assigned_object: 이 로봇에게 할당된 물체 이름 (거리 기반 자동 할당)
        assigned_role: 이 로봇의 역할 (예: "holder", "inserter")

    Returns:
        Tuple[str, Dict]: (생성된 코드, 객체 위치)
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    print(f"{CYAN}[Robot {robot_id}] Code Generation (Forward){RESET}")
    print(GRAY + "-" * line_width + RESET)

    # 검출 결과 출력
    found_count = sum(1 for v in object_positions.values() if v is not None)
    print(f"  Found: {found_count}/{len(object_positions)} objects")
    for name, info in object_positions.items():
        if info:
            pos = info.get("position", info) if isinstance(info, dict) else info
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        else:
            print(f"    x {name}: NOT FOUND")

    # 검출 결과 검증
    not_found = [name for name, info in object_positions.items() if info is None]
    if not_found:
        print(f"\n{YELLOW}[Robot {robot_id}] ERROR: Required objects not detected!{RESET}")
        print(f"  Missing objects: {not_found}")
        raise ValueError(f"Robot {robot_id}: Cannot generate code - objects not detected: {not_found}")

    # 태스크 프롬프트 매칭
    task_prompt = _get_task_prompt(instruction=instruction, task_type=task_type)
    if task_prompt:
        print(f"  Task type: {task_prompt.name} (context added)")

    # 프롬프트 생성
    print(f"\n{YELLOW}[Robot {robot_id}] Generating prompt{RESET}")
    if assigned_object:
        print(f"  Assigned object: '{assigned_object}'")
    if assigned_role:
        print(f"  Assigned role: '{assigned_role}'")
    prompt = ForwardTemplatePrompt.generate(
        instruction=instruction,
        object_positions=object_positions,
        robot_id=robot_id,
        task_prompt=task_prompt,
        spec=spec,
        assigned_object=assigned_object,
        assigned_role=assigned_role,
    )

    # LLM 호출
    print(f"{YELLOW}[Robot {robot_id}] Calling LLM ({llm_model}){RESET}")
    response = llm_response(llm_model, prompt, check_time=True)

    # 코드 추출
    code = extract_code_from_response(response)

    print(f"{LIGHT_GREEN}[Robot {robot_id}] Code generation completed.{RESET}")
    print(GRAY + "-" * line_width + RESET)

    return code, object_positions


def multi_robot_code_gen(
    instruction: str,
    robot_positions: Dict[int, Dict],
    robot_ids: List[int] = None,
    llm_model: str = "gpt-4o-mini",
    spec: str = None,
    task_type: str = None,
    parallel: bool = True,
) -> Dict[int, Tuple[str, Dict]]:
    """
    멀티 로봇용 코드 생성 (병렬 처리)

    여러 로봇에 대해 동시에 코드를 생성합니다.
    각 로봇은 자신의 프레임으로 변환된 객체 위치를 받습니다.

    Args:
        instruction: 자연어 목표 (모든 로봇 공통)
        robot_positions: 로봇별 객체 위치 {robot_id: {object_name: position_info}}
        robot_ids: 로봇 ID 목록 (None이면 robot_positions의 키 사용)
        llm_model: LLM 모델
        spec: 코드 생성 가이드라인/스펙
        task_type: 태스크 유형 ("towel_folding" 등)
        parallel: True면 병렬로 코드 생성

    Returns:
        Dict[int, Tuple[str, Dict]]: {robot_id: (code, positions)}
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    line_width = 70

    if robot_ids is None:
        robot_ids = list(robot_positions.keys())

    print(CYAN + "=" * line_width + RESET)
    print(CYAN + BOLD + "Multi-Robot Code Generation (Forward)".center(line_width) + RESET)
    print(CYAN + "=" * line_width + RESET)
    print(f"  Instruction: {instruction}")
    print(f"  Robots: {robot_ids}")
    print(f"  LLM Model: {llm_model}")
    print(f"  Parallel: {parallel}")
    print()

    results = {}

    def generate_for_robot(robot_id: int) -> Tuple[int, str, Dict]:
        """단일 로봇용 코드 생성 (스레드 작업)"""
        positions = robot_positions.get(robot_id, {})
        code, pos = single_robot_code_gen(
            instruction=instruction,
            object_positions=positions,
            robot_id=robot_id,
            llm_model=llm_model,
            spec=spec,
            task_type=task_type,
        )
        return robot_id, code, pos

    if parallel and len(robot_ids) > 1:
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=len(robot_ids)) as executor:
            futures = {
                executor.submit(generate_for_robot, rid): rid
                for rid in robot_ids
            }

            for future in as_completed(futures):
                robot_id = futures[future]
                try:
                    rid, code, pos = future.result()
                    results[rid] = (code, pos)
                except Exception as e:
                    print(f"[Robot {robot_id}] Code generation failed: {e}")
                    results[robot_id] = (None, {})
    else:
        # 순차 처리
        for robot_id in robot_ids:
            try:
                rid, code, pos = generate_for_robot(robot_id)
                results[rid] = (code, pos)
            except Exception as e:
                print(f"[Robot {robot_id}] Code generation failed: {e}")
                results[robot_id] = (None, {})

    print()
    print(CYAN + "=" * line_width + RESET)
    success_count = sum(1 for code, _ in results.values() if code is not None)
    print(f"  Code generation complete: {success_count}/{len(robot_ids)} robots")
    print(CYAN + "=" * line_width + RESET)

    return results


def extract_code_from_response(response: str) -> str:
    """
    LLM 응답에서 Python 코드 블록을 추출

    Args:
        response: LLM 응답 문자열

    Returns:
        추출된 Python 코드
    """
    if not response:
        return ""

    # 코드 블록 패턴 (```python ... ``` 또는 ``` ... ```)
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # 가장 긴 코드 블록 반환 (보통 메인 코드)
        return max(matches, key=len).strip()

    # 코드 블록이 없으면 전체 응답 반환
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute_task', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    return response.strip()
