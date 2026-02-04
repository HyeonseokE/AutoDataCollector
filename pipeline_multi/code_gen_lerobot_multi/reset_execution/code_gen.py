"""
Reset Execution Code Generator for Multi-Robot

Generates executable Python code for reset task execution
with support for multiple robots operating in parallel.
"""

import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from code_gen_lerobot.llm import llm_response
from ..prompt import ResetTemplatePrompt, TowelFoldingPrompt, PickAndPlacePrompt, StackingPrompt, DoorHingeAssemblyPrompt

# Task prompts mapping
_TASK_PROMPTS = {
    "towel_folding": TowelFoldingPrompt(),
    "pick_and_place": PickAndPlacePrompt(),
    "stacking": StackingPrompt(),
    "door_hinge_assembly": DoorHingeAssemblyPrompt(),
}


def _get_task_prompt(instruction: str = None, task_type: str = None):
    """Get matching task prompt based on instruction or task_type"""
    if task_type and task_type in _TASK_PROMPTS:
        return _TASK_PROMPTS[task_type]

    if instruction:
        for task in _TASK_PROMPTS.values():
            if task.matches(instruction):
                return task

    return None


def single_robot_reset_code_gen(
    original_instruction: str,
    target_positions: Dict,
    current_positions: Dict,
    robot_id: int = 3,
    llm_model: str = "gpt-4o-mini",
    forward_spec: Dict = None,
    forward_code: str = None,
    is_random_reset: bool = False,
    task_type: str = None,
) -> Tuple[str, Dict, Dict, Dict]:
    """
    단일 로봇용 리셋 코드 생성

    Args:
        original_instruction: 원래 태스크 목표
        target_positions: 목표 위치 (original 또는 random)
        current_positions: 현재 위치 (detection 결과)
        robot_id: 로봇 번호
        llm_model: LLM 모델
        forward_spec: Forward spec
        forward_code: Forward code
        is_random_reset: 랜덤 리셋 모드
        task_type: 태스크 유형

    Returns:
        Tuple[str, Dict, Dict, Dict]: (reset_code, original_positions, current_positions, target_positions)
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    mode_str = "Random Reset" if is_random_reset else "Original Reset"
    print(f"{CYAN}[Robot {robot_id}] Code Generation ({mode_str}){RESET}")
    print(GRAY + "-" * line_width + RESET)

    # 위치 정보 출력
    print(f"  Target positions:")
    for name, info in target_positions.items():
        if info:
            pos = info.get("position", info) if isinstance(info, dict) else info
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    print(f"  Current positions:")
    for name, info in current_positions.items():
        if info:
            pos = info.get("position", info) if isinstance(info, dict) else info
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # 태스크 프롬프트 매칭
    task_prompt = _get_task_prompt(instruction=original_instruction, task_type=task_type)
    if task_prompt:
        print(f"  Task type: {task_prompt.name} (reset context added)")

    # 프롬프트 생성
    print(f"\n{YELLOW}[Robot {robot_id}] Generating reset prompt{RESET}")
    prompt = ResetTemplatePrompt.generate(
        original_instruction=original_instruction,
        target_positions=target_positions,
        current_positions=current_positions,
        robot_id=robot_id,
        task_prompt=task_prompt,
        forward_spec=forward_spec,
        forward_code=forward_code,
        is_random_reset=is_random_reset,
    )

    # LLM 호출
    print(f"{YELLOW}[Robot {robot_id}] Calling LLM ({llm_model}){RESET}")
    response = llm_response(llm_model, prompt, check_time=True)

    # 코드 추출
    code = extract_code_from_response(response)

    print(f"{LIGHT_GREEN}[Robot {robot_id}] Reset code generation completed.{RESET}")
    print(GRAY + "-" * line_width + RESET)

    # original_positions는 forward 단계의 초기 위치
    # target_positions는 리셋 목표 위치 (original 모드면 original_positions와 동일)
    return code, target_positions, current_positions, target_positions


def multi_robot_reset_code_gen(
    original_instruction: str,
    robot_target_positions: Dict[int, Dict],
    robot_current_positions: Dict[int, Dict],
    robot_ids: List[int] = None,
    llm_model: str = "gpt-4o-mini",
    forward_specs: Dict[int, Dict] = None,
    forward_codes: Dict[int, str] = None,
    is_random_reset: bool = False,
    task_type: str = None,
    parallel: bool = True,
) -> Dict[int, Tuple[str, Dict, Dict, Dict]]:
    """
    멀티 로봇용 리셋 코드 생성 (병렬 처리)

    Args:
        original_instruction: 원래 태스크 목표
        robot_target_positions: 로봇별 목표 위치
        robot_current_positions: 로봇별 현재 위치
        robot_ids: 로봇 ID 목록
        llm_model: LLM 모델
        forward_specs: 로봇별 forward spec
        forward_codes: 로봇별 forward code
        is_random_reset: 랜덤 리셋 모드
        task_type: 태스크 유형
        parallel: 병렬 처리 여부

    Returns:
        Dict[int, Tuple]: {robot_id: (code, original_pos, current_pos, target_pos)}
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    line_width = 70

    if robot_ids is None:
        robot_ids = list(robot_target_positions.keys())

    if forward_specs is None:
        forward_specs = {}
    if forward_codes is None:
        forward_codes = {}

    mode_str = "Random Reset" if is_random_reset else "Original Reset"
    print(CYAN + "=" * line_width + RESET)
    print(CYAN + BOLD + f"Multi-Robot Reset Code Generation ({mode_str})".center(line_width) + RESET)
    print(CYAN + "=" * line_width + RESET)
    print(f"  Instruction: {original_instruction}")
    print(f"  Robots: {robot_ids}")
    print(f"  LLM Model: {llm_model}")
    print()

    results = {}

    def generate_for_robot(robot_id: int) -> Tuple[int, str, Dict, Dict, Dict]:
        """단일 로봇용 리셋 코드 생성"""
        target_pos = robot_target_positions.get(robot_id, {})
        current_pos = robot_current_positions.get(robot_id, {})
        fwd_spec = forward_specs.get(robot_id, None)
        fwd_code = forward_codes.get(robot_id, None)

        code, orig, curr, tgt = single_robot_reset_code_gen(
            original_instruction=original_instruction,
            target_positions=target_pos,
            current_positions=current_pos,
            robot_id=robot_id,
            llm_model=llm_model,
            forward_spec=fwd_spec,
            forward_code=fwd_code,
            is_random_reset=is_random_reset,
            task_type=task_type,
        )
        return robot_id, code, orig, curr, tgt

    if parallel and len(robot_ids) > 1:
        with ThreadPoolExecutor(max_workers=len(robot_ids)) as executor:
            futures = {
                executor.submit(generate_for_robot, rid): rid
                for rid in robot_ids
            }

            for future in as_completed(futures):
                robot_id = futures[future]
                try:
                    rid, code, orig, curr, tgt = future.result()
                    results[rid] = (code, orig, curr, tgt)
                except Exception as e:
                    print(f"[Robot {robot_id}] Reset code generation failed: {e}")
                    results[robot_id] = (None, {}, {}, {})
    else:
        for robot_id in robot_ids:
            try:
                rid, code, orig, curr, tgt = generate_for_robot(robot_id)
                results[rid] = (code, orig, curr, tgt)
            except Exception as e:
                print(f"[Robot {robot_id}] Reset code generation failed: {e}")
                results[robot_id] = (None, {}, {}, {})

    print()
    print(CYAN + "=" * line_width + RESET)
    success_count = sum(1 for code, _, _, _ in results.values() if code is not None)
    print(f"  Reset code generation complete: {success_count}/{len(robot_ids)} robots")
    print(CYAN + "=" * line_width + RESET)

    return results


def extract_code_from_response(response: str) -> str:
    """
    LLM 응답에서 Python 코드 추출

    Args:
        response: LLM 응답

    Returns:
        추출된 코드
    """
    if not response:
        return ""

    # 코드 블록 패턴
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        return max(matches, key=len).strip()

    # 코드 시작점 찾기
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute_reset_task', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    return response.strip()
