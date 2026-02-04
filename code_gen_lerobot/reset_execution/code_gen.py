"""
Reset Execution Code Generator

Generates executable Python code to reverse/undo forward task executions.
Uses object detection to find current positions and context for original positions.

Reset Modes:
- "original": Restore objects to their initial positions (default)
- "random": Shuffle objects to random positions within workspace
"""

import re
from typing import Dict, List, Optional, Tuple

from ..llm import llm_response
from .prompt import lerobot_reset_code_gen_prompt
from .workspace import classify_objects, generate_random_positions, ResetWorkspace


def lerobot_reset_code_gen(
    original_instruction: str,
    original_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
    forward_code: str = None,
    object_queries: List[str] = None,
    detection_timeout: float = 10.0,
    visualize_detection: bool = False,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
    external_camera=None,
    # Episode tracking for logging
    current_episode: int = 1,
    total_episodes: int = 1,
    # Pre-detected positions (for multi-robot shared detection)
    current_positions: Dict[str, Dict] = None,
) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    LeRobot SO-101용 리셋 코드 생성

    Detection을 사용하여 현재 물체 위치를 파악하고,
    Forward execution context를 참조하여 초기 환경으로 되돌리거나
    랜덤 위치로 shuffle하는 코드를 생성합니다.

    Args:
        original_instruction: 원래 태스크 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        original_positions: 객체들의 원래 위치 (forward 실행 전 위치, context에서 로드)
        forward_spec: Forward execution에서 생성된 spec (step-by-step plan)
        forward_code: Forward execution에서 생성된 Python 코드
        object_queries: Detection할 객체 리스트 (None이면 original_positions의 키 사용)
        detection_timeout: Detection 타임아웃 (초)
        visualize_detection: Detection 결과 시각화 여부
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호 (2 또는 3)
        reset_mode: "original" (초기 위치로 복귀) 또는 "random" (랜덤 위치로 shuffle)
        random_seed: 랜덤 위치 생성용 seed (재현성 보장)
        external_camera: 외부에서 전달받은 카메라 인스턴스 (공유 모드)
                        - Recording 카메라와 공유하여 리소스 충돌 방지
        current_positions: 미리 감지된 현재 위치 (multi-robot 공유 감지용)
                          - 제공되면 내부 detection을 건너뛰고 이 위치 사용
                          - extended format: {name: {"position": [...], "gripper_offset": float, ...}}

    Returns:
        Tuple[str, Dict, Dict, Dict]: (리셋 코드, 원래 위치, 현재 위치, 타겟 위치)
        - 타겟 위치: reset_mode에 따라 original_positions 또는 random positions

    Example:
        # Original mode (restore)
        reset_code, orig, curr, target = lerobot_reset_code_gen(
            original_instruction="pick red cup and place on blue box",
            original_positions={"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]},
            reset_mode="original",
        )

        # Random mode (shuffle)
        reset_code, orig, curr, target = lerobot_reset_code_gen(
            original_instruction="pick red cup and place on blue box",
            original_positions={"red cup": [0.15, 0.05, 0.02], "blue box": [0.20, -0.05, 0.03]},
            reset_mode="random",
            random_seed=42,
        )
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET_COLOR = "\033[0m"
    line_width = 60

    # Episode prefix for logging
    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    def _log(msg: str, step: str = None) -> str:
        prefix = f"[Reset][{ep_str}]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    mode_str = "RANDOM RESET" if reset_mode == "random" else "RESET TO ORIGINAL"

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(MAGENTA + f"LeRobot {mode_str}".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)

    # 1) Context 검증
    print(f"{YELLOW}" + _log("Validating execution context", step="1/5") + f"{RESET_COLOR}")

    if not original_instruction:
        raise ValueError("original_instruction is required for reset code generation")

    if not original_positions or len(original_positions) == 0:
        raise ValueError("original_positions is required for reset code generation")

    print(f"  Original task: {original_instruction[:50]}...")
    print(f"  Reset mode: {reset_mode}")
    print(f"  Objects: {list(original_positions.keys())}")

    # 원래 위치 출력 (extended format 지원)
    print(f"\n  Initial positions (forward 시작 시):")
    for name, info in original_positions.items():
        if info is None:
            continue
        elif isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # 2) Detection으로 현재 위치 획득 (확장 정보 포함)
    # current_positions가 제공되면 detection 건너뛰기 (multi-robot 공유 감지 모드)
    if current_positions is not None:
        print(f"\n{YELLOW}" + _log("Using pre-detected positions (shared detection)", step="2/5") + f"{RESET_COLOR}")
        extended_detections = current_positions
    else:
        print(f"\n{YELLOW}" + _log("Detecting current object positions", step="2/5") + f"{RESET_COLOR}")

        # object_queries가 없으면 original_positions의 키 사용
        if object_queries is None:
            object_queries = list(original_positions.keys())

        print(f"  Detecting: {object_queries}")

        try:
            from run_detect import run_realtime_detection
            # 확장 정보 요청 (bbox_size_m, grippable 포함)
            # external_camera가 있으면 공유하여 리소스 충돌 방지
            extended_detections = run_realtime_detection(
                queries=object_queries,
                timeout=detection_timeout,
                unit="m",
                visualize=visualize_detection,
                return_extended=True,
                robot_id=robot_id,
                external_camera=external_camera,
            )
        except ImportError as e:
            print(f"{RED}[Error] Could not import detection module: {e}{RESET_COLOR}")
            raise ValueError("Detection module not available")
        except Exception as e:
            print(f"{RED}[Error] Detection failed: {e}{RESET_COLOR}")
            raise ValueError(f"Detection failed: {e}")

    # Detection 결과 출력
    found_count = sum(1 for v in extended_detections.values() if v is not None)
    print(f"  Found: {found_count}/{len(extended_detections)} objects")

    print(f"\n  Current positions (detected):")
    for name, info in extended_detections.items():
        if info:
            pos = info["position"]
            bbox = info.get("bbox_size_m")
            grippable = info.get("grippable", True)
            bbox_str = f"[{bbox[0]*100:.1f}x{bbox[1]*100:.1f}cm]" if bbox else "[?x?]"
            grip_str = "grippable" if grippable else "OBSTACLE"
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] {bbox_str} ({grip_str})")
        else:
            print(f"    x {name}: NOT FOUND")

    # Workspace 범위 검증
    print(f"\n{YELLOW}[Validation] Checking workspace bounds...{RESET_COLOR}")
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from lerobot_cap.workspace import BaseWorkspace
    workspace = BaseWorkspace()
    print(f"  Workspace: x_min_world={workspace.x_min_world:.2f}m, reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

    for obj_name, obj_info in extended_detections.items():
        if obj_info is None:
            continue
        pos = obj_info.get("position")
        if pos is None:
            continue
        position_m = np.array([pos[0], pos[1], pos[2]])
        if not workspace.is_reachable(position_m):
            warning_msg = (
                f"[WARNING] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m "
                f"is OUTSIDE workspace!"
            )
            print(f"{RED}{warning_msg}{RESET_COLOR}")
            print(f"{RED}  Workspace: x_min_world={workspace.x_min_world:.2f}m, reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m{RESET_COLOR}")
            assert False, f"Object '{obj_name}' is outside workspace bounds"
        else:
            print(f"  {LIGHT_GREEN}✓ '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m - OK{RESET_COLOR}")

    # 3) 객체 분류 (grippable vs obstacle)
    print(f"\n{YELLOW}" + _log("Classifying objects", step="3/5") + f"{RESET_COLOR}")

    grippable_objects, obstacle_objects = classify_objects(extended_detections)

    print(f"  Grippable (will move): {list(grippable_objects.keys())}")
    print(f"  Obstacles (fixed): {list(obstacle_objects.keys())}")

    # grippable 객체 중 검출 실패 확인
    grippable_not_found = [
        name for name in original_positions.keys()
        if name not in grippable_objects and name not in obstacle_objects
    ]
    if grippable_not_found:
        print(f"\n{RED}[Error] Required objects not detected: {grippable_not_found}{RESET_COLOR}")
        raise ValueError(f"Cannot generate reset code: objects not detected - {grippable_not_found}")

    # 4) 타겟 위치 결정
    print(f"\n{YELLOW}" + _log(f"Determining target positions ({reset_mode} mode)", step="4/5") + f"{RESET_COLOR}")

    if reset_mode == "random":
        # 랜덤 위치 생성
        target_positions = generate_random_positions(
            grippable_objects=grippable_objects,
            obstacle_objects=obstacle_objects,
            initial_positions=original_positions,
            seed=random_seed,
        )
        print(f"  Random target positions generated:")
    else:
        # original 모드: grippable 객체만 초기 위치로
        target_positions = {
            name: original_positions[name]
            for name in grippable_objects.keys()
            if name in original_positions and original_positions[name] is not None
        }
        print(f"  Restoring to initial positions:")

    for name, info in target_positions.items():
        if isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # 현재 위치 추출 (단순 형식으로 변환)
    current_positions = {
        name: info["position"]
        for name, info in grippable_objects.items()
        if name in target_positions
    }

    # 5) 프롬프트 생성 및 LLM 호출
    print(f"\n{YELLOW}" + _log(f"Generating reset code via LLM ({llm_model})", step="5/5") + f"{RESET_COLOR}")

    prompt = lerobot_reset_code_gen_prompt(
        original_instruction=original_instruction,
        target_positions=target_positions,
        current_positions=current_positions,
        forward_spec=forward_spec,
        forward_code=forward_code,
        robot_id=robot_id,
        is_random_reset=(reset_mode == "random"),
    )

    response = llm_response(llm_model, prompt, check_time=True)

    # 코드 추출
    code = extract_code_from_response(response)

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(LIGHT_GREEN + f"{mode_str} code generation completed.".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)

    return code, original_positions, current_positions, target_positions


def lerobot_reset_code_gen_from_context(
    context_path: str,
    detection_timeout: float = 10.0,
    visualize_detection: bool = False,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    저장된 execution context 파일에서 리셋 코드 생성

    Args:
        context_path: execution_context.json 파일 경로
        detection_timeout: Detection 타임아웃 (초)
        visualize_detection: Detection 결과 시각화 여부
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호
        reset_mode: "original" (초기 위치로 복귀) 또는 "random" (랜덤 위치로 shuffle)
        random_seed: 랜덤 위치 생성용 seed

    Returns:
        Tuple[str, Dict, Dict, Dict]: (리셋 코드, 원래 위치, 현재 위치, 타겟 위치)

    Example:
        reset_code, orig_pos, curr_pos, target_pos = lerobot_reset_code_gen_from_context(
            context_path="outputs/execution_context.json",
            reset_mode="random",
        )
    """
    import json

    # Context 파일 로드
    with open(context_path, "r", encoding="utf-8") as f:
        context = json.load(f)

    # 필수 필드 확인
    required_fields = ["instruction", "object_positions"]
    for field in required_fields:
        if field not in context:
            raise ValueError(f"Context file missing required field: {field}")

    return lerobot_reset_code_gen(
        original_instruction=context["instruction"],
        original_positions=context["object_positions"],
        forward_spec=context.get("generated_spec"),
        forward_code=context.get("generated_code"),
        detection_timeout=detection_timeout,
        visualize_detection=visualize_detection,
        llm_model=llm_model,
        robot_id=robot_id,
        reset_mode=reset_mode,
        random_seed=random_seed,
    )


def extract_code_from_response(response: str) -> str:
    """
    LLM 응답에서 Python 코드 블록을 추출

    Args:
        response: LLM 응답 문자열

    Returns:
        추출된 Python 코드
    """
    # 코드 블록 패턴 (```python ... ``` 또는 ``` ... ```)
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # 가장 긴 코드 블록 반환 (보통 메인 코드)
        return max(matches, key=len).strip()

    # 코드 블록이 없으면 전체 응답 반환 (plain text 형식일 수 있음)
    # "from skills" 또는 "def execute"로 시작하는 부분 찾기
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    # 그래도 없으면 전체 반환
    return response.strip()
