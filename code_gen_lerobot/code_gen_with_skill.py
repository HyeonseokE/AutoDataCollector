from .llm import llm_response
from .forward_execution.prompt import lerobot_code_gen_prompt

import re
from typing import Dict, List, Optional

def lerobot_code_gen(
    instruction: str,
    object_queries: List[str] = None,
    object_positions: Dict = None,
    use_detection: bool = True,
    detection_timeout: float = 10.0,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    visualize_detection: bool = False,
    # Episode tracking for logging
    current_episode: int = 1,
    total_episodes: int = 1,
) -> str:
    """
    LeRobot SO-101용 스킬 기반 코드 생성 (단순화 버전)

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_queries: 디텍션 ON시 찾을 객체 리스트 (예: ["red cup", "blue box"])
        object_positions: 디텍션 OFF시 직접 전달할 위치 딕셔너리
                         Extended format: {"name": {"position": [x,y,z], "gripper_offset": float, ...}}
                         Legacy format: {"name": [x,y,z]} (gripper_offset will be 0.02 default)
        use_detection: True면 object_detection으로 위치 획득, False면 object_positions 사용
        detection_timeout: 디텍션 타임아웃 (초)
        llm_model: LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        실행 가능한 Python 코드 문자열

    Raises:
        ValueError: use_detection=True인데 object_queries가 없거나,
                   use_detection=False인데 object_positions가 없는 경우

    Example:
        # 디텍션 ON (카메라로 객체 위치 자동 획득)
        code = lerobot_code_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_queries=["red cup", "blue box"],
            use_detection=True,
        )

        # 디텍션 OFF (위치 직접 전달 - extended format)
        code = lerobot_code_gen(
            instruction="빨간 컵을 파란 상자에 놓아라",
            object_positions={
                "red cup": {"position": [0.15, 0.05, 0.02], "gripper_offset": 0.025},
                "blue box": {"position": [0.20, -0.05, 0.03], "gripper_offset": 0.0},
            },
            use_detection=False,
        )
    """

    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    # Episode prefix for logging
    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    def _log(msg: str, step: str = None) -> str:
        prefix = f"[Forward][{ep_str}]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Code Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 1) 객체 위치 획득
    if use_detection:
        if object_queries is None or len(object_queries) == 0:
            raise ValueError("use_detection=True requires object_queries (non-empty list)")

        print(f"{YELLOW}" + _log(f"Detecting objects: {object_queries}", step="1/3") + f"{RESET}")

        try:
            from run_detect import run_realtime_detection
            positions = run_realtime_detection(
                queries=object_queries,
                timeout=detection_timeout,
                unit="m",
                visualize=visualize_detection,
            )
        except ImportError as e:
            print(f"[Error] Could not import object_positions module: {e}")
            print("[Fallback] Using empty positions")
            positions = {q: None for q in object_queries}
        except Exception as e:
            print(f"[Error] Detection failed: {e}")
            positions = {q: None for q in object_queries}

    else:
        if object_positions is None or len(object_positions) == 0:
            raise ValueError("use_detection=False requires object_positions (non-empty dict)")

        print(f"{YELLOW}" + _log("Using provided positions", step="1/3") + f"{RESET}")
        positions = object_positions

    # Normalize to extended format if legacy format is used
    # Legacy: {"name": [x,y,z]} → Extended: {"name": {"position": [x,y,z], "gripper_offset": 0.02}}
    normalized_positions = {}
    for name, info in positions.items():
        if info is None:
            normalized_positions[name] = None
        elif isinstance(info, dict) and "position" in info:
            # Already extended format
            normalized_positions[name] = info
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            # Legacy format - convert to extended with default gripper_offset
            normalized_positions[name] = {
                "position": list(info[:3]),
                "gripper_offset": 0.02,  # default 2cm
            }
        else:
            normalized_positions[name] = None

    positions = normalized_positions

    # 검출 결과 출력
    found_count = sum(1 for v in positions.values() if v is not None)
    print(f"  Found: {found_count}/{len(positions)} objects")
    for name, info in positions.items():
        if info:
            pos = info["position"]
            offset = info.get("gripper_offset", 0.0)
            bbox = info.get("bbox_size_m")
            grippable = info.get("grippable", True)

            # 기본 정보
            base_info = f"pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"

            # bbox 크기 정보
            if bbox:
                width_cm = bbox[0] * 100
                height_cm = bbox[1] * 100
                size_info = f"size=[{width_cm:.1f}x{height_cm:.1f}]cm"
            else:
                size_info = "size=N/A"

            # grippable 및 offset 정보
            grip_info = f"grippable={grippable}, offset={offset*1000:.1f}mm"

            print(f"    ✓ {name}: {base_info}, {size_info}, {grip_info}")
        else:
            print(f"    ✗ {name}: NOT FOUND")

    # 검출 결과 검증 - 모든 객체가 검출되어야 함
    not_found = [name for name, info in positions.items() if info is None]
    if not_found:
        print("\n" + "="*60)
        print("ERROR: Required objects not detected!")
        print("="*60)
        print(f"  Missing objects: {not_found}")
        print(f"  Detected objects: {[name for name, info in positions.items() if info is not None]}")
        print("="*60)
        assert False, f"Cannot generate code: objects not detected - {not_found}"

    # 2) 프롬프트 생성
    print(f"\n{YELLOW}" + _log("Generating prompt", step="2/3") + f"{RESET}")
    prompt = lerobot_code_gen_prompt(
        instruction=instruction,
        object_positions=positions,
        robot_id=robot_id,
    )

    # 3) LLM 호출
    print(f"\n{YELLOW}" + _log(f"Calling LLM ({llm_model})", step="3/3") + f"{RESET}")
    response = llm_response(llm_model, prompt, check_time=True)

    # LLM 응답 검증
    assert response is not None, "LLM response is None - check server connection or API key"

    # 4) 코드 추출
    code = extract_code_from_response(response)

    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + "Code generation completed successfully.".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # 코드와 positions 함께 반환
    return code, positions


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
    # "from skills" 또는 "def execute_task"로 시작하는 부분 찾기
    lines = response.split('\n')
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('from skills', 'from code_gen', 'def execute_task', 'import ')):
            code_start = i
            break

    if code_start >= 0:
        return '\n'.join(lines[code_start:]).strip()

    # 그래도 없으면 전체 반환
    return response.strip()