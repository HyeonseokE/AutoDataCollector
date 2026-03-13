"""
Reset Execution Code Generator

Generates executable Python code to reverse/undo forward task executions.
Uses object detection to find current positions and context for original positions.

Reset Modes:
- "original": Restore objects to their initial positions (default)
- "random": Shuffle objects to random positions within workspace

Multi-turn mode:
- Uses VLM crop-then-point pipeline (same as forward)
- Grounding DINO 의존 제거, VLM 기반 검출
"""

import re
from typing import Dict, List, Optional, Tuple

from ..llm import llm_response
from .prompt import (
    lerobot_reset_code_gen_prompt,
    turn0_reset_scene_understanding_prompt,
    turn1_reset_bbox_detection_prompt,
    turn_codegen_reset_prompt,
)
from .workspace import (
    classify_objects,
    generate_random_positions,
    ResetWorkspace,
    compute_workspace_bounds,
    draw_workspace_on_image,
)


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


# ============================================================
# Multi-Turn Reset Code Generation
# ============================================================

def _compute_bbox_size_m(
    valid_objects: list,
    img_w: int,
    img_h: int,
    coord_transformer,
) -> Dict[str, Tuple[float, float]]:
    """
    VLM Turn 1의 bbox(0-1000 스케일) → 미터 단위 크기.

    각 bbox의 좌상단/우하단을 pixel → world_2d 변환 후
    width_m = |world_x2 - world_x1|, height_m = |world_y2 - world_y1|

    Args:
        valid_objects: Turn 1 bbox 결과 리스트 [{"box_2d": [ymin,xmin,ymax,xmax], "label": str}, ...]
        img_w: 이미지 폭 (pixels)
        img_h: 이미지 높이 (pixels)
        coord_transformer: CoordinateTransformer 인스턴스

    Returns:
        {label: (width_m, height_m)}
    """
    sizes = {}
    for obj in valid_objects:
        label = obj.get("label", "")
        box = obj.get("box_2d", [])
        if len(box) != 4 or not label:
            continue

        ymin, xmin, ymax, xmax = box

        # 0-1000 → pixel
        px1 = int(xmin * img_w / 1000)
        py1 = int(ymin * img_h / 1000)
        px2 = int(xmax * img_w / 1000)
        py2 = int(ymax * img_h / 1000)

        try:
            # pixel → world (cm) → meters
            wx1, wy1, _ = coord_transformer.pixel_to_world_2d(px1, py1)
            wx2, wy2, _ = coord_transformer.pixel_to_world_2d(px2, py2)

            width_m = abs(wx2 - wx1) / 100.0
            height_m = abs(wy2 - wy1) / 100.0
            sizes[label] = (width_m, height_m)
        except Exception as e:
            print(f"  [BBoxSize] Failed for '{label}': {e}")
            sizes[label] = (0.03, 0.03)  # fallback 3cm

    return sizes


def _match_labels(
    detected_labels: list,
    original_labels: list,
) -> Dict[str, str]:
    """
    Reset VLM 검출 라벨과 Forward original_positions 라벨 간 매칭.

    VLM은 매번 다른 라벨을 생성할 수 있으므로 (예: "chocolate pie" vs "chocolate pie 1")
    substring/fuzzy 매칭으로 대응.

    Args:
        detected_labels: Reset VLM에서 검출된 라벨 리스트
        original_labels: original_positions의 키 리스트

    Returns:
        {detected_label: original_label} 매핑
    """
    def _normalize(label: str) -> str:
        """라벨 정규화: 언더스코어→공백, 소문자, strip"""
        return label.lower().strip().replace("_", " ")

    mapping = {}
    used_originals = set()

    # Pass 0: 정규화 후 정확한 매칭 (언더스코어 vs 공백 차이 해결)
    for dl in detected_labels:
        for ol in original_labels:
            if ol in used_originals:
                continue
            if _normalize(dl) == _normalize(ol):
                mapping[dl] = ol
                used_originals.add(ol)
                break

    # Pass 1: 정확한 매칭 (원본 문자열)
    for dl in detected_labels:
        if dl in mapping:
            continue
        if dl in original_labels and dl not in used_originals:
            mapping[dl] = dl
            used_originals.add(dl)

    # Pass 2: 정규화 후 substring 매칭
    remaining_detected = [dl for dl in detected_labels if dl not in mapping]
    remaining_original = [ol for ol in original_labels if ol not in used_originals]

    for dl in remaining_detected:
        best_match = None
        best_len = 0
        dl_norm = _normalize(dl)
        for ol in remaining_original:
            ol_norm = _normalize(ol)
            # original 라벨이 detected 라벨에 포함
            if ol_norm in dl_norm and len(ol_norm) > best_len:
                best_match = ol
                best_len = len(ol_norm)
            # detected 라벨이 original 라벨에 포함
            elif dl_norm in ol_norm and len(dl_norm) > best_len:
                best_match = ol
                best_len = len(dl_norm)
        if best_match:
            mapping[dl] = best_match
            remaining_original.remove(best_match)

    # Pass 3: 단어 기반 매칭 (정규화 후)
    remaining_detected = [dl for dl in detected_labels if dl not in mapping]
    for dl in remaining_detected:
        dl_words = set(_normalize(dl).split())
        best_match = None
        best_overlap = 0
        for ol in remaining_original:
            ol_words = set(_normalize(ol).split())
            overlap = len(dl_words & ol_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ol
        if best_match and best_overlap > 0:
            mapping[dl] = best_match
            remaining_original.remove(best_match)

    return mapping


def lerobot_reset_code_gen_multi_turn(
    original_instruction: str,
    original_positions: Dict,
    current_state_image_path: str,
    initial_state_image_path: str,
    llm_model: str = "gemini-2.0-flash",
    robot_id: int = 3,
    reset_mode: str = "original",
    random_seed: int = None,
    camera=None,
    coord_transformer=None,
    workspace=None,
    current_episode: int = 1,
    total_episodes: int = 1,
    codegen_model: str = None,
) -> Tuple[str, Dict, Dict, Dict, Dict]:
    """
    VLM Multi-Turn Reset 코드 생성 파이프라인.

    Forward와 동일한 crop-then-point 방식으로 현재 물체 위치를 검출하고,
    reset 코드를 생성합니다. Grounding DINO 의존 없이 VLM만으로 동작.

    Turn 0: Scene Understanding (current + initial 이미지)
    Turn 1: BBox Detection (forward turn1_prompt 재사용)
    Turn 2~N: Crop → Critical Point (forward turn2_prompt 재사용)
    CodeGen: Reset 코드 생성

    Args:
        original_instruction: 원래 forward 태스크 명령
        original_positions: Forward 전 초기 위치 (extended format)
        current_state_image_path: Forward 후 현재 이미지 경로
        initial_state_image_path: Forward 전 초기 이미지 경로
        llm_model: LLM 모델
        robot_id: 로봇 번호
        reset_mode: "original" | "random"
        random_seed: 랜덤 위치 생성용 seed
        camera: RealSense camera (depth용)
        coord_transformer: CoordinateTransformer (pix2world)
        workspace: ResetWorkspace 인스턴스
        current_episode: 현재 에피소드 번호
        total_episodes: 총 에피소드 수

    Returns:
        (generated_code, current_positions, target_positions,
         grippable_objects, obstacle_objects)
    """
    import cv2
    import tempfile
    import json
    import numpy as np
    from pathlib import Path

    from ..llm_utils.gemini import gemini_chat_start, gemini_chat_send
    from ..forward_execution.turn2_prompt import turn2_crop_pointing_prompt
    from ..code_gen_with_skill import _points_to_positions, _parse_json_from_response, _get_system_prompt

    CROP_PADDING = 25  # bbox 패딩 (0-1000 스케일)

    # ANSI colors
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET_COLOR = "\033[0m"
    line_width = 60

    ep_str = f"{current_episode:02d}/{total_episodes:02d}"
    mode_str = "RANDOM RESET" if reset_mode == "random" else "RESET TO ORIGINAL"

    def _log(msg, step=None):
        prefix = f"[Reset][{ep_str}][MultiTurn]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET_COLOR)
    print(MAGENTA + f"LeRobot {mode_str} (Multi-Turn VLM)".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)
    print(f"  Model: {llm_model}")
    print(f"  Current image: {current_state_image_path}")
    print(f"  Initial image: {initial_state_image_path}")

    # ── Step 1: Workspace 준비 ──
    print(f"\n{YELLOW}" + _log("Preparing workspace bounds", step="Setup") + f"{RESET_COLOR}")

    if workspace is None:
        workspace = ResetWorkspace()

    # Auto-compute workspace bounds
    ws_bounds = compute_workspace_bounds(workspace)
    print(f"  Workspace bounds: x=[{ws_bounds[0][0]:.2f}, {ws_bounds[0][1]:.2f}], "
          f"y=[{ws_bounds[1][0]:.2f}, {ws_bounds[1][1]:.2f}]")

    # CoordinateTransformer 로드 (if not provided)
    if coord_transformer is None:
        try:
            from object_detection.localization.coordinate_transform import CoordinateTransformer
            calib_path = Path(__file__).parent.parent.parent / "robot_configs" / "pix2world_matrices" / "pix2world_transform_data.npz"
            if calib_path.exists():
                coord_transformer = CoordinateTransformer(str(calib_path))
                if not coord_transformer.is_ready:
                    coord_transformer = None
                else:
                    print(f"  CoordinateTransformer loaded")
        except Exception as e:
            print(f"  CoordinateTransformer not available: {e}")

    # Workspace 시각화 이미지 생성
    reset_dir = Path(current_state_image_path).parent
    annotated_image_path = None
    if coord_transformer is not None:
        current_img = cv2.imread(current_state_image_path)
        if current_img is not None:
            annotated = draw_workspace_on_image(current_img, ws_bounds, coord_transformer)
            annotated_image_path = str(reset_dir / "workspace_annotated.jpg")
            cv2.imwrite(annotated_image_path, annotated)
            print(f"  Workspace annotated image: {annotated_image_path}")

    # ── Step 2: Gemini chat 시작 ──
    print(f"\n{YELLOW}" + _log("Starting Gemini chat", step="Chat") + f"{RESET_COLOR}")
    system_prompt = _get_system_prompt()
    chat, gen_config = gemini_chat_start(llm_model, system_prompt=system_prompt)

    # ── Turn 0: Scene Understanding ──
    print(f"\n{YELLOW}" + _log("Turn 0 — Scene Understanding", step="Turn0") + f"{RESET_COLOR}")
    original_labels = list(original_positions.keys()) if original_positions else []
    turn0_text = turn0_reset_scene_understanding_prompt(
        original_instruction=original_instruction,
        reset_mode=reset_mode,
        workspace_bounds=ws_bounds,
        original_object_labels=original_labels,
    )
    if original_labels:
        print(f"  Original labels provided to VLM: {original_labels}")
    # Image 1 = annotated current state (or original if no transformer)
    turn0_image = annotated_image_path or current_state_image_path
    turn0_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn0_text,
            "image_path": turn0_image,
            "image_paths": [initial_state_image_path],
        },
        turn_label="Turn 0 (Reset)")
    print(f"  {turn0_resp[:300]}{'...' if len(turn0_resp) > 300 else ''}")

    # ── Turn 1: BBox Detection (Reset 전용 — Forward 라벨 강제 사용) ──
    print(f"\n{YELLOW}" + _log("Turn 1 — BBox Detection", step="Turn1") + f"{RESET_COLOR}")
    turn1_text = turn1_reset_bbox_detection_prompt(
        original_object_labels=original_labels if original_labels else None,
    )
    if original_labels:
        print(f"  Enforcing Forward labels in Turn 1: {original_labels}")
    turn1_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn1_text,
            "image_path": current_state_image_path,  # 원본 이미지 (시각화 없음)
        },
        turn_label="Turn 1 (Reset)")
    print(f"  {turn1_resp[:300]}{'...' if len(turn1_resp) > 300 else ''}")

    # Parse bboxes
    turn1_data = _parse_json_from_response(turn1_resp)
    if isinstance(turn1_data, list):
        obj_list = turn1_data
    elif isinstance(turn1_data, dict):
        obj_list = turn1_data.get("objects", turn1_data.get("detected_objects", []))
    else:
        obj_list = []

    valid_objects = []
    for obj in obj_list:
        box = obj.get("box_2d") or obj.get("bbox") or []
        if len(box) == 4 and obj.get("label"):
            obj["box_2d"] = box
            valid_objects.append(obj)
            print(f"    [{obj['label']}] bbox={box}")

    assert valid_objects, "No valid bboxes detected from Turn 1"

    # 이미지 로드
    full_img = cv2.imread(current_state_image_path)
    assert full_img is not None, f"Cannot read image: {current_state_image_path}"
    img_h, img_w = full_img.shape[:2]

    # ── Turn 2+: Crop-then-Point ──
    all_points = []
    crop_responses = []
    crop_dir = str(reset_dir / "crops")
    Path(crop_dir).mkdir(parents=True, exist_ok=True)

    for i, obj in enumerate(valid_objects):
        label = obj["label"]
        ymin, xmin, ymax, xmax = obj["box_2d"]

        print(f"\n{YELLOW}" + _log(f"Crop — {label}", step=f"Crop{i}") + f"{RESET_COLOR}")

        # Padding + clamp (0-1000 스케일)
        ymin_p = max(0, ymin - CROP_PADDING)
        xmin_p = max(0, xmin - CROP_PADDING)
        ymax_p = min(1000, ymax + CROP_PADDING)
        xmax_p = min(1000, xmax + CROP_PADDING)

        # 0-1000 → pixel
        crop_x1 = int(xmin_p * img_w / 1000)
        crop_y1 = int(ymin_p * img_h / 1000)
        crop_x2 = int(xmax_p * img_w / 1000)
        crop_y2 = int(ymax_p * img_h / 1000)

        crop_img = full_img[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w = crop_img.shape[:2]
        print(f"    bbox=[{ymin},{xmin},{ymax},{xmax}] → crop ({crop_w}x{crop_h})")

        safe_label = label.replace(" ", "_").replace("/", "_")
        crop_path = f"{crop_dir}/crop_{safe_label}.jpg"
        cv2.imwrite(crop_path, crop_img)

        # Send crop + pointing prompt
        resp = gemini_chat_send(chat, gen_config,
            {
                "text": turn2_crop_pointing_prompt(label, has_side_view=False),
                "image_path": crop_path,
            },
            turn_label=f"Crop: {label}")
        crop_responses.append({"label": label, "response": resp})
        print(f"    {resp[:200]}{'...' if len(resp) > 200 else ''}")

        # Parse critical_points
        parsed = _parse_json_from_response(resp)
        if not parsed:
            print(f"    [Warning] Failed to parse points for '{label}'")
            continue

        if "critical_points" in parsed:
            oh_points = parsed["critical_points"]
        elif "overhead_critical_points" in parsed:
            oh_points = parsed["overhead_critical_points"]
        else:
            print(f"    [Warning] No recognized point keys for '{label}'")
            continue

        for pt in oh_points:
            point_2d = pt.get("point_2d", [])
            if len(point_2d) != 2:
                continue
            norm_y, norm_x = point_2d
            crop_px = int(norm_x * crop_w / 1000)
            crop_py = int(norm_y * crop_h / 1000)
            px = crop_x1 + crop_px
            py = crop_y1 + crop_py

            entry = {
                "object_label": label,
                "label": pt.get("label", ""),
                "role": pt.get("role", "interaction"),
                "reasoning": pt.get("reasoning", ""),
                "point_2d": point_2d,
                "px": px, "py": py,
                "crop_px": crop_px, "crop_py": crop_py,
            }
            print(f"    [{pt.get('role','?')}] ({norm_y},{norm_x}) → full({px},{py})")
            all_points.append(entry)

    print(f"\n  Total: {len(all_points)} points across {len(valid_objects)} objects")

    # ── Step 6: 좌표 변환 & 분류 ──
    print(f"\n{YELLOW}" + _log("Building positions (pixel → world)", step="Positions") + f"{RESET_COLOR}")
    current_positions = _points_to_positions(all_points, robot_id=robot_id, camera=camera)

    # BBox → meters size
    bbox_sizes = {}
    if coord_transformer is not None:
        bbox_sizes = _compute_bbox_size_m(valid_objects, img_w, img_h, coord_transformer)

    # Enrich positions with bbox_size_m and grippable flag
    from .workspace import is_grippable, GRIPPER_MAX_OPEN_WIDTH
    for label, info in current_positions.items():
        if label in bbox_sizes:
            info["bbox_size_m"] = list(bbox_sizes[label])
            info["grippable"] = is_grippable(bbox_sizes[label])
        else:
            info["bbox_size_m"] = [0.03, 0.03]
            info["grippable"] = True

    # Classify objects
    grippable_objects, obstacle_objects = classify_objects(current_positions)
    print(f"  Grippable: {list(grippable_objects.keys())}")
    print(f"  Obstacles: {list(obstacle_objects.keys())}")

    # ── Step 7: 라벨 매칭 + Target 위치 결정 ──
    print(f"\n{YELLOW}" + _log("Matching labels (reset VLM → original)", step="LabelMatch") + f"{RESET_COLOR}")

    detected_labels = list(grippable_objects.keys()) + list(obstacle_objects.keys())
    original_labels = list(original_positions.keys())
    label_map = _match_labels(detected_labels, original_labels)

    for dl, ol in label_map.items():
        match_type = "exact" if dl == ol else "fuzzy"
        print(f"    {dl} → {ol} ({match_type})")

    unmatched = [dl for dl in detected_labels if dl not in label_map]
    if unmatched:
        print(f"  {RED}[Warning] Unmatched labels: {unmatched}{RESET_COLOR}")

    print(f"\n{YELLOW}" + _log(f"Determining target positions ({reset_mode} mode)", step="Target") + f"{RESET_COLOR}")

    if reset_mode == "random":
        # random 모드: label_map을 사용하여 initial_positions 키를 매핑
        mapped_initial = {}
        for dl, ol in label_map.items():
            if ol in original_positions:
                mapped_initial[dl] = original_positions[ol]

        target_positions = generate_random_positions(
            grippable_objects=grippable_objects,
            obstacle_objects=obstacle_objects,
            initial_positions=mapped_initial if mapped_initial else original_positions,
            workspace=workspace,
            seed=random_seed,
        )
        print(f"  Random target positions generated:")
    else:
        # original mode: label_map을 사용하여 grippable → original 매핑
        target_positions = {}
        for name in grippable_objects.keys():
            original_key = label_map.get(name, name)
            if original_key in original_positions and original_positions[original_key] is not None:
                target_positions[name] = original_positions[original_key]
                print(f"    {name} → target from '{original_key}'")

        if not target_positions:
            print(f"  {RED}[Error] No target positions matched!{RESET_COLOR}")
            print(f"  Grippable labels: {list(grippable_objects.keys())}")
            print(f"  Original labels: {list(original_positions.keys())}")
            print(f"  Label map: {label_map}")
            raise ValueError(
                f"Reset target matching failed: grippable={list(grippable_objects.keys())} "
                f"vs original={list(original_positions.keys())}. Label map: {label_map}"
            )

        print(f"  Restoring to initial positions:")

    for name, info in target_positions.items():
        if isinstance(info, dict) and "position" in info:
            pos = info["position"]
            print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            print(f"    + {name}: [{info[0]:.4f}, {info[1]:.4f}, {info[2]:.4f}]")

    # ── Context Summary Turn (Session 1 마지막) ──
    from .prompt import reset_context_summary_prompt, codegen_reset_with_context_prompt
    print(f"\n{YELLOW}" + _log("Context Summary (handoff)", step="Summary") + f"{RESET_COLOR}")
    summary_resp = gemini_chat_send(chat, gen_config,
        {"text": reset_context_summary_prompt()},
        turn_label="Context Summary (Reset)")
    print(f"  Summary: {summary_resp[:200]}{'...' if len(summary_resp) > 200 else ''}")

    # ── Code Generation (새 Session 2) ──
    session2_model = codegen_model or llm_model
    print(f"\n{YELLOW}" + _log(f"Code Generation (new session: {session2_model})", step="CodeGen") + f"{RESET_COLOR}")
    codegen_chat, codegen_config = gemini_chat_start(session2_model, system_prompt=system_prompt)
    codegen_resp = gemini_chat_send(codegen_chat, codegen_config,
        {"text": codegen_reset_with_context_prompt(
            context_summary=summary_resp,
            target_positions=target_positions,
            current_positions={
                name: grippable_objects[name]
                for name in grippable_objects
                if name in target_positions
            },
            robot_id=robot_id,
            is_random_reset=(reset_mode == "random"),
            workspace_bounds=ws_bounds,
            all_points=all_points,
        )},
        turn_label="CodeGen (Reset)")

    code = extract_code_from_response(codegen_resp)
    assert code, "Failed to extract code from CodeGen response"

    # Summary
    print(GRAY + "=" * line_width + RESET_COLOR)
    print(LIGHT_GREEN + f"{mode_str} multi-turn code gen completed.".center(line_width) + RESET_COLOR)
    print(GRAY + "=" * line_width + RESET_COLOR)

    return code, current_positions, target_positions, grippable_objects, obstacle_objects
