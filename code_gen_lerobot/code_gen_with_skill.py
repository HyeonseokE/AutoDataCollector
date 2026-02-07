from .llm import llm_response, llm_chat
from .forward_execution.user_prompt import lerobot_code_gen_prompt, turn3_code_gen_prompt
from .forward_execution.turn1_prompt import turn1_detect_objects_prompt
from .forward_execution.turn2_prompt import turn2_grasp_point_prompt

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# System prompt 로드 (고정, 한번만 읽기)
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "forward_execution" / "system_prompt.py"
_SYSTEM_PROMPT = None


def _get_system_prompt() -> str:
    """system_prompt.py에서 docstring 내용을 로드"""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        # ''' ... ''' 사이의 내용 추출
        match = re.search(r"'''(.*?)'''", text, re.DOTALL)
        if match:
            _SYSTEM_PROMPT = match.group(1).strip()
        else:
            _SYSTEM_PROMPT = text.strip()
    return _SYSTEM_PROMPT

def lerobot_code_gen(
    instruction: str,
    object_queries: List[str] = None,
    object_positions: Dict = None,
    use_detection: bool = True,
    detection_timeout: float = 10.0,
    llm_model: str = "gpt-4o-mini",
    robot_id: int = 3,
    visualize_detection: bool = False,
    image_path: str = None,
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
    system_prompt = _get_system_prompt()
    if image_path:
        print(f"\n{YELLOW}" + _log(f"Calling LLM ({llm_model}) with image: {image_path}", step="3/3") + f"{RESET}")
    else:
        print(f"\n{YELLOW}" + _log(f"Calling LLM ({llm_model})", step="3/3") + f"{RESET}")
    response = llm_response(
        llm_model, prompt,
        check_time=True,
        system_prompt=system_prompt,
        image_path=image_path,
    )

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


def _parse_json_from_response(response: str) -> Optional[Dict]:
    """LLM 응답에서 JSON 블록을 추출하여 파싱"""
    # ```json ... ``` 블록 찾기
    json_block_pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(json_block_pattern, response, re.DOTALL)
    if matches:
        for m in matches:
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue

    # { ... } 직접 찾기
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _build_positions_from_grasp_points(
    turn1_response: str,
    turn2_response: str,
    image_path: str,
    robot_id: int = 3,
) -> Dict:
    """
    Turn 1-2 응답에서 pixel 좌표를 추출하고 world 좌표로 변환하여 positions dict를 구성

    CoordinateTransformer가 사용 가능하면 pixel→world 변환을 수행하고,
    사용 불가능하면 Turn 2 응답에서 추출한 pixel 좌표만 반환합니다.

    Args:
        turn1_response: Turn 1 LLM 응답 (물체 리스트)
        turn2_response: Turn 2 LLM 응답 (grasp point 좌표)
        image_path: 초기 이미지 경로 (pixel→world 변환에 필요)
        robot_id: 로봇 번호

    Returns:
        Extended format positions dict:
        {"object_name": {"position": [x,y,z], "gripper_offset": float}}
    """
    import cv2

    # 이미지 해상도 로드 (Gemini 0-1000 좌표 → 실제 pixel 변환용)
    img = cv2.imread(image_path)
    if img is not None:
        img_h, img_w = img.shape[:2]
    else:
        img_w, img_h = 640, 480
    print(f"  [MultiTurn] Image resolution: {img_w}x{img_h}")

    # Turn 1에서 물체 정보 파싱
    # LLM 응답 형식: list [{"label": "...", "estimated_size_cm": [...]}]
    # 또는 dict {"objects": [{"name": "...", ...}]}
    turn1_data = _parse_json_from_response(turn1_response)
    objects_info = {}
    if turn1_data:
        obj_list = None
        if isinstance(turn1_data, list):
            obj_list = turn1_data
        elif isinstance(turn1_data, dict) and "objects" in turn1_data:
            obj_list = turn1_data["objects"]

        if obj_list:
            for obj in obj_list:
                name = obj.get("label") or obj.get("name", "")
                size = obj.get("estimated_size_cm", [5, 5])
                objects_info[name] = {
                    "estimated_size_cm": size,
                    "bbox": obj.get("box_2d") or obj.get("bbox_pixel"),
                }
    print(f"  [MultiTurn] Turn 1 objects: {list(objects_info.keys())}")

    # Turn 2에서 grasp point 파싱
    turn2_data = _parse_json_from_response(turn2_response)
    grasp_points = {}
    if turn2_data and "grasp_points" in turn2_data:
        for gp in turn2_data["grasp_points"]:
            name = gp.get("object_name", "")
            role = gp.get("role", "pick")
            pixel = gp.get("point_pixel")
            if pixel and len(pixel) == 2:
                # Gemini 0-1000 정규화 좌표 → 실제 pixel 좌표 변환
                px, py = pixel[0], pixel[1]
                if px > img_w or py > img_h:
                    # 0-1000 스케일로 판단, 실제 pixel로 변환
                    px = int(px * img_w / 1000)
                    py = int(py * img_h / 1000)
                    print(f"  [MultiTurn] Rescaled '{name}' point: {pixel} → [{px}, {py}] (0-1000 → pixel)")
                else:
                    px, py = int(px), int(py)
                grasp_points[name] = {
                    "role": role,
                    "pixel": [px, py],
                }

    # Pixel→World 변환 시도
    transformer = None
    try:
        from object_detection.localization.coordinate_transform import CoordinateTransformer
        calib_path = Path(__file__).parent.parent / "robot_configs" / "pix2world_matrices" / "pix2world_transform_data.npz"
        if calib_path.exists():
            transformer = CoordinateTransformer(str(calib_path))
            if not transformer.is_ready:
                print(f"  [MultiTurn] CoordinateTransformer loaded but not ready")
                transformer = None
            else:
                print(f"  [MultiTurn] CoordinateTransformer loaded from {calib_path}")
        else:
            print(f"  [MultiTurn] Calibration file not found: {calib_path}")
    except (ImportError, Exception) as e:
        print(f"  [MultiTurn] CoordinateTransformer not available: {e}")

    # positions dict 구성
    positions = {}
    for name, gp_info in grasp_points.items():
        pixel = gp_info["pixel"]
        obj_info = objects_info.get(name, {})

        # 물체 크기에서 gripper_offset 추정 (cm → m)
        size_cm = obj_info.get("estimated_size_cm", [5, 5])
        if isinstance(size_cm, (list, tuple)) and len(size_cm) >= 2:
            width_m = size_cm[0] / 100.0
        else:
            width_m = 0.05
        # gripper_offset: 물체 폭이 gripper opening(7cm)보다 크면 offset 필요
        gripper_offset = max(0.0, (width_m - 0.07) / 2.0) if width_m > 0.07 else 0.02

        if transformer:
            # pixel→world 변환 (2D homography, 평면 가정)
            try:
                wx, wy, wz = transformer.pixel_to_world_2d(int(pixel[0]), int(pixel[1]))
                # 단위 변환: CoordinateTransformer는 cm 단위, 시스템은 m 단위
                wx_m = wx / 100.0
                wy_m = wy / 100.0
                # z는 물체 높이 추정 (cm→m)
                height_cm = size_cm[1] if isinstance(size_cm, (list, tuple)) and len(size_cm) >= 2 else 3.0
                wz_m = height_cm / 100.0

                positions[name] = {
                    "position": [wx_m, wy_m, wz_m],
                    "gripper_offset": gripper_offset,
                    "pixel": pixel,
                }
                print(f"  [MultiTurn] '{name}': pixel={pixel} → world=[{wx_m:.4f}, {wy_m:.4f}, {wz_m:.4f}]m")
                continue
            except Exception as e:
                print(f"  [MultiTurn] pixel→world failed for '{name}': {e}")

        # Fallback: pixel 좌표만 저장 (world 변환 실패)
        height_cm = size_cm[1] if isinstance(size_cm, (list, tuple)) and len(size_cm) >= 2 else 3.0
        positions[name] = {
            "position": [0.0, 0.0, height_cm / 100.0],
            "gripper_offset": gripper_offset,
            "pixel": pixel,
            "_needs_world_coords": True,
        }

    return positions


def lerobot_code_gen_multi_turn(
    instruction: str,
    image_path: str,
    llm_model: str = "gemini-2.0-flash",
    robot_id: int = 3,
    current_episode: int = 1,
    total_episodes: int = 1,
    fallback_positions: Dict = None,
) -> Tuple[str, Dict, Dict]:
    """
    3-Turn 멀티턴 LLM 코드 생성 파이프라인

    Turn 1: 이미지 + instruction → 물체 검출
    Turn 2: grasp point pointing
    Turn 3: 코드 생성

    Args:
        instruction: 자연어 태스크 명령
        image_path: 초기 오버헤드 카메라 이미지 경로
        llm_model: LLM 모델 (Gemini 모델 필요)
        robot_id: 로봇 번호 (2 또는 3)
        current_episode: 현재 에피소드 번호
        total_episodes: 총 에피소드 수
        fallback_positions: Grounding DINO fallback 용 positions
            (LLM 검출 실패 시 사용)

    Returns:
        Tuple[str, Dict, Dict]:
            - 실행 가능한 Python 코드
            - positions dict (extended format)
            - multi_turn_info: 각 턴별 응답 등 메타 정보

    Raises:
        AssertionError: LLM 응답이 None이거나 코드 추출 실패 시
    """
    # ANSI 색상 코드
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    ep_str = f"{current_episode:02d}/{total_episodes:02d}"

    def _log(msg: str, step: str = None) -> str:
        prefix = f"[Forward][{ep_str}][MultiTurn]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Multi-Turn Code Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    # System prompt 로드
    system_prompt = _get_system_prompt()

    # 3-Turn 구성
    print(f"\n{YELLOW}" + _log("Building 3-turn chat...", step="Setup") + f"{RESET}")
    print(f"  Model: {llm_model}")
    print(f"  Image: {image_path}")

    # Turn 1: Object Detection
    turn1_text = turn1_detect_objects_prompt(instruction)

    # Turn 2: Grasp Point Pointing
    turn2_text = turn2_grasp_point_prompt()

    # Turn 3: Code Generation
    turn3_text = turn3_code_gen_prompt(instruction=instruction, robot_id=robot_id)

    turns = [
        {"text": turn1_text, "image_path": image_path},
        {"text": turn2_text, "image_path": None},
        {"text": turn3_text, "image_path": None},
    ]

    # LLM 멀티턴 호출
    print(f"\n{YELLOW}" + _log(f"Calling LLM chat ({llm_model}, 3 turns)...", step="Chat") + f"{RESET}")
    responses = llm_chat(
        model=llm_model,
        system_prompt=system_prompt,
        turns=turns,
        temperature=0.0,
        check_time=True,
    )

    assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"

    turn1_resp, turn2_resp, turn3_resp = responses

    # Turn 1 결과 출력
    print(f"\n{YELLOW}" + _log("Turn 1 — Object Detection result:", step="Turn1") + f"{RESET}")
    turn1_preview = turn1_resp[:300]
    if len(turn1_resp) > 300:
        turn1_preview += "\n... (truncated)"
    print(f"  {turn1_preview}")

    # Turn 2 결과 출력
    print(f"\n{YELLOW}" + _log("Turn 2 — Grasp Point result:", step="Turn2") + f"{RESET}")
    turn2_preview = turn2_resp[:300]
    if len(turn2_resp) > 300:
        turn2_preview += "\n... (truncated)"
    print(f"  {turn2_preview}")

    # Turn 2 응답에서 positions 구성
    print(f"\n{YELLOW}" + _log("Building positions from grasp points...", step="Positions") + f"{RESET}")
    positions = _build_positions_from_grasp_points(
        turn1_response=turn1_resp,
        turn2_response=turn2_resp,
        image_path=image_path,
        robot_id=robot_id,
    )

    # world 좌표 변환 실패한 물체가 있으면 fallback 시도
    needs_fallback = any(
        info.get("_needs_world_coords", False)
        for info in positions.values()
    )
    if needs_fallback and fallback_positions:
        print(f"  {YELLOW}Some objects need world coords — using fallback positions{RESET}")
        for name, info in positions.items():
            if info.get("_needs_world_coords") and name in fallback_positions:
                fb = fallback_positions[name]
                if fb is not None:
                    fb_pos = fb["position"] if isinstance(fb, dict) and "position" in fb else fb
                    info["position"] = list(fb_pos[:3])
                    info["gripper_offset"] = fb.get("gripper_offset", 0.02) if isinstance(fb, dict) else 0.02
                    info.pop("_needs_world_coords", None)
                    print(f"    ✓ '{name}' fallback: pos={info['position']}")

    # positions 결과 출력
    for name, info in positions.items():
        pos = info["position"]
        offset = info.get("gripper_offset", 0.0)
        pixel = info.get("pixel", "N/A")
        needs_world = info.get("_needs_world_coords", False)
        status = "⚠ NO WORLD COORDS" if needs_world else "OK"
        print(f"    {name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], offset={offset:.3f}, pixel={pixel} [{status}]")

    # Turn 3 코드 추출
    print(f"\n{YELLOW}" + _log("Turn 3 — Extracting code...", step="Turn3") + f"{RESET}")
    code = extract_code_from_response(turn3_resp)
    assert code, "Failed to extract code from Turn 3 response"

    # Multi-turn info (디버깅/로깅용)
    multi_turn_info = {
        "turn1_response": turn1_resp,
        "turn2_response": turn2_resp,
        "turn3_response": turn3_resp,
        "turn1_parsed": _parse_json_from_response(turn1_resp),
        "turn2_parsed": _parse_json_from_response(turn2_resp),
    }

    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + "Multi-turn code generation completed.".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    return code, positions, multi_turn_info