from .llm import llm_response
from .forward_execution.user_prompt import lerobot_code_gen_prompt, turn3_code_gen_prompt
from .forward_execution.turn0_prompt import turn0_scene_understanding_prompt
from .forward_execution.turn1_prompt import turn1_detect_task_relevant_objects_prompt
from .forward_execution.turn2_prompt import turn2_crop_pointing_prompt
from .forward_execution.turn_test_prompt import turn_test_waypoint_trajectory_prompt

import json
import os
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
                "red cup": {"position": [0.15, 0.05, 0.02], "gripper_offset": 0.015},
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
    # Legacy: {"name": [x,y,z]} → Extended: {"name": {"position": [x,y,z], "gripper_offset": 0.01}}
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
                "gripper_offset": 0.01,  # default 2cm
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


def _points_to_positions(
    all_points: list,
    robot_id: int = 3,
    camera=None,
) -> Dict:
    """
    Crop-then-point 결과에서 object별 grasp point를 선택하고 pixel→world 변환.

    각 object에서 "grasp" role 포인트를 우선 선택, 없으면 첫 번째 포인트 사용.

    Args:
        all_points: [{"object_label", "label", "role", "px", "py"}, ...]
        robot_id: 로봇 번호
        camera: depth 카메라 (3D 변환용)

    Returns:
        {object_label: {"position": [x,y,z], "gripper_offset": float, "pixel": [px,py]}}
    """
    # Object별로 grasp point 선택
    grasp_by_object = {}
    for pt in all_points:
        obj = pt["object_label"]
        if obj not in grasp_by_object:
            grasp_by_object[obj] = pt  # 첫 번째 point (fallback)
        elif pt.get("role") == "grasp" and grasp_by_object[obj].get("role") != "grasp":
            grasp_by_object[obj] = pt  # grasp 우선

    if not grasp_by_object:
        return {}

    # CoordinateTransformer 로드
    transformer = None
    use_3d = False
    try:
        from object_detection.localization.coordinate_transform import CoordinateTransformer
        calib_path = Path(__file__).parent.parent / "robot_configs" / "pix2world_matrices" / "pix2world_transform_data.npz"
        if calib_path.exists():
            transformer = CoordinateTransformer(str(calib_path))
            if not transformer.is_ready:
                transformer = None
            else:
                use_3d = (transformer.transform_matrix_3d is not None
                          and transformer.camera_intrinsics is not None
                          and camera is not None)
                mode = "3D" if use_3d else "2D"
                print(f"  [CropPoint] CoordinateTransformer [{mode}]")
    except Exception as e:
        print(f"  [CropPoint] CoordinateTransformer not available: {e}")

    # Depth 프레임 (3D 변환용)
    depth_frame = None
    if use_3d and camera is not None:
        try:
            _, depth_frame = camera.get_frames()
            if depth_frame is None:
                use_3d = False
        except Exception:
            use_3d = False

    # positions 구성
    positions = {}
    for obj_label, pt in grasp_by_object.items():
        px, py = pt["px"], pt["py"]

        if transformer:
            try:
                if use_3d and depth_frame is not None:
                    depth_m = camera.get_depth_at_pixel(px, py, depth_frame)
                    wx, wy, wz = transformer.pixel_depth_to_world(px, py, depth_m)
                else:
                    wx, wy, wz = transformer.pixel_to_world_2d(px, py)

                positions[obj_label] = {
                    "position": [wx / 100.0, wy / 100.0, wz / 100.0],
                    "gripper_offset": 0.01,
                    "pixel": [px, py],
                }
                print(f"    {obj_label}: pixel=({px},{py}) → world={positions[obj_label]['position']}")
                continue
            except Exception as e:
                print(f"    {obj_label}: pixel→world failed: {e}")

        # Fallback
        positions[obj_label] = {
            "position": [0.0, 0.0, 0.03],
            "gripper_offset": 0.01,
            "pixel": [px, py],
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
    camera=None,
    cad_image_dirs: List[str] = None,
    side_view_image: str = None,
) -> Tuple[str, Dict, Dict]:
    """
    Crop-then-Point 멀티턴 LLM 코드 생성 파이프라인

    Turn 0: 이미지 (+ CAD) + instruction → 장면 이해 (reasoning only)
    Turn 1: bbox 검출 (JSON)
    Turn 2~N: 물체별 crop → critical point pointing
    Turn N+1: 코드 생성

    Args:
        instruction: 자연어 태스크 명령
        image_path: 초기 오버헤드 카메라 이미지 경로
        llm_model: LLM 모델 (Gemini 모델 필요)
        robot_id: 로봇 번호 (2 또는 3)
        current_episode: 현재 에피소드 번호
        total_episodes: 총 에피소드 수
        fallback_positions: Grounding DINO fallback 용 positions
        camera: RealSenseD435 카메라 (depth 기반 3D 좌표 변환용)
        cad_image_dirs: CAD 참조 이미지 디렉토리 리스트 (옵션)

    Returns:
        Tuple[str, Dict, Dict]:
            - 실행 가능한 Python 코드
            - positions dict (extended format)
            - multi_turn_info: 각 턴별 응답 등 메타 정보
    """
    import cv2
    import glob as glob_mod
    import tempfile
    from .llm_utils.gemini import gemini_chat_start, gemini_chat_send

    CROP_PADDING = 25  # bbox 패딩 (0-1000 스케일)

    GRAY = "\033[90m"
    CYAN = "\033[96m"
    LIGHT_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    line_width = 60

    ep_str = f"{current_episode:02d}/{total_episodes:02d}"

    def _log(msg, step=None):
        prefix = f"[Forward][{ep_str}][CropPoint]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {msg}"

    print(GRAY + "=" * line_width + RESET)
    print(CYAN + "LeRobot Crop-then-Point Code Generation".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)
    print(f"  Model: {llm_model}")
    print(f"  Image: {image_path}")

    # CAD 이미지 수집 (Step 6)
    cad_paths = []
    if cad_image_dirs:
        for d in cad_image_dirs:
            cad_paths.extend(sorted(glob_mod.glob(f"{d}/*.jpg")))
            cad_paths.extend(sorted(glob_mod.glob(f"{d}/*.png")))
        print(f"  CAD images: {len(cad_paths)} files from {len(cad_image_dirs)} dirs")

    # Chat session 시작
    system_prompt = _get_system_prompt()
    chat, gen_config = gemini_chat_start(llm_model, system_prompt=system_prompt)

    has_cad = bool(cad_paths)

    # ── Turn 0: Scene Understanding (이미지 + CAD) — test6 방식 ──
    print(f"\n{YELLOW}" + _log("Turn 0 — Scene Understanding", step="Turn0") + f"{RESET}")
    turn0_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn0_scene_understanding_prompt(instruction, has_cad=has_cad),
            "image_path": image_path,
            "image_paths": cad_paths,
        },
        turn_label="Turn 0")
    print(f"  {turn0_resp[:300]}{'...' if len(turn0_resp) > 300 else ''}")

    # ── Turn 1: BBox Detection (이미지 재전송) — test6 방식 ──
    print(f"\n{YELLOW}" + _log("Turn 1 — BBox Detection", step="Turn1") + f"{RESET}")
    turn1_resp = gemini_chat_send(chat, gen_config,
        {
            "text": turn1_detect_task_relevant_objects_prompt(),
            "image_path": image_path,
        },
        turn_label="Turn 1")
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
    full_img = cv2.imread(image_path)
    assert full_img is not None, f"Cannot read image: {image_path}"
    img_h, img_w = full_img.shape[:2]

    # ── Turn 2+: Crop-then-Point ──
    all_points = []
    crop_responses = []
    crop_dir = tempfile.mkdtemp(prefix="crop_")

    for i, obj in enumerate(valid_objects):
        label = obj["label"]
        ymin, xmin, ymax, xmax = obj["box_2d"]

        print(f"\n{YELLOW}" + _log(f"Crop — {label}", step=f"Crop{i}") + f"{RESET}")

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

        # Save crop
        safe_label = label.replace(" ", "_").replace("/", "_")
        crop_path = f"{crop_dir}/crop_{safe_label}.jpg"
        cv2.imwrite(crop_path, crop_img)

        # Send crop + pointing prompt
        resp = gemini_chat_send(chat, gen_config,
            {"text": turn2_crop_pointing_prompt(label), "image_path": crop_path},
            turn_label=f"Crop: {label}")
        crop_responses.append({"label": label, "response": resp})
        print(f"    {resp[:200]}{'...' if len(resp) > 200 else ''}")

        # Parse critical_points
        parsed = _parse_json_from_response(resp)
        if not parsed or "critical_points" not in parsed:
            print(f"    [Warning] Failed to parse points for '{label}'")
            continue

        for pt in parsed["critical_points"]:
            point_2d = pt.get("point_2d", [])
            if len(point_2d) != 2:
                continue
            norm_y, norm_x = point_2d
            crop_px = int(norm_x * crop_w / 1000)
            crop_py = int(norm_y * crop_h / 1000)
            px = crop_x1 + crop_px
            py = crop_y1 + crop_py
            all_points.append({
                "object_label": label,
                "label": pt.get("label", ""),
                "role": pt.get("role", "interaction"),
                "reasoning": pt.get("reasoning", ""),
                "point_2d": point_2d,
                "px": px, "py": py,
                "crop_px": crop_px, "crop_py": crop_py,
            })
            print(f"    [{pt.get('role','?')}] ({norm_y},{norm_x}) "
                  f"→ crop({crop_px},{crop_py}) → full({px},{py})")

    print(f"\n  Total: {len(all_points)} points across {len(valid_objects)} objects")

    # Positions 구성 (pixel → world)
    print(f"\n{YELLOW}" + _log("Building positions...", step="Positions") + f"{RESET}")
    positions = _points_to_positions(all_points, robot_id=robot_id, camera=camera)

    # Fallback
    if fallback_positions:
        for name, info in positions.items():
            if info.get("_needs_world_coords") and name in fallback_positions:
                fb = fallback_positions[name]
                if fb is not None:
                    fb_pos = fb["position"] if isinstance(fb, dict) and "position" in fb else fb
                    info["position"] = list(fb_pos[:3])
                    info["gripper_offset"] = fb.get("gripper_offset", 0.01) if isinstance(fb, dict) else 0.01
                    info.pop("_needs_world_coords", None)

    for name, info in positions.items():
        pos = info["position"]
        status = "NEED WORLD" if info.get("_needs_world_coords") else "OK"
        print(f"    {name}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] [{status}]")

    # ── Turn Test: Waypoint Trajectory Prediction ──
    turn_test_overhead_waypoints = []
    turn_test_sideview_waypoints = []
    turn_test_resp = ""
    has_side_view = side_view_image and os.path.isfile(side_view_image)

    if len(all_points) >= 2:
        print(f"\n{YELLOW}" + _log("Waypoint Trajectory", step="TurnTest") + f"{RESET}")
        print(f"    All detected points: {len(all_points)}")
        for pt in all_points:
            print(f"      - [{pt['role']}] {pt['object_label']}: {pt['label']} @ ({pt['py']}, {pt['px']})")
        print(f"    Phase (from instruction): {instruction}")
        if has_side_view:
            print(f"    Side-view image: {side_view_image}")

        turn_msg = {
            "text": turn_test_waypoint_trajectory_prompt(
                instruction=instruction,
                phase=instruction,
                all_points=all_points,
                has_side_view=has_side_view,
            ),
        }
        if has_side_view:
            turn_msg["image_path"] = side_view_image

        turn_test_resp = gemini_chat_send(chat, gen_config, turn_msg,
            turn_label="Waypoint Trajectory")
        print(f"    {turn_test_resp[:300]}{'...' if len(turn_test_resp) > 300 else ''}")

        parsed_test = _parse_json_from_response(turn_test_resp)
        if parsed_test:
            # Log selected points
            selected = parsed_test.get("selected_points", [])
            if selected:
                print(f"    Selected points: {selected}")

            # Parse overhead waypoints
            oh_wps = parsed_test.get("overhead_waypoints", [])
            for wp in oh_wps:
                point_2d = wp.get("point_2d", [])
                if len(point_2d) != 2:
                    continue
                wy, wx = point_2d
                turn_test_overhead_waypoints.append({
                    "label": wp.get("label", ""),
                    "reasoning": wp.get("reasoning", ""),
                    "point_2d": point_2d,
                    "py": wy, "px": wx,
                })
                print(f"    [overhead] ({wy}, {wx}) — {wp.get('label', '')}")
            print(f"    Overhead waypoints: {len(turn_test_overhead_waypoints)}")

            # Parse side-view waypoints
            sv_wps = parsed_test.get("sideview_waypoints", [])
            for wp in sv_wps:
                point_2d = wp.get("point_2d", [])
                if len(point_2d) != 2:
                    continue
                wy, wx = point_2d
                turn_test_sideview_waypoints.append({
                    "label": wp.get("label", ""),
                    "reasoning": wp.get("reasoning", ""),
                    "point_2d": point_2d,
                    "py": wy, "px": wx,
                })
                print(f"    [sideview] ({wy}, {wx}) — {wp.get('label', '')}")
            print(f"    Side-view waypoints: {len(turn_test_sideview_waypoints)}")
        else:
            print(f"    [Warning] Failed to parse waypoints from response")
    else:
        print(f"\n{YELLOW}" + _log("Waypoint Trajectory — skipped (< 2 points)", step="TurnTest") + f"{RESET}")

    # ── Code Generation Turn ──
    print(f"\n{YELLOW}" + _log("Code Generation", step="CodeGen") + f"{RESET}")
    codegen_resp = gemini_chat_send(chat, gen_config,
        {"text": turn3_code_gen_prompt(instruction=instruction, robot_id=robot_id)},
        turn_label="Code Gen")
    code = extract_code_from_response(codegen_resp)
    assert code, "Failed to extract code from Code Gen response"

    # multi_turn_info (하위호환: execution_forward_and_reset.py가 사용하는 키 유지)
    turn2_compat = {
        "grasp_points": [
            {
                "object_name": pt["object_label"],
                "label": pt.get("label", ""),
                "role": pt["role"],
                "point_pixel": [
                    int(pt["py"] * 1000 / img_h),
                    int(pt["px"] * 1000 / img_w),
                ],
            }
            for pt in all_points
        ]
    }

    multi_turn_info = {
        "turn0_response": turn0_resp,
        "turn1_response": turn1_resp,
        "turn2_response": "\n".join(cr["response"] for cr in crop_responses),
        "turn3_response": codegen_resp,
        "turn1_parsed": valid_objects,
        "turn2_parsed": turn2_compat,
        # 신규 필드
        "detected_objects": valid_objects,
        "crop_responses": crop_responses,
        "all_points": all_points,
        "crop_dir": crop_dir,
        "turn_test_response": turn_test_resp,
        "turn_test_overhead_waypoints": turn_test_overhead_waypoints,
        "turn_test_sideview_waypoints": turn_test_sideview_waypoints,
        "side_view_image": side_view_image if has_side_view else None,
    }

    n_turns = 2 + len(valid_objects) + 1
    print(GRAY + "=" * line_width + RESET)
    print(LIGHT_GREEN + f"Crop-then-Point completed ({n_turns} turns).".center(line_width) + RESET)
    print(GRAY + "=" * line_width + RESET)

    return code, positions, multi_turn_info