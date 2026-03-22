#!/usr/bin/env python3
"""
VLM Multi-Turn Detect & Move Test Script
Gemini VLM 멀티턴 프롬프팅으로 물체를 감지하고 해당 위치로 로봇을 이동하는 테스트

forward_and_reset.sh 파이프라인의 VLM 감지 부분을 간소화하여
독립적으로 테스트할 수 있는 스크립트.

Pipeline:
  Turn 0: Scene understanding (이미지 + instruction)
  Turn 1: BBox detection (물체 위치 검출)
  Turn 2+: Crop-then-Point (각 물체별 정밀 포인팅)
  → Pixel→World 좌표 변환
  → IK로 로봇 이동

Usage:
    python scripts/test_vlm_detect_and_move.py --robot 2 --instruction "pick up the chocolate pie"
    python scripts/test_vlm_detect_and_move.py --robot 2 --instruction "grab the red cup" --dry-run
    python scripts/test_vlm_detect_and_move.py --robot 2 --instruction "move the block" --save-image /tmp/vlm_detect.jpg
    python scripts/test_vlm_detect_and_move.py --robot 2 --instruction "pick up the fork" --model gemini-2.5-flash
"""

import sys
import json
import re
import argparse
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from object_detection.camera.realsense import RealSenseD435
from object_detection.localization.coordinate_transform import CoordinateTransformer
from skills.skills_lerobot import LeRobotSkills
from code_gen_lerobot.llm_utils.gemini import gemini_chat_start, gemini_chat_send
from code_gen_lerobot.forward_execution.turn0_prompt import turn0_scene_understanding_prompt
from code_gen_lerobot.forward_execution.turn1_prompt import turn1_detect_task_relevant_objects_prompt
from code_gen_lerobot.forward_execution.turn2_prompt import turn2_crop_pointing_prompt


CROP_PADDING = 25  # bbox padding (0-1000 scale)
Z_MAX = 0.15       # 15cm max height for table objects
Z_DEFAULT = 0.02   # default z when out of range


def parse_json_from_response(response: str):
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

    # { ... } or [ ... ] 직접 찾기
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    start = response.find('[')
    end = response.rfind(']')
    if start != -1 and end != -1:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def vlm_detect_objects(image_path: str, instruction: str, model: str = "gemini-2.0-flash"):
    """
    VLM 멀티턴으로 물체를 감지하고 critical points를 반환.

    Returns:
        list of dict: [{"object_label": str, "label": str, "role": str, "px": int, "py": int}, ...]
    """
    # System prompt
    system_prompt = (
        "You are an expert robotic manipulation assistant. "
        "You analyze scenes from overhead camera views and identify objects "
        "and their critical manipulation points for a robotic arm."
    )

    print(f"\n  VLM Model: {model}")
    chat, gen_config = gemini_chat_start(model, system_prompt=system_prompt)

    # ── Turn 0: Scene Understanding ──
    print(f"\n  [Turn 0] Scene Understanding...")
    turn0_resp = gemini_chat_send(chat, gen_config, {
        "text": turn0_scene_understanding_prompt(instruction, has_cad=False),
        "image_path": image_path,
    }, turn_label="Turn 0")
    print(f"    {turn0_resp[:200]}{'...' if len(turn0_resp) > 200 else ''}")

    # ── Turn 1: BBox Detection ──
    print(f"\n  [Turn 1] BBox Detection...")
    turn1_resp = gemini_chat_send(chat, gen_config, {
        "text": turn1_detect_task_relevant_objects_prompt(has_side_view=False),
        "image_path": image_path,
    }, turn_label="Turn 1")
    print(f"    {turn1_resp[:300]}{'...' if len(turn1_resp) > 300 else ''}")

    turn1_data = parse_json_from_response(turn1_resp)
    if not turn1_data:
        print("    ERROR: Turn 1 JSON 파싱 실패")
        return []

    # Parse object list
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

    if not valid_objects:
        print("    ERROR: 유효한 bbox 없음")
        return []

    # Load image for cropping
    full_img = cv2.imread(image_path)
    if full_img is None:
        print(f"    ERROR: 이미지 읽기 실패: {image_path}")
        return []
    img_h, img_w = full_img.shape[:2]

    # ── Turn 2+: Crop-then-Point ──
    all_points = []
    crop_dir = tempfile.mkdtemp(prefix="vlm_crop_")

    for i, obj in enumerate(valid_objects):
        label = obj["label"]
        ymin, xmin, ymax, xmax = obj["box_2d"]

        # Padding + clamp (0-1000 scale)
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

        if crop_h < 5 or crop_w < 5:
            print(f"    [{label}] crop이 너무 작음 ({crop_w}x{crop_h}), 건너뜀")
            continue

        # Save crop
        safe_label = label.replace(" ", "_").replace("/", "_")
        crop_path = f"{crop_dir}/crop_{safe_label}.jpg"
        cv2.imwrite(crop_path, crop_img)

        print(f"\n  [Turn 2/{i}] Crop-then-Point: '{label}' ({crop_w}x{crop_h})")

        # Send crop + pointing prompt
        resp = gemini_chat_send(chat, gen_config, {
            "text": turn2_crop_pointing_prompt(label, has_side_view=False),
            "image_path": crop_path,
        }, turn_label=f"Crop: {label}")
        print(f"    {resp[:200]}{'...' if len(resp) > 200 else ''}")

        # Parse critical_points
        parsed = parse_json_from_response(resp)
        if not parsed:
            print(f"    [{label}] JSON 파싱 실패")
            continue

        if "critical_points" in parsed:
            points = parsed["critical_points"]
        elif "overhead_critical_points" in parsed:
            points = parsed["overhead_critical_points"]
        else:
            print(f"    [{label}] critical_points 키 없음")
            continue

        for pt in points:
            point_2d = pt.get("point_2d", [])
            if len(point_2d) != 2:
                continue
            norm_y, norm_x = point_2d
            crop_px = int(norm_x * crop_w / 1000)
            crop_py = int(norm_y * crop_h / 1000)
            px = crop_x1 + crop_px
            py = crop_y1 + crop_py

            role = pt.get("role", "interaction")
            pt_label = pt.get("label", label)

            all_points.append({
                "object_label": label,
                "label": pt_label,
                "role": role,
                "px": px,
                "py": py,
                "reasoning": pt.get("reasoning", ""),
            })
            print(f"    [{role}] '{pt_label}' → full({px},{py})")

    print(f"\n  Total: {len(all_points)} points across {len(valid_objects)} objects")
    return all_points


def main():
    parser = argparse.ArgumentParser(description="VLM multi-turn detect and move robot")
    parser.add_argument("--robot", type=int, default=2, choices=[2, 3], help="Robot number")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Task instruction (e.g., 'pick up the chocolate pie')")
    parser.add_argument("--approach-height", type=float, default=0.20,
                        help="Approach height above object (meters)")
    parser.add_argument("--dry-run", action="store_true", help="Detect only, don't move robot")
    parser.add_argument("--save-image", type=str, default=None, help="Save detection image to path")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name")
    args = parser.parse_args()

    robot_config = f"robot_configs/robot/so101_robot{args.robot}.yaml"
    pix2world_file = "robot_configs/pix2world_matrices/pix2world_transform_data.npz"

    # ========== 1. Camera ==========
    print("\n[1/5] Camera initialization...")
    camera = RealSenseD435()
    camera.start()
    intrinsics = camera.get_intrinsics()
    print(f"  Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Warm up
    for _ in range(10):
        camera.get_frames()

    color, depth = camera.get_frames()
    if color is None:
        print("ERROR: Failed to capture image")
        camera.stop()
        return 1
    print(f"  Captured: {color.shape[1]}x{color.shape[0]}")

    # Save captured image for VLM
    capture_path = "/tmp/vlm_detect_capture.jpg"
    cv2.imwrite(capture_path, color)

    # ========== 2. VLM Multi-Turn Detection ==========
    print(f"\n[2/5] VLM Multi-Turn Detection: '{args.instruction}'")
    all_points = vlm_detect_objects(capture_path, args.instruction, model=args.model)

    if not all_points:
        print("  No objects detected!")
        camera.stop()
        return 1

    # Pick the first grasp/interaction point
    grasp_points = [p for p in all_points if p["role"] == "grasp"]
    if not grasp_points:
        grasp_points = [p for p in all_points if p["role"] == "interaction"]
    if not grasp_points:
        grasp_points = all_points  # fallback: use any point

    best = grasp_points[0]
    px, py = best["px"], best["py"]
    print(f"\n  Selected: '{best['label']}' role={best['role']} pixel=({px},{py})")

    # ========== 3. Pixel → World ==========
    print(f"\n[3/5] Pixel → World coordinate transform...")
    transformer = CoordinateTransformer(pix2world_file)
    transformer.set_camera_intrinsics(intrinsics)

    depth_m = camera.get_depth_at_pixel(px, py, depth)
    print(f"  Pixel: ({px}, {py}), depth={depth_m:.3f}m")

    if depth_m > 0 and transformer.transform_matrix_3d is not None:
        wx, wy, wz = transformer.pixel_depth_to_world(px, py, depth_m)
        print(f"  World (3D): ({wx:.2f}, {wy:.2f}, {wz:.2f}) cm")
    else:
        wx, wy, wz = transformer.pixel_to_world_2d(px, py)
        print(f"  World (2D): ({wx:.2f}, {wy:.2f}, {wz:.2f}) cm")

    # cm → meters
    world_pos = np.array([wx / 100.0, wy / 100.0, wz / 100.0])

    # Z clamping
    if world_pos[2] < 0 or world_pos[2] > Z_MAX:
        print(f"  [Z-FIX] z={world_pos[2]*100:.1f}cm out of range, clamping to {Z_DEFAULT*100:.0f}cm")
        world_pos[2] = Z_DEFAULT

    print(f"  World position: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}] m")

    # Save detection visualization
    if args.save_image:
        vis = color.copy()
        for pt in all_points:
            cx, cy = pt["px"], pt["py"]
            role_color = (0, 0, 255) if pt["role"] == "grasp" else (0, 255, 0)
            cv2.circle(vis, (cx, cy), 6, role_color, -1)
            cv2.putText(vis, f"{pt['label']} ({pt['role']})", (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, role_color, 1)
        # Highlight selected point
        cv2.circle(vis, (px, py), 12, (255, 0, 255), 2)
        cv2.putText(vis, "TARGET", (px + 15, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imwrite(args.save_image, vis)
        print(f"  Saved: {args.save_image}")

    camera.stop()

    # ========== 4. Dry-run check ==========
    if args.dry_run:
        print(f"\n[DRY RUN] Would move robot{args.robot} to:")
        print(f"  World:    [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]")
        print(f"  Approach: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {args.approach_height:.4f}]")
        return 0

    # ========== 5. Move Robot ==========
    print(f"\n[4/5] Connecting robot{args.robot}...")
    skills = LeRobotSkills(
        robot_config=robot_config,
        frame="world",
        verbose=True,
    )
    skills.connect()

    try:
        print(f"\n[5/5] Moving to detected object...")
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach above object
        approach_pos = [world_pos[0], world_pos[1], args.approach_height]
        print(f"\n  Approach: {approach_pos}")
        target_name = best["object_label"]
        success = skills.move_to_position(
            approach_pos,
            target_name=target_name,
            skill_description=f"approach above {target_name}",
        )

        if success:
            print(f"\n  Successfully reached above '{target_name}'!")
            # Descend to object
            descend_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.02]
            print(f"  Descend: {descend_pos}")
            skills.move_to_position(
                descend_pos,
                target_name=target_name,
                skill_description=f"descend to {target_name}",
            )
        else:
            print(f"\n  Failed to reach approach position")

        # Return to initial
        skills.move_to_initial_state()

    finally:
        skills.disconnect()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
