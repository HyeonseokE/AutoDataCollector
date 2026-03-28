#!/usr/bin/env python3
"""
Multi-Arm Skills API

Provides synchronized bi-arm control by wrapping two LeRobotSkills instances.
Supports parallel execution with ThreadPoolExecutor and "wait" semantics
for single-arm-only moves.

Usage:
    from skills.multi_arm_skills import MultiArmSkills

    multi = MultiArmSkills(
        left_config="robot_configs/robot/so101_robot2.yaml",
        right_config="robot_configs/robot/so101_robot3.yaml",
    )
    multi.connect()

    # Both arms move simultaneously
    multi.move_to_position(left_arm=[0.15, -0.10, 0.20], right_arm=[0.15, 0.10, 0.20])

    # Only right arm moves, left stays
    multi.move_to_position(left_arm="wait", right_arm=[0.15, 0.10, 0.05])

    multi.disconnect()
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from skills.skills_lerobot import LeRobotSkills


# Sentinel value for "no movement" on one arm
WAIT = "wait"


class MultiArmSkills:
    """
    Bi-arm skill controller wrapping two LeRobotSkills instances.

    robot_ids[0] → left_arm, robot_ids[1] → right_arm.

    Design decisions (D-2, D-3 from plan):
    - move_to_position blocks until BOTH arms finish (or timeout).
    - When one arm is "wait", its action = current state (hold position).
    - Skill features are split: left_skill.* / right_skill.* recorded separately.
    """

    def __init__(
        self,
        left_config: str = "robot_configs/robot/so101_robot2.yaml",
        right_config: str = "robot_configs/robot/so101_robot3.yaml",
        frame: str = "world",
        movement_duration: float = 3.0,
        use_compensation: bool = True,
        use_deceleration: bool = True,
        verbose: bool = True,
        pick_offset: float = 0.015,
        recording_callback=None,
        camera=None,
    ):
        """
        Args:
            left_config: Robot YAML config path for left arm (robot_ids[0]).
            right_config: Robot YAML config path for right arm (robot_ids[1]).
            recording_callback: Set to None to disable internal per-arm recording.
                Multi-arm recording is handled externally by MultiArmRecorder.
            camera: Shared RealSense camera instance (both arms share one camera).
        """
        self.verbose = verbose

        # Create LeRobotSkills instances with recording disabled
        # (MultiArmRecorder handles unified 12-axis recording externally)
        self.left_arm = LeRobotSkills(
            robot_config=left_config,
            frame=frame,
            movement_duration=movement_duration,
            use_compensation=use_compensation,
            use_deceleration=use_deceleration,
            verbose=verbose,
            pick_offset=pick_offset,
            recording_callback=recording_callback,
            camera=camera,
        )

        self.right_arm = LeRobotSkills(
            robot_config=right_config,
            frame=frame,
            movement_duration=movement_duration,
            use_compensation=use_compensation,
            use_deceleration=use_deceleration,
            verbose=verbose,
            pick_offset=pick_offset,
            recording_callback=recording_callback,
            camera=camera,
        )

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="multi_arm")

    def _log(self, message: str):
        if self.verbose:
            print(f"[MultiArm] {message}")

    # ─────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect both arms in parallel. Returns True if both succeed."""
        self._log("Connecting both arms...")
        # Re-create executor if it was shut down by a previous disconnect()
        if self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="multi_arm")
        futures = {
            self._executor.submit(self.left_arm.connect): "left_arm",
            self._executor.submit(self.right_arm.connect): "right_arm",
        }
        results = {}
        for future in as_completed(futures):
            arm_name = futures[future]
            try:
                results[arm_name] = future.result()
            except Exception as e:
                self._log(f"  {arm_name} connect failed: {e}")
                results[arm_name] = False

        ok = all(results.values())
        if ok:
            self._log("Both arms connected successfully")
        else:
            self._log(f"Connection results: {results}")
        return ok

    def disconnect(self):
        """Disconnect both arms."""
        self._log("Disconnecting both arms...")
        for arm, name in [(self.left_arm, "left_arm"), (self.right_arm, "right_arm")]:
            try:
                arm.disconnect()
            except Exception as e:
                self._log(f"  {name} disconnect error: {e}")
        self._executor.shutdown(wait=False)

    # ─────────────────────────────────────────────
    # Parallel execution helper
    # ─────────────────────────────────────────────

    def _run_both(
        self,
        left_fn,
        right_fn,
        left_args: tuple = (),
        right_args: tuple = (),
        left_kwargs: dict = None,
        right_kwargs: dict = None,
        description: str = "action",
    ) -> Dict[str, bool]:
        """
        Execute left_fn and right_fn in parallel.
        Blocks until both complete. Returns {"left": result, "right": result}.
        """
        left_kwargs = left_kwargs or {}
        right_kwargs = right_kwargs or {}

        futures = {}
        if left_fn is not None:
            futures[self._executor.submit(left_fn, *left_args, **left_kwargs)] = "left"
        if right_fn is not None:
            futures[self._executor.submit(right_fn, *right_args, **right_kwargs)] = "right"

        results = {"left": True, "right": True}
        for future in as_completed(futures):
            side = futures[future]
            try:
                results[side] = future.result()
            except Exception as e:
                self._log(f"  {side}_arm {description} failed: {e}")
                results[side] = False

        return results

    @staticmethod
    def _is_wait(value) -> bool:
        """Check if value is the 'wait' sentinel."""
        return isinstance(value, str) and value.lower() == "wait"

    # ─────────────────────────────────────────────
    # Bi-arm movement skills
    # ─────────────────────────────────────────────

    def move_to_position(
        self,
        left_arm="wait",
        right_arm="wait",
        left_duration: Optional[float] = None,
        right_duration: Optional[float] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Move arms simultaneously. Pass "wait" to skip one arm.

        Args:
            left_arm: [x,y,z] target for left arm, or "wait" to hold.
            right_arm: [x,y,z] target for right arm, or "wait" to hold.

        Returns:
            {"left": bool, "right": bool} success status.
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.move_to_position
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.move_to_position

        skip_msg = []
        if left_fn is None:
            skip_msg.append("left=wait")
        if right_fn is None:
            skip_msg.append("right=wait")
        if skip_msg:
            self._log(f"move_to_position: {', '.join(skip_msg)}")

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "position": left_arm,
                "duration": left_duration,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "position": right_arm,
                "duration": right_duration,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="move_to_position",
        )

    def pick_object(
        self,
        left_arm="wait",
        right_arm="wait",
        left_object_name: Optional[str] = None,
        right_object_name: Optional[str] = None,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Execute pick on arms. Pass "wait" to skip one arm.

        Args:
            left_arm: Object position [x,y,z] for left arm, or "wait".
            right_arm: Object position [x,y,z] for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_pick_object
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_pick_object

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "object_position": left_arm,
                "object_name": left_object_name,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "object_position": right_arm,
                "object_name": right_object_name,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="pick_object",
        )

    def place_object(
        self,
        left_arm="wait",
        right_arm="wait",
        left_is_table: bool = True,
        right_is_table: bool = True,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Execute place on arms. Pass "wait" to skip one arm.

        Args:
            left_arm: Place position [x,y,z] for left arm, or "wait".
            right_arm: Place position [x,y,z] for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_place_object
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_place_object

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "place_position": left_arm,
                "is_table": left_is_table,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "place_position": right_arm,
                "is_table": right_is_table,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="place_object",
        )

    def move_to_pixel(
        self,
        left_arm="wait",
        right_arm="wait",
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Move arms to positions specified by normalized pixel coordinates [y, x] (0–1000).
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] normalized coordinates for left arm, or "wait".
            right_arm: [y, x] normalized coordinates for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.move_to_pixel
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.move_to_pixel

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "pixel": left_arm,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "pixel": right_arm,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="move_to_pixel",
        )

    def place_at_pixel(
        self,
        left_arm="wait",
        right_arm="wait",
        left_is_table: bool = True,
        right_is_table: bool = True,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Place at positions specified by normalized pixel coordinates [y, x] (0–1000).
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] normalized coordinates for left arm, or "wait".
            right_arm: [y, x] normalized coordinates for right arm, or "wait".
        """
        left_fn = None
        right_fn = None

        if not self._is_wait(left_arm):
            left_fn = self.left_arm.execute_place_at_pixel
        if not self._is_wait(right_arm):
            right_fn = self.right_arm.execute_place_at_pixel

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs={
                "pixel": left_arm,
                "is_table": left_is_table,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            } if left_fn else {},
            right_kwargs={
                "pixel": right_arm,
                "is_table": right_is_table,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            } if right_fn else {},
            description="place_at_pixel",
        )

    def gripper_control(
        self,
        left_arm: str = "wait",
        right_arm: str = "wait",
        left_duration: float = 1.5,
        right_duration: float = 1.5,
        left_ratio: float = 1.0,
        right_ratio: float = 1.0,
        left_skill_description: Optional[str] = None,
        right_skill_description: Optional[str] = None,
        left_verification_question: Optional[str] = None,
        right_verification_question: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Control grippers. Actions: "open", "close", "wait".

        Args:
            left_arm: "open", "close", or "wait" for left arm.
            right_arm: "open", "close", or "wait" for right arm.
            left_ratio / right_ratio: Open ratio for "open" action (0.0-1.0).
        """
        left_fn = None
        right_fn = None
        left_kwargs = {}
        right_kwargs = {}

        if left_arm.lower() == "open":
            left_fn = self.left_arm.gripper_open
            left_kwargs = {
                "duration": left_duration,
                "ratio": left_ratio,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            }
        elif left_arm.lower() == "close":
            left_fn = self.left_arm.gripper_close
            left_kwargs = {
                "duration": left_duration,
                "skill_description": left_skill_description,
                "verification_question": left_verification_question,
            }

        if right_arm.lower() == "open":
            right_fn = self.right_arm.gripper_open
            right_kwargs = {
                "duration": right_duration,
                "ratio": right_ratio,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            }
        elif right_arm.lower() == "close":
            right_fn = self.right_arm.gripper_close
            right_kwargs = {
                "duration": right_duration,
                "skill_description": right_skill_description,
                "verification_question": right_verification_question,
            }

        return self._run_both(
            left_fn=left_fn,
            right_fn=right_fn,
            left_kwargs=left_kwargs,
            right_kwargs=right_kwargs,
            description="gripper_control",
        )

    # ─────────────────────────────────────────────
    # Convenience: both arms to known poses
    # ─────────────────────────────────────────────

    def move_to_initial_state(self) -> Dict[str, bool]:
        """Move both arms to their initial (home) positions simultaneously."""
        self._log("Moving both arms to initial state...")
        return self._run_both(
            left_fn=self.left_arm.move_to_initial_state,
            right_fn=self.right_arm.move_to_initial_state,
            description="move_to_initial_state",
        )

    def move_to_free_state(self) -> Dict[str, bool]:
        """Move both arms to their free (parking) positions simultaneously."""
        self._log("Moving both arms to free state...")
        return self._run_both(
            left_fn=self.left_arm.move_to_free_state,
            right_fn=self.right_arm.move_to_free_state,
            description="move_to_free_state",
        )

    # ─────────────────────────────────────────────
    # Subtask labeling (for recording)
    # ─────────────────────────────────────────────

    def set_subtask(self, description: str) -> None:
        """Set sub-task label for recording. Delegates to left_arm's set_subtask."""
        self.left_arm.set_subtask(description)

    def clear_subtask(self) -> None:
        """Clear sub-task label. Delegates to left_arm's clear_subtask."""
        self.left_arm.clear_subtask()

    def detect_objects(self, queries: list, timeout: float = 5.0, point_labels: dict = None) -> dict:
        """Re-detect objects and return positions in both arm frames.

        Uses left_arm to run detection (shared camera), then converts
        pixel coordinates through each arm's pix2robot calibration.

        Args:
            queries: 검출할 객체 이름 리스트
            timeout: 검출 타임아웃
            point_labels: 물체별 포인트 라벨 딕셔너리 (Turn 2 라벨 재사용)
                         {"red block": ["grasp center", "top surface center"], ...}

        Returns:
            {"left_arm": {obj: {"position": [...], ...}, ...},
             "right_arm": {obj: {"position": [...], ...}, ...}}
        """
        # 저장된 point_labels가 있으면 자동 사용 (Turn 2 라벨 재사용)
        if point_labels is None:
            point_labels = getattr(self, '_point_labels', None)

        # Run detection via left_arm (camera + VLM)
        raw = self.left_arm.detect_objects(queries, timeout=timeout, point_labels=point_labels)

        # Build dual-arm result by re-converting pixel coords per arm
        dual = {"left_arm": {}, "right_arm": {}}
        for obj_name, info in raw.items():
            if info is None:
                dual["left_arm"][obj_name] = None
                dual["right_arm"][obj_name] = None
                continue

            pixel = info.get("pixel")
            depth_m = info.get("depth_m")

            pixel_points = info.get("_pixel_points", {})

            for arm_key, arm in [("left_arm", self.left_arm), ("right_arm", self.right_arm)]:
                if arm.pix2robot is not None and pixel is not None:
                    px, py = int(pixel[0]), int(pixel[1])
                    pos = arm.pix2robot.pixel_to_robot(px, py, depth_m=depth_m)
                    # 각 포인트를 해당 포인트의 pixel 좌표로 변환
                    arm_points = {}
                    for pt_label, pt_pos in info.get("points", {}).items():
                        pt_pixel = pixel_points.get(pt_label)
                        if pt_pixel is not None:
                            arm_points[pt_label] = arm.pix2robot.pixel_to_robot(
                                int(pt_pixel[0]), int(pt_pixel[1]), depth_m=depth_m)
                        else:
                            arm_points[pt_label] = arm.pix2robot.pixel_to_robot(px, py, depth_m=depth_m)
                    dual[arm_key][obj_name] = {
                        "position": pos,
                        "points": arm_points,
                        "pixel": pixel,
                        "bbox_px": info.get("bbox_px"),
                    }
                else:
                    dual[arm_key][obj_name] = info.copy()

        return dual

    # ─────────────────────────────────────────────
    # State access helpers (for recording)
    # ─────────────────────────────────────────────

    def read_left_positions(self):
        """Read current left arm joint positions (6-axis normalized)."""
        if self.left_arm.robot:
            return self.left_arm.robot.read_positions()
        return None

    def read_right_positions(self):
        """Read current right arm joint positions (6-axis normalized)."""
        if self.right_arm.robot:
            return self.right_arm.robot.read_positions()
        return None

    def get_left_gripper_pos(self) -> float:
        """Get left arm current gripper target position."""
        return self.left_arm.current_gripper_pos

    def get_right_gripper_pos(self) -> float:
        """Get right arm current gripper target position."""
        return self.right_arm.current_gripper_pos
