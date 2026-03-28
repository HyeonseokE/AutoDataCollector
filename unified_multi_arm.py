#!/usr/bin/env python3
"""
UnifiedMultiArmPipeline — Bi-arm orchestrator

Same interface as ForwardAndResetPipeline but controls two arms:
  1. MultiArmSkills (left_arm + right_arm)
  2. Multi-turn VLM code generation (code_gen_lerobot/multi_arm/)
  3. MultiArmRecorder (ALOHA 12-axis recording)
  4. Judge (1 evaluation per episode — whole-task success/failure)
  5. Multi-arm reset code generation + execution

Detection is handled by multi-turn VLM pipeline (Turn 0~2), not Grounding DINO.

Usage (from execution_forward_and_reset.py):
    if len(robot_ids) > 1:
        pipeline = UnifiedMultiArmPipeline(robot_ids=[2, 3], ...)
        pipeline.run_multiple_episodes(...)
"""

import copy
import json
import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skills.multi_arm_skills import MultiArmSkills


# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET_COLOR = "\033[0m"
BOLD = "\033[1m"


class UnifiedMultiArmPipeline:
    """
    Unified bi-arm pipeline with the same interface as ForwardAndResetPipeline.

    Entry point routing:
        ROBOT_IDS=(2)   → ForwardAndResetPipeline (single-arm, existing)
        ROBOT_IDS=(2 3) → UnifiedMultiArmPipeline (this class)
    """

    def __init__(
        self,
        robot_ids: List[int],
        llm_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o",
        judge_timeout_ms: int = 5000,
        num_random_seeds: int = 1,
        verbose: bool = True,
        # Recording options
        record_dataset: bool = False,
        dataset_repo_id: Optional[str] = None,
        recording_fps: int = 30,
        resume_recording: bool = False,
        # Multi-turn options
        multi_turn: bool = False,
        cad_image_dirs: List[str] = None,
        side_view_image: str = None,
        codegen_model: str = None,
        task_type: str = "pick_place",
        reset_instruction: str = None,
        skip_turn_test: bool = False,
    ):
        assert len(robot_ids) >= 2, f"Multi-arm requires >= 2 robots, got {robot_ids}"
        self.robot_ids = robot_ids
        self.left_id = robot_ids[0]
        self.right_id = robot_ids[1]
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.judge_timeout_ms = judge_timeout_ms
        self.num_random_seeds = num_random_seeds
        self.verbose = verbose

        # Multi-turn
        self.multi_turn = multi_turn
        self.cad_image_dirs = cad_image_dirs or []
        self.side_view_image = side_view_image
        self.codegen_model = codegen_model
        self.task_type = task_type
        self.reset_instruction = reset_instruction or "move objects to their original positions"
        self.skip_turn_test = skip_turn_test
        self.multi_turn_info: Dict = {}
        self.reset_multi_turn_info: Dict = {}

        # Recording
        self.record_dataset = record_dataset
        self.dataset_repo_id = dataset_repo_id
        self.recording_fps = recording_fps
        self.resume_recording = resume_recording

        # State
        self.camera = None
        self.camera_manager = None
        self.multi_arm: Optional[MultiArmSkills] = None
        self.dataset_recorder = None

        # Episode tracking
        self.current_episode = 1
        self.total_episodes = 1
        self.current_phase = "Forward"
        self.instruction = ""

        # Positions tracking
        self.detected_positions: Dict = {}
        self.first_episode_positions: Optional[Dict] = None
        self._all_previous_seed_positions = []

        # Code cache
        self.cached_forward_code: Optional[str] = None
        self.cached_forward_keys: List[str] = []
        self.cached_reset_code: Optional[str] = None
        self.cached_reset_keys: List[str] = []

        # 레코딩 모드 초기화 (resume 모드는 run_multiple_episodes에서 초기화)
        if self.record_dataset and not self.resume_recording:
            self._init_recording()

    def _log(self, message: str, step: str = None) -> str:
        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"
        prefix = f"[{self.current_phase}][{ep_str}]"
        if step:
            prefix += f"[{step}]"
        return f"{prefix} {message}"

    # ─────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────

    def _init_multi_arm(self) -> bool:
        """Initialize MultiArmSkills with both robot configs.

        Note: connect()는 여기서 하지 않음.
        LLM 생성 코드가 skills.connect()를 직접 호출.
        RecordingContext가 활성화되어 있으면 connect() 시 양팔 모두 자동으로 콜백 획득.
        """
        left_config = f"robot_configs/robot/so101_robot{self.left_id}.yaml"
        right_config = f"robot_configs/robot/so101_robot{self.right_id}.yaml"

        self.multi_arm = MultiArmSkills(
            left_config=left_config,
            right_config=right_config,
            frame="base_link",
            verbose=self.verbose,
        )
        return True  # connect()는 LLM 코드에서 호출

    def _init_camera(self) -> bool:
        """Initialize camera for image capture.

        camera_manager가 있으면 거기서 realsense를 가져옴 (리소스 충돌 방지).
        없으면 직접 RealSense를 열음.
        """
        # camera_manager가 이미 있으면 그것의 realsense 사용
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                # feature_name 기준: shared realsense는 "top"으로 등록됨
                # "top" 우선, fallback으로 "realsense" (레거시 호환)
                for cam_name in ["top", "realsense"]:
                    try:
                        self.camera = self.camera_manager.get_camera(cam_name)
                        print(f"[MultiArm] Camera initialized: '{cam_name}' (from camera_manager)")
                        return True
                    except KeyError:
                        continue
            except Exception:
                pass

        # 직접 열기
        try:
            from object_detection.camera import RealSenseD435
            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            for _ in range(30):
                self.camera.get_frames()
            print(f"[MultiArm] Camera initialized (direct)")
            return True
        except Exception as e:
            print(f"{RED}[MultiArm] Camera init failed: {e}{RESET_COLOR}")
            return False

    def _init_recording(self):
        """Initialize recording with MultiArmRecorder (12-axis ALOHA format)."""
        if not self.record_dataset:
            return

        if not self.dataset_repo_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_repo_id = f"local/multi_arm_{timestamp}"

        try:
            from record_dataset.recorder import DatasetRecorder
            from record_dataset.config import (
                create_camera_manager_from_config,
                build_multi_arm_features,
                load_cameras_from_yaml,
            )
            from record_dataset.multi_arm_recorder import MultiArmRecorder

            print(f"\n[Recording] Initializing multi-arm dataset recorder...")
            print(f"  Repo ID: {self.dataset_repo_id}")

            # 1. 기존 dataset 존재 여부 미리 체크 (forward + reset)
            from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
            reset_repo_id = self.dataset_repo_id + "_reset"
            existing = []
            for rid in [self.dataset_repo_id, reset_repo_id]:
                ds_path = HF_LEROBOT_HOME / rid
                if ds_path.exists() and not self.resume_recording:
                    existing.append(str(ds_path))
            if existing:
                paths_str = "\n".join(f"     rm -rf {p}" for p in existing)
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Dataset already exists!\n"
                    f"========================================\n"
                    f"Paths:\n" + "\n".join(f"  - {p}" for p in existing) + "\n"
                    f"\n"
                    f"To continue, either:\n"
                    f"  1. Delete the existing datasets:\n"
                    f"{paths_str}\n"
                    f"  2. Use a different repo_id\n"
                    f"========================================"
                )

            # 2. Camera manager (YAML에서 동적 로드)
            if self.camera_manager is None:
                self.camera_manager = create_camera_manager_from_config()
                self.camera_manager.connect_all()
                print(f"[Recording] Cameras connected: {self.camera_manager.camera_names}")

            # 3. YAML에 enabled된 카메라가 모두 연결되었는지 검증
            cameras = load_cameras_from_yaml()
            enabled_cameras = [cam for cam in cameras if cam.enabled]
            connected_names = set(self.camera_manager.camera_names)
            expected_names = {cam.feature_name for cam in enabled_cameras}
            missing = expected_names - connected_names
            if missing:
                missing_details = []
                for cam in enabled_cameras:
                    if cam.feature_name in missing:
                        device = cam.get_device_path() or cam.serial_number or "unknown"
                        missing_details.append(f"  - {cam.feature_name} ({cam.type}, device={device})")
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Camera connection failed!\n"
                    f"========================================\n"
                    f"The following cameras are enabled in recording_config.yaml\n"
                    f"but failed to connect:\n"
                    + "\n".join(missing_details) + "\n"
                    f"\n"
                    f"To fix, either:\n"
                    f"  1. Connect the camera hardware and verify device path\n"
                    f"     (run: v4l2-ctl --list-devices)\n"
                    f"  2. Set 'enabled: false' for unavailable cameras in\n"
                    f"     pipeline_config/recording_config.yaml\n"
                    f"========================================"
                )

            # 4. Dataset recorder with 12-axis multi-arm features
            multi_arm_features = build_multi_arm_features(cameras=enabled_cameras)

            self.dataset_recorder = DatasetRecorder(
                repo_id=self.dataset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
                features=multi_arm_features,
            )
            print(f"[Recording] Recorder initialized (12-axis ALOHA format)")

            # 3. MultiArmRecorder will be created after skills.connect()
            # (needs multi_arm instance which is created later)
            self._multi_arm_recorder: Optional['MultiArmRecorder'] = None

        except AssertionError:
            raise
        except Exception as e:
            print(f"[Recording] Init failed: {e}")
            import traceback
            traceback.print_exc()
            self.record_dataset = False
            self.dataset_recorder = None

    def _start_episode_recording(self, task: str) -> None:
        """에피소드 레코딩 시작 (single-arm과 동일)."""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.start_episode(task=task)
                print(f"[Recording] Episode started: {task}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to start episode: {e}")

    def _end_episode_recording(self, discard: bool = False) -> None:
        """에피소드 레코딩 종료 (single-arm과 동일)."""
        if self.dataset_recorder and self.record_dataset:
            try:
                info = self.dataset_recorder.end_episode(discard=discard)
                if not discard:
                    print(f"[Recording] Episode saved: {info.get('num_frames', 0)} frames")
                else:
                    print(f"[Recording] Episode discarded")
            except Exception as e:
                print(f"[Recording] Warning: Failed to end episode: {e}")

    # ─────────────────────────────────────────────
    # Code Generation
    # ─────────────────────────────────────────────

    def _generate_forward_code(
        self,
        instruction: str,
        image_path: str,
    ) -> str:
        """Generate forward code via multi-turn VLM (single-arm 패턴 차용).

        lerobot_code_gen_multi_turn()를 직접 호출.
        Detection은 VLM Turn 1~2에서 내부적으로 처리됨.
        """
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen_multi_turn

        # depth 기반 3D 좌표 변환을 위해 카메라 전달
        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("top")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        code, mt_positions, mt_info = lerobot_code_gen_multi_turn(
            instruction=instruction,
            image_path=image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_ids[0],  # primary robot for coordinate transform
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            fallback_positions={},
            camera=active_camera,
            cad_image_dirs=self.cad_image_dirs,
            side_view_image=self.side_view_image,
            codegen_model=self.codegen_model,
            task_type=self.task_type,
            skip_turn_test=self.skip_turn_test,
            robot_ids=self.robot_ids,
        )

        # multi-turn 정보 저장
        self.multi_turn_info = mt_info

        # Build dual-arm positions: re-run pixel→world for each robot's calibration
        from code_gen_lerobot.code_gen_with_skill import _points_to_positions
        all_points = mt_info.get("all_points", [])
        valid_objects = mt_info.get("detected_objects", [])

        self.detected_positions = {
            "left_arm": _points_to_positions(all_points, robot_id=self.left_id, camera=active_camera, valid_objects=valid_objects),
            "right_arm": _points_to_positions(all_points, robot_id=self.right_id, camera=active_camera, valid_objects=valid_objects),
        }

        return code

    def _generate_reset_code(
        self,
        original_instruction: str,
        original_positions: Dict,
        current_state_image_path: str,
        initial_state_image_path: str,
    ) -> str:
        """Generate reset code via multi-turn VLM pipeline (same as forward).

        Turn 0~2 perception is reused from single-arm pipeline.
        CodeGen stage branches to multi-arm reset prompt.
        """
        from code_gen_lerobot.reset_execution.code_gen import lerobot_reset_code_gen_multi_turn

        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("top")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        # Multi-arm original_positions is {"left_arm": {obj: ...}, "right_arm": {obj: ...}}.
        # Reset perception only needs object labels (keys) — pick one arm's dict.
        # Codegen needs per-arm coordinates — pass original dual structure separately.
        is_dual = "left_arm" in original_positions or "right_arm" in original_positions
        if is_dual:
            one_arm_positions = original_positions.get("left_arm") or original_positions.get("right_arm") or {}
        else:
            one_arm_positions = original_positions

        reset_code, current_pos, target_pos, grippable, obstacles, reset_mt_info = lerobot_reset_code_gen_multi_turn(
            original_instruction=original_instruction,
            original_positions=one_arm_positions,
            original_positions_dual=original_positions if is_dual else None,
            current_state_image_path=current_state_image_path,
            initial_state_image_path=initial_state_image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_ids[0],
            camera=active_camera,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            codegen_model=self.codegen_model,
            robot_ids=self.robot_ids,
        )

        self.reset_multi_turn_info = reset_mt_info
        self._reset_current_positions = current_pos
        self._reset_target_positions = target_pos
        return reset_code

    # ─────────────────────────────────────────────
    # Code Execution
    # ─────────────────────────────────────────────

    def _build_exec_globals(self, positions: Dict, extra_globals: Dict = None) -> Dict:
        """Build the exec_globals dict for LLM-generated code.

        Single-arm 패턴과 동일: positions만 주입 + skills 인스턴스 주입.
        LLM 코드가 skills.connect() / skills.disconnect()를 직접 호출.
        """
        exec_globals = {
            "__name__": "__generated__",
            "skills": self.multi_arm,  # pre-created MultiArmSkills instance
            "positions": positions,
        }
        if extra_globals:
            exec_globals.update(extra_globals)
        return exec_globals

    def execute_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """Execute LLM-generated code (with recording support).

        Single-arm ForwardAndResetPipeline.execute_code() 패턴과 동일:
        - Recording 모드: RecordingContext 설정 → exec → 정리
        - Non-recording 모드: exec만 수행
        """
        try:
            exec_globals = self._build_exec_globals(positions, extra_globals)

            if self.record_dataset and self.dataset_recorder:
                return self._execute_code_with_recording(code, exec_globals)
            else:
                exec(code, exec_globals)
                if "execute_task" in exec_globals:
                    exec_globals["execute_task"]()
                elif "execute_reset_task" in exec_globals:
                    exec_globals["execute_reset_task"]()
                return True

        except AssertionError as e:
            error_msg = str(e)
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                print(f"\n{RED}[MultiArm] FATAL: Robot connection failed - {e}{RESET_COLOR}")
                raise
            print(f"{RED}[MultiArm] Code execution failed: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"{RED}[MultiArm] Code execution failed: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_code_with_recording(self, code: str, exec_globals: Dict) -> bool:
        """MultiArmRecorder를 사용한 12축 레코딩 포함 코드 실행.

        1. MultiArmRecorder 생성 + 각 arm에 action 콜백 주입
        2. recorder.start() → 50Hz 백그라운드 루프 시작
        3. exec(code) → 스킬 실행 시 action이 자동으로 MultiArmRecorder에 전달
        4. recorder.stop() → 정리
        """
        from record_dataset.multi_arm_recorder import MultiArmRecorder

        try:
            # Create MultiArmRecorder
            mar = MultiArmRecorder(
                multi_arm=self.multi_arm,
                recorder=self.dataset_recorder,
                camera_manager=self.camera_manager,
                target_fps=self.recording_fps,
                control_hz=50,
            )
            self._multi_arm_recorder = mar

            # Inject callbacks that update both state and action in MultiArmRecorder
            # This avoids concurrent serial port access (Issue 5)
            def make_callback(set_state_fn, set_action_fn):
                def callback(state, action):
                    set_state_fn(state)
                    set_action_fn(action)
                return callback

            self.multi_arm.left_arm.recording_callback = make_callback(mar.set_left_state, mar.set_left_action)
            self.multi_arm.right_arm.recording_callback = make_callback(mar.set_right_state, mar.set_right_action)

            # Skill info callbacks (bypass RecordingContext for multi-arm)
            self.multi_arm.left_arm.skill_info_callback = mar.set_left_skill_info
            self.multi_arm.right_arm.skill_info_callback = mar.set_right_skill_info

            # exec(code) defines execute_task() but doesn't call it yet
            exec(code, exec_globals)

            # Now skills.connect() will be called inside execute_task().
            # We hook into connect() completion by starting recorder after connect.
            # Override connect to start recorder after hardware init.
            original_connect = self.multi_arm.connect

            def connect_then_record():
                result = original_connect()
                # Initialize actions with current state (Issue 3: avoid None fallback)
                if self.multi_arm.left_arm.robot:
                    left_state = self.multi_arm.left_arm.robot.read_positions()
                    if left_state is not None:
                        mar.set_left_action(np.asarray(left_state, dtype=np.float32))
                if self.multi_arm.right_arm.robot:
                    right_state = self.multi_arm.right_arm.robot.read_positions()
                    if right_state is not None:
                        mar.set_right_action(np.asarray(right_state, dtype=np.float32))
                # Start recording AFTER robots are connected
                mar.start()
                print(f"[Recording] MultiArmRecorder started (12-axis, 50Hz → {self.recording_fps}fps)")
                return result

            self.multi_arm.connect = connect_then_record

            if "execute_task" in exec_globals:
                exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                exec_globals["execute_reset_task"]()

            # Restore original connect and clear callbacks
            self.multi_arm.connect = original_connect
            self.multi_arm.left_arm.skill_info_callback = None
            self.multi_arm.right_arm.skill_info_callback = None

            mar.stop()
            stats = mar.get_stats()
            print(f"[Recording] Recorded {stats['recorded_frames']} frames")

            return True

        except AssertionError as e:
            error_msg = str(e)
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                raise
            print(f"{RED}[MultiArm] Code execution failed: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"{RED}[MultiArm] Code execution failed: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if self._multi_arm_recorder and self._multi_arm_recorder.is_running:
                self._multi_arm_recorder.stop()
            self._multi_arm_recorder = None

    def _update_llm_cost_with_detect_usage(self, forward_dir: str) -> None:
        """Merge detect_objects token usage into llm_cost.json."""
        # Collect usage from both arms
        detect_turns = []
        if self.multi_arm:
            for arm in [self.multi_arm.left_arm, self.multi_arm.right_arm]:
                usage = getattr(arm, '_detect_token_usage', [])
                if usage:
                    detect_turns.extend(usage)
                    arm._detect_token_usage = []  # reset after collecting

        if not detect_turns:
            return

        cost_path = Path(forward_dir) / "llm_cost.json"
        try:
            # Load existing cost
            if cost_path.exists():
                with open(cost_path) as f:
                    llm_cost = json.load(f)
            else:
                llm_cost = {}

            # Build detect summary
            detect_summary = {
                "model": "gemini-3-flash-preview",
                "inference_time_s": round(sum(t.get("inference_time_s", 0) for t in detect_turns), 2),
                "input_tokens": sum(t.get("input_tokens", 0) for t in detect_turns),
                "output_tokens": sum(t.get("output_tokens", 0) for t in detect_turns),
                "total_tokens": sum(t.get("total_tokens", 0) for t in detect_turns),
                "num_calls": len(detect_turns),
                "turns": detect_turns,
            }
            llm_cost["detect_objects"] = detect_summary

            # Update total and move it to the end
            old_total = llm_cost.pop("total", {})
            for key in ["input_tokens", "output_tokens", "total_tokens"]:
                old_total[key] = old_total.get(key, 0) + detect_summary.get(key, 0)
            old_total["inference_time_s"] = round(
                old_total.get("inference_time_s", 0) + detect_summary["inference_time_s"], 2)
            llm_cost["total"] = old_total

            with open(cost_path, 'w') as f:
                json.dump(llm_cost, f, indent=2)
            print(f"  detect_objects cost merged into: {cost_path}")
            print(f"    detect calls: {len(detect_turns)}, tokens: in={detect_summary['input_tokens']}, out={detect_summary['output_tokens']}")

        except Exception as e:
            print(f"  [Warning] Failed to update llm_cost with detect usage: {e}")

    # ─────────────────────────────────────────────
    # Judge
    # ─────────────────────────────────────────────

    def _run_judge(
        self,
        instruction: str,
        initial_image: np.ndarray,
        final_image: np.ndarray,
        object_positions: Dict = None,
        executed_code: str = "",
    ) -> Dict:
        """Run judge evaluation (1 evaluation per episode, task-level)."""
        try:
            from judge import TaskJudge

            use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
            judge = TaskJudge(
                model=self.judge_model,
                verbose=self.verbose,
                use_server=use_server,
            )
            result = judge.judge(
                instruction=instruction,
                initial_image=initial_image,
                final_image=final_image,
                object_positions=object_positions or {},
                executed_code=executed_code,
            )
            return result
        except Exception as e:
            print(f"[MultiArm] Judge error: {e}")
            return {"prediction": "UNCERTAIN", "reasoning": str(e)}

    # ─────────────────────────────────────────────
    # Multi-turn artifact saving
    # ─────────────────────────────────────────────

    def _save_multi_turn_artifacts(self, forward_dir: str, initial_image=None) -> None:
        """Save multi-turn VLM artifacts (turn logs, LLM cost, crop images, visualizations).

        Replicates the save logic from single-arm ForwardAndResetPipeline.
        """
        mt_info = getattr(self, 'multi_turn_info', None)
        if not mt_info:
            return

        fwd = Path(forward_dir)
        import json as _json

        # 1. multi_turn_info.json
        mt_save = {
            "turn0_response": mt_info.get("turn0_response", ""),
            "turn1_response": mt_info.get("turn1_response", ""),
            "turn2_response": mt_info.get("turn2_response", ""),
            "turn3_response": mt_info.get("turn3_response", ""),
            "turn1_parsed": mt_info.get("turn1_parsed"),
            "turn1_sideview_parsed": mt_info.get("turn1_sideview_parsed"),
            "turn2_parsed": mt_info.get("turn2_parsed"),
            "detected_objects": mt_info.get("detected_objects"),
            "all_points": mt_info.get("all_points"),
            "crop_responses": mt_info.get("crop_responses"),
            "turn_test_response": mt_info.get("turn_test_response", ""),
            "turn_test_overhead_waypoints": mt_info.get("turn_test_overhead_waypoints"),
            "turn_test_sideview_waypoints": mt_info.get("turn_test_sideview_waypoints"),
        }
        with open(fwd / "multi_turn_info.json", 'w', encoding='utf-8') as f:
            _json.dump(mt_save, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Multi-turn info saved: {fwd / 'multi_turn_info.json'}")

        # 2. LLM cost
        llm_cost = mt_info.get("llm_cost")
        if llm_cost:
            with open(fwd / "llm_cost.json", 'w') as f:
                _json.dump({"phase": "forward", **llm_cost}, f, indent=2)
            print(f"  LLM cost saved: {fwd / 'llm_cost.json'}")

        # 3. Turn visualization images
        if initial_image is not None:
            t1_parsed = mt_info.get("turn1_parsed")
            t2_parsed = mt_info.get("turn2_parsed")

            try:
                from execution_forward_and_reset import ForwardAndResetPipeline
                # Borrow visualization methods (they are stateless)
                dummy = object.__new__(ForwardAndResetPipeline)

                if t1_parsed:
                    dummy._visualize_turn1(
                        initial_image.copy(), t1_parsed,
                        str(fwd / "turn1_detection.jpg"),
                        turn1_raw=mt_info.get("turn0_response", ""),
                    )

                if t2_parsed:
                    dummy._visualize_turn2(
                        initial_image.copy(), t1_parsed, t2_parsed,
                        str(fwd / "turn2_grasp_points.jpg"),
                    )

                oh_waypoints = mt_info.get("turn_test_overhead_waypoints")
                if oh_waypoints:
                    dummy._visualize_turn_test(
                        initial_image.copy(), t2_parsed, oh_waypoints,
                        str(fwd / "turn_test_overhead_waypoints.jpg"),
                    )
            except Exception as e:
                print(f"  [Warning] Turn visualization failed: {e}")

        # 4. Crop images
        crop_dir = mt_info.get("crop_dir")
        if crop_dir and os.path.isdir(crop_dir):
            import shutil
            for fname in sorted(os.listdir(crop_dir)):
                if fname.endswith(('.jpg', '.png')):
                    shutil.copy2(os.path.join(crop_dir, fname), os.path.join(forward_dir, fname))
            print(f"  Crop images saved to: {forward_dir}")

        # 5. Turn text logs
        self._save_turn_logs(forward_dir, mt_info)

    def _save_turn_logs(self, forward_dir: str, mt_info: Dict) -> None:
        """Save per-turn text logs."""
        fwd = Path(forward_dir)

        # Turn 0
        turn0_raw = mt_info.get("turn0_response", "")
        if turn0_raw:
            lines = ["=" * 60, "Turn 0: Scene Understanding", "=" * 60, "", turn0_raw]
            (fwd / "turn0_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 0 log saved: {fwd / 'turn0_log.txt'}")

        # Turn 1
        turn1_raw = mt_info.get("turn1_response", "")
        if turn1_raw:
            lines = ["=" * 60, "Turn 1: Bounding Box Detection", "=" * 60, "", turn1_raw]
            t1_parsed = mt_info.get("turn1_parsed")
            if t1_parsed:
                lines.append("\n[Parsed Summary]")
                obj_list = t1_parsed if isinstance(t1_parsed, list) else t1_parsed.get("objects", []) if isinstance(t1_parsed, dict) else []
                for obj in obj_list:
                    name = obj.get("label") or obj.get("name", "?")
                    box = obj.get("box_2d") or obj.get("bbox_pixel", "N/A")
                    lines.append(f"  - {name}: bbox={box}")
            (fwd / "turn1_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 1 log saved: {fwd / 'turn1_log.txt'}")

        # Turn 2+
        all_points = mt_info.get("all_points", [])
        crop_responses = mt_info.get("crop_responses", [])
        if all_points or crop_responses:
            lines = ["=" * 60, "Turn 2+: Crop-then-Point", "=" * 60, ""]
            for cr in crop_responses:
                lines.append(f"--- Crop: {cr.get('label', '?')} ---")
                lines.append(cr.get("response", ""))
                lines.append("")
            lines.append("[Parsed Points Summary]")
            for pt in all_points:
                obj = pt.get("object_label", "?")
                label = pt.get("label", "?")
                role = pt.get("role", "?")
                px, py = pt.get("px", 0), pt.get("py", 0)
                reasoning = pt.get("reasoning", "")
                lines.append(f"  - {obj}: {label} ({role}) pixel=({px},{py})")
                if reasoning:
                    lines.append(f"    reasoning: {reasoning}")
            (fwd / "turn2_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 2+ log saved: {fwd / 'turn2_log.txt'}")

        # Turn 3
        turn3_raw = mt_info.get("turn3_response", "")
        if turn3_raw:
            lines = ["=" * 60, "Turn 3: Code Generation", "=" * 60, "", turn3_raw]
            (fwd / "turn3_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 3 log saved: {fwd / 'turn3_log.txt'}")

    # ─────────────────────────────────────────────
    # Capture helpers
    # ─────────────────────────────────────────────

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a color frame from shared camera."""
        if self.camera:
            try:
                color, _ = self.camera.get_frames()
                return color
            except Exception as e:
                print(f"[MultiArm] Frame capture failed: {e}")
        return None

    # ─────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────

    def run(
        self,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        use_timestamp_subdir: bool = True,
        skip_reset: bool = False,
        reset_target_positions: Optional[Dict] = None,
        pre_reset_callback=None,
        post_judge_callback=None,
    ) -> Dict:
        """
        Run full bi-arm pipeline for one episode.

        Same interface as ForwardAndResetPipeline.run().

        Flow:
            1. Initialize arms + camera + recording
            2. Capture image → multi-turn VLM code generation (detection inside VLM)
            3. Execute code with both arms
            4. Judge evaluation
            5. Reset (optional)
        """
        self.instruction = instruction

        # Result dir
        if save_dir is None:
            save_dir = "results"
        if use_timestamp_subdir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = str(Path(save_dir) / timestamp)
        else:
            result_dir = str(Path(save_dir))
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        forward_dir = str(Path(result_dir) / "forward")
        reset_dir = str(Path(result_dir) / "reset")
        Path(forward_dir).mkdir(parents=True, exist_ok=True)
        Path(reset_dir).mkdir(parents=True, exist_ok=True)

        result = {
            'forward': {'positions': {}, 'code': '', 'execution_success': False},
            'judge': {'prediction': 'UNCERTAIN', 'reasoning': ''},
            'reset': {'mode': 'original', 'current_positions': {}, 'target_positions': {},
                      'code': '', 'execution_success': False},
            'reset_judge': {'prediction': 'UNCERTAIN', 'reasoning': '', 'reset_mode': 'original'},
            'saved_files': {},
        }

        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"
        print(f"\n{CYAN}{'='*70}{RESET_COLOR}")
        print(f"{CYAN}{BOLD}[{ep_str}] Multi-Arm Forward + Reset Pipeline{RESET_COLOR}")
        print(f"{CYAN}{'='*70}{RESET_COLOR}")

        try:
            # ══════════════════════════════════════
            # PHASE 0: INITIALIZATION
            # ══════════════════════════════════════
            self.current_phase = "Init"

            # Initialize arms (connect는 LLM 코드에서)
            if self.multi_arm is None:
                print(f"\n{YELLOW}" + self._log("Initializing both arms...") + f"{RESET_COLOR}")
                if not self._init_multi_arm():
                    print(f"{RED}[Error] Failed to initialize robot arms{RESET_COLOR}")
                    return result

            # Initialize recording first (camera_manager 생성)
            # → 그 다음 camera는 camera_manager에서 가져옴 (리소스 충돌 방지)
            if self.record_dataset and self.dataset_recorder is None:
                self._init_recording()

            # Initialize camera (camera_manager가 있으면 그것 사용)
            if self.camera is None:
                print(f"{YELLOW}" + self._log("Initializing camera...") + f"{RESET_COLOR}")
                if not self._init_camera():
                    print(f"{RED}[Error] Camera initialization failed{RESET_COLOR}")
                    return result

            # Camera를 양팔 skills에 주입 (detect_objects에서 사용)
            if self.multi_arm is not None and self.camera is not None:
                self.multi_arm.left_arm.camera = self.camera
                self.multi_arm.right_arm.camera = self.camera

            # ══════════════════════════════════════
            # PHASE 1: FORWARD EXECUTION
            # ══════════════════════════════════════
            self.current_phase = "Forward"
            print(f"\n{GREEN}{BOLD}" + self._log("FORWARD EXECUTION") + f"{RESET_COLOR}")
            print(f"{GREEN}{'-'*70}{RESET_COLOR}")

            # Step 1: Capture initial image
            print(f"\n{YELLOW}" + self._log("Capturing initial image...", step="Step 1/4") + f"{RESET_COLOR}")
            initial_image = self._capture_frame()
            if initial_image is not None:
                cv2.imwrite(str(Path(forward_dir) / "initial_state.jpg"), initial_image)
                print(f"  Image captured ({initial_image.shape[1]}x{initial_image.shape[0]})")

            # Step 2: Code generation (detection is handled by VLM Turn 1~2 internally)
            print(f"\n{YELLOW}" + self._log(f"Generating forward code via multi-turn VLM...", step="Step 2/4") + f"{RESET_COLOR}")
            image_path = str(Path(forward_dir) / "initial_state.jpg")
            code = self._generate_forward_code(instruction, image_path)
            result['forward']['code'] = code
            result['forward']['positions'] = self.detected_positions

            # Store first episode positions
            if self.first_episode_positions is None and self.detected_positions:
                self.first_episode_positions = copy.deepcopy(self.detected_positions)

            # Save generated code
            with open(Path(forward_dir) / "generated_code.py", 'w') as f:
                f.write(code)

            # Save multi-turn info (turn logs, LLM cost, crop images, visualizations)
            self._save_multi_turn_artifacts(forward_dir, initial_image)

            # Step 3: Execute code
            print(f"\n{YELLOW}" + self._log("Executing forward code...", step="Step 3/4") + f"{RESET_COLOR}")

            # Set execution dir for skill_detect_results logging
            import builtins as _builtins
            _builtins._current_execution_dir = forward_dir

            # Start episode recording (RecordingContext 기반 — single-arm과 동일)
            if self.record_dataset and self.dataset_recorder:
                self._start_episode_recording(instruction)

            # Turn 2 라벨을 추출하여 detect_objects 재검출 시 재사용
            point_labels = {}
            if self.detected_positions:
                for arm_key in ["left_arm", "right_arm"]:
                    arm_pos = self.detected_positions.get(arm_key, {})
                    for obj_name, obj_info in arm_pos.items():
                        if obj_info and "points" in obj_info and obj_name not in point_labels:
                            point_labels[obj_name] = list(obj_info["points"].keys())
            if point_labels:
                self.multi_arm._point_labels = point_labels

            execution_success = self.execute_code(code, self.detected_positions)
            result['forward']['execution_success'] = execution_success

            # End episode recording
            if self.record_dataset and self.dataset_recorder:
                self._end_episode_recording()

            # Save pixel_moves_overlay
            try:
                for arm in [self.multi_arm.left_arm, self.multi_arm.right_arm]:
                    if hasattr(arm, 'pixel_move_log') and arm.pixel_move_log:
                        grasp_img_path = str(Path(forward_dir) / "turn2_grasp_points.jpg")
                        if os.path.isfile(grasp_img_path):
                            from execution_forward_and_reset import ForwardAndResetPipeline
                            _dummy = object.__new__(ForwardAndResetPipeline)
                            _dummy._visualize_pixel_moves(
                                grasp_img_path, arm.pixel_move_log,
                                str(Path(forward_dir) / "pixel_moves_overlay.jpg"),
                            )
                        break
            except Exception as e:
                print(f"  [Warning] pixel move visualization failed: {e}")

            # Update llm_cost.json with detect_objects token usage
            self._update_llm_cost_with_detect_usage(forward_dir)

            # Save execution_context.json
            exec_ctx = {
                "instruction": instruction,
                "object_positions": self._make_serializable(self.detected_positions or {}),
                "generated_code": code,
                "execution_success": execution_success,
                "robot_ids": self.robot_ids,
            }
            with open(Path(forward_dir) / "execution_context.json", 'w') as f:
                json.dump(exec_ctx, f, indent=2, default=str)
            print(f"  Execution context saved: {Path(forward_dir) / 'execution_context.json'}")

            # Capture final image
            time.sleep(1.0)
            final_image = self._capture_frame()
            if final_image is not None:
                cv2.imwrite(str(Path(forward_dir) / "final_state.jpg"), final_image)

            # Step 4: Judge evaluation
            print(f"\n{YELLOW}" + self._log("Running judge evaluation...", step="Step 4/4") + f"{RESET_COLOR}")
            if initial_image is not None and final_image is not None:
                judge_result = self._run_judge(
                    instruction, initial_image, final_image,
                    object_positions=self.detected_positions or {},
                    executed_code=code,
                )
                result['judge'] = judge_result
                prediction = judge_result.get('prediction', 'UNCERTAIN')
                color = GREEN if prediction == 'TRUE' else RED
                print(f"  Judge: {color}{prediction}{RESET_COLOR}")

            # Post-judge callback
            if post_judge_callback:
                post_judge_callback(result)

            # Cache successful code
            if execution_success and result['judge'].get('prediction') == 'TRUE':
                self.cached_forward_code = code
                self.cached_forward_keys = self._extract_position_keys(code)

            # Save forward result + judge
            with open(Path(forward_dir) / "result.json", 'w') as f:
                json.dump(self._make_serializable(result['forward']), f, indent=2)
            with open(Path(forward_dir) / "judge_result.json", 'w') as f:
                json.dump(result['judge'], f, indent=2, default=str)

            # Save judge visualization image
            try:
                from judge.visualize import create_result_image
                judge_vis = create_result_image(
                    initial_image, final_image,
                    result['judge'].get('prediction', 'UNCERTAIN'),
                    result['judge'].get('reasoning', ''),
                    instruction,
                )
                if judge_vis is not None:
                    cv2.imwrite(str(Path(forward_dir) / "judge_result.jpg"), judge_vis)
            except Exception:
                pass

            # Save forward_log.txt (콘솔 로그 요약)
            fwd_log_lines = [
                f"Instruction: {instruction}",
                f"Robot IDs: {self.robot_ids}",
                f"Execution success: {execution_success}",
                f"Judge: {result['judge'].get('prediction', 'UNCERTAIN')}",
                f"Judge reasoning: {result['judge'].get('reasoning', '')}",
            ]
            (Path(forward_dir) / "forward_log.txt").write_text("\n".join(fwd_log_lines), encoding="utf-8")

            # ══════════════════════════════════════
            # PHASE 2: RESET EXECUTION
            # ══════════════════════════════════════
            if not skip_reset:
                self.current_phase = "Reset"
                print(f"\n{GREEN}{BOLD}" + self._log("RESET EXECUTION") + f"{RESET_COLOR}")
                print(f"{GREEN}{'-'*70}{RESET_COLOR}")

                # Pre-reset callback (seed transition)
                if pre_reset_callback:
                    reset_target = pre_reset_callback(current_positions=positions)
                    if reset_target:
                        reset_target_positions = reset_target

                # Target positions: use provided or first episode positions
                target_positions = reset_target_positions or self.first_episode_positions or {}

                result['reset']['target_positions'] = target_positions

                # Capture current scene image for reset (= initial state for reset)
                print(f"\n{YELLOW}" + self._log("Capturing reset scene...") + f"{RESET_COLOR}")
                reset_image = self._capture_frame()
                if reset_image is not None:
                    cv2.imwrite(str(Path(reset_dir) / "current_state.jpg"), reset_image)
                    cv2.imwrite(str(Path(reset_dir) / "initial_state.jpg"), reset_image)
                    print(f"  Reset scene captured")
                reset_image_path = str(Path(reset_dir) / "current_state.jpg")

                # Save reset positions
                reset_positions_save = {
                    "current_positions": self._make_serializable(self.detected_positions or {}),
                    "target_positions": self._make_serializable(target_positions),
                }
                with open(Path(reset_dir) / "positions.json", 'w') as f:
                    json.dump(reset_positions_save, f, indent=2, default=str)

                # Generate reset code via multi-turn VLM pipeline
                print(f"\n{YELLOW}" + self._log("Generating reset code (multi-turn VLM)...") + f"{RESET_COLOR}")
                initial_image_path = str(Path(forward_dir) / "initial_state.jpg")
                reset_code = self._generate_reset_code(
                    original_instruction=instruction,
                    original_positions=target_positions,
                    current_state_image_path=reset_image_path,
                    initial_state_image_path=initial_image_path,
                )
                result['reset']['code'] = reset_code

                with open(Path(reset_dir) / "generated_code.py", 'w') as f:
                    f.write(reset_code)

                # Execute reset
                print(f"\n{YELLOW}" + self._log("Executing reset code...") + f"{RESET_COLOR}")

                # Set execution dir for skill_detect_results logging
                _builtins._current_execution_dir = reset_dir

                if self.record_dataset and self.dataset_recorder:
                    self._start_episode_recording(self.reset_instruction)

                # current_positions / target_positions를 globals로 주입 (싱글암 패턴과 동일)
                reset_current = getattr(self, '_reset_current_positions', {})
                reset_target = target_positions
                reset_success = self.execute_code(reset_code, {}, extra_globals={
                    "current_positions": reset_current,
                    "target_positions": reset_target,
                })
                result['reset']['execution_success'] = reset_success

                if self.record_dataset and self.dataset_recorder:
                    self._end_episode_recording()

                # Capture reset final image
                time.sleep(1.0)
                reset_final_image = self._capture_frame()
                if reset_final_image is not None:
                    cv2.imwrite(str(Path(reset_dir) / "final_state.jpg"), reset_final_image)

                # Save reset result + judge
                with open(Path(reset_dir) / "result.json", 'w') as f:
                    json.dump(self._make_serializable(result['reset']), f, indent=2)

                # Reset judge (if images available)
                if reset_image is not None and reset_final_image is not None:
                    try:
                        from judge import ResetJudge
                        use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
                        reset_judge = ResetJudge(model=self.judge_model, verbose=self.verbose, use_server=use_server)
                        reset_judge_result = reset_judge.judge(
                            reset_mode="original",
                            current_positions=self._make_serializable(self.detected_positions or {}),
                            target_positions=self._make_serializable(target_positions),
                            initial_image=reset_image,
                            final_image=reset_final_image,
                            executed_code=reset_code,
                            original_instruction=instruction,
                        )
                        result['reset_judge'] = reset_judge_result
                        with open(Path(reset_dir) / "reset_judge_result.json", 'w') as f:
                            json.dump(reset_judge_result, f, indent=2, default=str)
                    except Exception as e:
                        print(f"  [Warning] Reset judge failed: {e}")

                # Save reset_log.txt
                reset_log_lines = [
                    f"Instruction: {instruction}",
                    f"Reset success: {reset_success}",
                    f"Reset judge: {result.get('reset_judge', {}).get('prediction', 'UNCERTAIN')}",
                ]
                (Path(reset_dir) / "reset_log.txt").write_text("\n".join(reset_log_lines), encoding="utf-8")

        except AssertionError:
            raise
        except Exception as e:
            print(f"\n{RED}[MultiArm] Pipeline error: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()

        return result

    # ─────────────────────────────────────────────
    # Multi-episode execution
    # ─────────────────────────────────────────────

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
    ) -> Dict:
        """
        Run multiple episodes sequentially.
        Same interface as ForwardAndResetPipeline.run_multiple_episodes().
        """
        self.total_episodes = num_episodes
        self.instruction = instruction

        # Create session directory
        if save_dir is None:
            save_dir = "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(save_dir) / f"session_{timestamp}")
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        # Results accumulator
        all_results = {
            'session_dir': session_dir,
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'robot_ids': self.robot_ids,
            'episodes': [],
            'summary': {
                'forward_success': 0,
                'forward_judge_true': 0,
                'reset_success': 0,
                'reset_judge_true': 0,
            },
        }

        episodes_per_seed = max(1, num_episodes // max(1, self.num_random_seeds))
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        print(f"\n{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"{MAGENTA}{BOLD}  MULTI-ARM SESSION: {num_episodes} Episodes (robots {self.robot_ids})  {RESET_COLOR}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Save Dir: {session_dir}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")

        for episode_idx in range(num_episodes):
            episode_num = episode_idx + 1
            batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
            self.current_episode = episode_num

            print(f"\n{CYAN}{'='*70}{RESET_COLOR}")
            print(f"{CYAN}{BOLD}  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1})  {RESET_COLOR}")
            print(f"{CYAN}{'='*70}{RESET_COLOR}")

            episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")
            reset_target = seed_positions[batch_index]

            try:
                result = self.run(
                    instruction=instruction,
                    objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir,
                    use_timestamp_subdir=False,
                    skip_reset=skip_reset,
                    reset_target_positions=reset_target,
                )

                # Store first episode positions for seed[0]
                if seed_positions[0] is None and self.first_episode_positions is not None:
                    seed_positions[0] = copy.deepcopy(self.first_episode_positions)

                # Update summary
                s = all_results['summary']
                if result['forward']['execution_success']:
                    s['forward_success'] += 1
                if result['judge'].get('prediction') == 'TRUE':
                    s['forward_judge_true'] += 1
                if result['reset']['execution_success']:
                    s['reset_success'] += 1
                if result.get('reset_judge', {}).get('prediction') == 'TRUE':
                    s['reset_judge_true'] += 1

                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': result,
                    'success': result['forward']['execution_success'],
                })

            except AssertionError:
                raise
            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET_COLOR}")
                import traceback
                traceback.print_exc()
                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': None,
                    'success': False,
                    'error': str(e),
                })

            time.sleep(2)

        # Finalize
        self._finalize_session(all_results, session_dir)
        return all_results

    def resume_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        resume_session_dir: str,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
    ) -> Dict:
        """
        Resume a previous session. Same interface as ForwardAndResetPipeline.
        For now, delegates to run_multiple_episodes (full re-run).
        TODO: Implement proper batch-level resume logic.
        """
        print(f"{YELLOW}[MultiArm] Resume not yet fully implemented, running fresh session{RESET_COLOR}")
        return self.run_multiple_episodes(
            num_episodes=num_episodes,
            instruction=instruction,
            objects=objects,
            detection_timeout=detection_timeout,
            visualize_detection=visualize_detection,
            save_dir=save_dir or str(Path(resume_session_dir).parent),
            skip_reset=skip_reset,
        )

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    @staticmethod
    def _extract_position_keys(code: str) -> List[str]:
        """Extract positions['xxx'] pattern keys from code."""
        import re
        pattern = r'positions\[["\'](.+?)["\']\]'
        return list(dict.fromkeys(re.findall(pattern, code)))

    def _finalize_session(self, all_results: Dict, session_dir: str):
        """Print summary and save session results."""
        n = all_results['num_episodes']
        s = all_results['summary']

        print(f"\n{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"{MAGENTA}{BOLD}  SESSION SUMMARY  {RESET_COLOR}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")
        print(f"  Episodes: {n}")
        print(f"  Robot IDs: {self.robot_ids}")
        if n > 0:
            rate = s['forward_judge_true'] / n * 100
            color = GREEN if rate >= 80 else YELLOW if rate >= 50 else RED
            print(f"  Forward Judge TRUE: {color}{s['forward_judge_true']}/{n} ({rate:.1f}%){RESET_COLOR}")
            print(f"  Forward Success: {s['forward_success']}/{n}")
            print(f"  Reset Success: {s['reset_success']}/{n}")
        print(f"{MAGENTA}{'='*70}{RESET_COLOR}")

        # Save summary
        summary_path = Path(session_dir) / "session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self._make_serializable(all_results), f, indent=2)
        print(f"  Results saved: {summary_path}")

        # Finalize recording
        if self.dataset_recorder:
            try:
                self.dataset_recorder.finalize()
                print(f"  Dataset finalized: {self.dataset_recorder.repo_id}")
            except Exception as e:
                print(f"  Dataset finalization error: {e}")

        # Camera cleanup: camera_manager가 소유한 카메라는 stop하지 않음
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
            except Exception:
                pass
            self.camera_manager = None
            self.camera = None  # camera_manager가 관리하므로 참조만 해제
        elif self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None

    @staticmethod
    def _make_serializable(obj):
        """Make result dict JSON-serializable."""
        if isinstance(obj, dict):
            return {k: UnifiedMultiArmPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [UnifiedMultiArmPipeline._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, bytes):
            return "<bytes>"
        return obj
