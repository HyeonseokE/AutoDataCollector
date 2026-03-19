#!/usr/bin/env python3
"""
Forward + Reset Integrated Pipeline

Forward Execution → Judge (Evaluation) → Reset Execution 통합 파이프라인

Usage:
    # 전체 파이프라인 실행
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" "blue box"

    # Reset 건너뛰기
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --skip-reset

    # 결과 저장
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --save results/

    # LeRobot 데이터셋 레코딩 모드
    python execution_forward_and_reset.py -i "pick up the red cup" -o "red cup" --record --dataset-repo-id "user/my_dataset"
"""

import argparse
import io
import json
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class TeeLogger:
    """
    터미널 출력을 파일과 stdout 모두에 기록하는 로거.

    Usage:
        logger = TeeLogger("output.txt")
        logger.start()
        print("Hello, World!")  # stdout과 파일 모두에 기록
        logger.stop()
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.original_stdout = None
        self.buffer = io.StringIO()

    def write(self, text):
        """stdout과 버퍼 모두에 기록"""
        if self.original_stdout:
            self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        """버퍼 플러시"""
        if self.original_stdout:
            self.original_stdout.flush()

    def start(self):
        """로깅 시작"""
        self.original_stdout = sys.stdout
        self.buffer = io.StringIO()
        sys.stdout = self

    def stop(self):
        """로깅 종료 및 파일 저장"""
        if self.original_stdout:
            sys.stdout = self.original_stdout

        # 로그 내용 정리 (ANSI 색상 코드 제거)
        log_content = self.buffer.getvalue()

        # ANSI 이스케이프 시퀀스 제거
        import re
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_content = ansi_escape.sub('', log_content)

        # 적당한 위치에 줄바꿈 추가 (가독성 향상)
        clean_content = self._format_log(clean_content)

        # 파일 저장
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(clean_content)

        self.original_stdout = None
        return self.filepath

    def _format_log(self, content: str) -> str:
        """로그 내용 포맷팅 (가독성 향상)"""
        # 주요 섹션 앞에 빈 줄 추가
        formatted = content

        # 섹션 구분자 패턴들
        section_patterns = [
            r'(\[PHASE \d\])',
            r'(\[Step \d+/\d+\])',
            r'(={50,})',
            r'(-{40,})',
            r'(Moving to)',
            r'(Gripper:)',
            r'(PICK AND PLACE)',
            r'(EXECUTE PICK)',
            r'(EXECUTE PLACE)',
            r'(ROTATE 90)',
            r'(Error:)',
            r'(WARNING:)',
        ]

        return formatted

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))


class ForwardAndResetPipeline:
    """Forward → Judge → Reset 통합 파이프라인"""

    def __init__(
        self,
        robot_id: int = 3,
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
    ):
        """
        초기화

        Args:
            robot_id: 로봇 번호 (2 또는 3)
            llm_model: 코드 생성용 LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
            judge_model: Judge VLM 모델
            judge_timeout_ms: Judge UI 타임아웃 (밀리초)
            num_random_seeds: 배치 수 (1=초기 위치 유지, N>1=N종류 랜덤 배치)
            verbose: 상세 출력 여부
            record_dataset: LeRobot 데이터셋 레코딩 활성화
            dataset_repo_id: 데이터셋 저장 경로 (예: "user/my_dataset")
            recording_fps: 레코딩 FPS (기본: 30)
            multi_turn: True면 crop-then-point 멀티턴 LLM 코드 생성 사용
            cad_image_dirs: CAD 참조 이미지 디렉토리 리스트 (옵션)
        """
        self.robot_id = robot_id
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.judge_timeout_ms = judge_timeout_ms
        self.num_random_seeds = num_random_seeds
        self.verbose = verbose

        # Multi-turn options
        self.multi_turn = multi_turn
        self.cad_image_dirs = cad_image_dirs or []
        self.side_view_image = side_view_image
        self.codegen_model = codegen_model
        self.multi_turn_info: Dict = {}

        # Recording options
        self.record_dataset = record_dataset
        self.dataset_repo_id = dataset_repo_id
        self.recording_fps = recording_fps
        self.resume_recording = resume_recording
        self.dataset_recorder = None
        self.recording_skills_wrapper = None
        self.camera_manager = None  # MultiCameraManager for recording

        # 상태 저장
        self.camera = None
        self.initial_image: Optional[np.ndarray] = None
        self.final_image: Optional[np.ndarray] = None
        self.detection_image: Optional[np.ndarray] = None
        self.detected_positions: Dict = {}
        self.extended_detections: Dict = {}
        self.generated_spec: Dict = {}
        self.generated_code: str = ""
        self.instruction: str = ""

        # 이미지 해상도 저장 (Judge용)
        self.initial_image_resolution: Optional[Tuple[int, int]] = None  # (width, height)
        self.final_image_resolution: Optional[Tuple[int, int]] = None

        # Reset 관련 상태
        self.reset_initial_image: Optional[np.ndarray] = None
        self.reset_final_image: Optional[np.ndarray] = None
        self.reset_initial_resolution: Optional[Tuple[int, int]] = None
        self.reset_final_resolution: Optional[Tuple[int, int]] = None
        self.reset_code: str = ""

        # Forward initial image path (for reset multi-turn)
        self.forward_initial_image_path: Optional[str] = None

        # Context 저장용
        self.execution_context: Dict = {}

        # Episode tracking for logging
        self.current_episode: int = 1
        self.total_episodes: int = 1
        self.current_phase: str = "Forward"  # "Forward" or "Reset"

        # First episode positions for "original" reset mode
        # Stores the first episode's detection result to avoid cumulative drift
        self.first_episode_positions: Optional[Dict] = None

        # 과거 모든 배치 위치 누적 (랜덤 위치 생성 시 겹침 방지)
        self._all_previous_seed_positions = []

        # Code reuse cache: 성공한 코드를 캐싱하여 이후 에피소드에서 재사용
        self.cached_forward_code: Optional[str] = None
        self.cached_forward_keys: List[str] = []  # 캐싱 코드가 참조하는 position keys
        self.cached_reset_code: Optional[str] = None
        self.cached_reset_keys: List[str] = []

        # 레코딩 모드 초기화
        if self.record_dataset:
            self._init_recording()

    def _log(self, message: str, step: str = None, tag: str = None) -> str:
        """
        통일된 로그 포맷 생성

        Args:
            message: 출력할 메시지
            step: 단계 정보 (예: "Step 1/5")
            tag: 추가 태그 (예: "Validation", "Recording")

        Returns:
            포맷된 로그 문자열

        Examples:
            _log("Starting execution")
            # -> [Forward][01/50] Starting execution

            _log("Detecting objects", step="Step 1/5")
            # -> [Forward][01/50][Step 1/5] Detecting objects

            _log("Checking workspace", tag="Validation")
            # -> [Forward][01/50][Validation] Checking workspace
        """
        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"
        prefix = f"[{self.current_phase}][{ep_str}]"

        if step:
            prefix += f"[{step}]"
        if tag:
            prefix += f"[{tag}]"

        return f"{prefix} {message}"

    @staticmethod
    def _extract_position_keys(code: str) -> List[str]:
        """캐싱된 코드에서 positions['xxx'] 패턴의 key 추출"""
        import re
        pattern = r'positions\[["\'](.+?)["\']\]'
        return list(dict.fromkeys(re.findall(pattern, code)))

    def _can_reuse_code(self, cached_code: Optional[str], cached_keys: List[str],
                        new_positions: Dict) -> bool:
        """캐싱된 코드를 재사용할 수 있는지 확인 (key 일치 검사)

        cached_keys가 비어있으면 코드가 positions를 참조하지 않는 것이므로
        (좌표 하드코딩 등) 무조건 재사용 가능.
        """
        if cached_code is None:
            return False
        if not cached_keys:
            return True
        return set(cached_keys) <= set(new_positions.keys())

    def initialize_camera(self) -> bool:
        """카메라 초기화 (camera_manager가 있으면 그것을 사용)"""
        # camera_manager가 이미 RealSense를 가지고 있으면 재사용
        if self.camera_manager:
            if not self.camera_manager.is_connected:
                try:
                    self.camera_manager.connect_all()
                except Exception as e:
                    print(f"[Pipeline] Camera manager reconnect failed: {e}")
            if self.camera_manager.is_connected:
                try:
                    self.camera = self.camera_manager.get_camera("realsense")
                    if self.verbose:
                        print("[Pipeline] Camera initialized (from camera_manager)")
                    return True
                except KeyError:
                    pass

        # Fallback: 직접 RealSense 연결
        try:
            from object_detection.camera import RealSenseD435

            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            for _ in range(30):
                self.camera.get_frames()
            if self.verbose:
                print("[Pipeline] Camera initialized (direct)")
            return True
        except Exception as e:
            print(f"[Pipeline] Camera initialization failed: {e}")
            return False

    def shutdown_camera(self) -> None:
        """카메라 종료 (camera_manager 소유 카메라는 참조만 해제)"""
        if self.camera:
            # camera_manager가 소유한 카메라면 stop하지 않음 (manager가 관리)
            if self.camera_manager:
                try:
                    cm_cam = self.camera_manager.get_camera("realsense")
                    if self.camera is cm_cam:
                        self.camera = None
                        if self.verbose:
                            print("[Pipeline] Camera reference cleared (managed by camera_manager)")
                        return
                except (KeyError, Exception):
                    pass
            self.camera.stop()
            self.camera = None
            if self.verbose:
                print("[Pipeline] Camera shutdown")

    def _init_recording(self) -> None:
        """LeRobot 데이터셋 레코딩 초기화 (멀티 카메라 지원)"""
        if not self.record_dataset:
            return

        if not self.dataset_repo_id:
            # 기본 repo_id 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_repo_id = f"local/cap_dataset_{timestamp}"

        try:
            from record_dataset import DatasetRecorder
            from record_dataset.config import create_camera_manager_from_config

            print(f"\n[Recording] Initializing multi-camera dataset recorder...")
            print(f"  Repo ID: {self.dataset_repo_id}")
            print(f"  FPS: {self.recording_fps}")

            # 1. 카메라 매니저 초기화 (YAML에서 동적 로드)
            print(f"[Recording] Loading camera configuration...")
            self.camera_manager = create_camera_manager_from_config()

            # 2. 카메라 연결
            print(f"[Recording] Connecting cameras...")
            self.camera_manager.connect_all()
            print(f"[Recording] Cameras connected: {self.camera_manager.camera_names}")

            # 3. 레코더 초기화 (YAML에서 features 동적 생성)
            self.dataset_recorder = DatasetRecorder(
                repo_id=self.dataset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
            )
            print(f"[Recording] Recorder initialized successfully")
            print(f"[Recording] Features: {list(self.dataset_recorder.features.keys())}")

        except ImportError as e:
            print(f"[Recording] Warning: Failed to import record_dataset: {e}")
            print(f"[Recording] Dataset recording will be disabled")
            self.record_dataset = False
            self.dataset_recorder = None
            self.camera_manager = None
        except AssertionError:
            # Dataset already exists - 파이프라인 완전 종료
            raise
        except Exception as e:
            print(f"[Recording] Warning: Failed to initialize recorder: {e}")
            import traceback
            traceback.print_exc()
            self.record_dataset = False
            self.dataset_recorder = None
            self.camera_manager = None

    def _start_episode_recording(self, task: str) -> None:
        """에피소드 레코딩 시작"""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.start_episode(task=task)
                print(f"[Recording] Episode started: {task}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to start episode: {e}")

    def _end_episode_recording(self, discard: bool = False) -> None:
        """에피소드 레코딩 종료"""
        if self.dataset_recorder and self.record_dataset:
            try:
                info = self.dataset_recorder.end_episode(discard=discard)
                if not discard:
                    print(f"[Recording] Episode saved: {info.get('num_frames', 0)} frames")
                else:
                    print(f"[Recording] Episode discarded")
            except Exception as e:
                print(f"[Recording] Warning: Failed to end episode: {e}")

    def _finalize_recording(self) -> None:
        """데이터셋 레코딩 완료 및 카메라 연결 해제"""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.finalize()
                print(f"\n[Recording] Dataset finalized at: {self.dataset_recorder._dataset.root}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to finalize dataset: {e}")

        # 멀티 카메라 연결 해제
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
                print(f"[Recording] Camera manager disconnected")
            except Exception as e:
                print(f"[Recording] Warning: Failed to disconnect cameras: {e}")
            self.camera_manager = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 캡처 (camera 또는 camera_manager 사용)"""
        try:
            # 1순위: camera_manager의 realsense 카메라 사용
            if self.camera_manager and self.camera_manager.is_connected:
                try:
                    realsense = self.camera_manager.get_camera("realsense")
                    color, _ = realsense.get_frames()
                    return color
                except (KeyError, Exception):
                    pass  # fallback to self.camera

            # 2순위: self.camera 사용
            if self.camera is not None:
                color, _ = self.camera.get_frames()
                return color

            return None
        except Exception as e:
            print(f"[Pipeline] Frame capture failed: {e}")
            return None

    def run_detection(
        self,
        queries: list,
        timeout: float = 10.0,
        visualize: bool = False,
    ) -> Dict:
        """객체 검출 수행 (통합된 run_realtime_detection 사용)

        Recording 카메라가 있으면 공유하여 리소스 충돌 방지.
        """
        from run_detect import run_realtime_detection

        # 카메라 공유 (리소스 충돌 방지)
        # 우선순위: 1. self.camera (initialize_camera로 생성된 것)
        #          2. camera_manager의 카메라
        #          3. None (run_realtime_detection에서 자체 생성)
        external_camera = None
        if self.camera is not None:
            external_camera = self.camera
            print("  [Detection] Using pipeline camera")
        elif self.camera_manager and self.camera_manager.is_connected:
            try:
                external_camera = self.camera_manager.get_camera("realsense")
                print("  [Detection] Using shared camera from recording")
            except KeyError:
                print("  [Detection] No realsense camera in manager, using internal camera")

        extended_results, last_frame, vis_image = run_realtime_detection(
            queries=queries,
            timeout=timeout,
            unit="m",
            return_last_frame=True,
            return_extended=True,
            visualize=visualize,
            robot_id=self.robot_id,
            external_camera=external_camera,
        )

        if last_frame is not None:
            self.initial_image = last_frame

        if vis_image is not None:
            self.detection_image = vis_image

        # Store extended info for later use
        # Note: Workspace filtering is done at detection level (run_detect.py)
        self.extended_detections = extended_results

        # Return extended format: {name: {"position": [x,y,z], ...}}
        return extended_results

    def generate_forward_code(
        self,
        instruction: str,
        positions: Dict,
        image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
        canonical_point_labels: Dict[str, List[str]] = None,
    ) -> str:
        """Forward 코드 생성

        Args:
            instruction: 자연어 목표
            positions: Extended format {name: {"position": [x,y,z], ...}}
            image_path: 초기 이미지 경로 (multi-turn 모드에서 필수)
            skip_codegen: True면 검출만 수행, 코드 생성 스킵 (multi-turn 전용)
            canonical_labels: 코드 재사용 시 강제할 라벨 목록 (multi-turn 전용)
            canonical_point_labels: 코드 재사용 시 강제할 point 라벨 {obj: [labels]}

        Returns:
            생성된 Python 코드 문자열 (skip_codegen=True면 빈 문자열)
        """
        if self.multi_turn:
            return self._generate_forward_code_multi_turn(
                instruction, positions, image_path,
                skip_codegen=skip_codegen,
                canonical_labels=canonical_labels,
                canonical_point_labels=canonical_point_labels,
            )
        else:
            return self._generate_forward_code_single(instruction, positions)

    def _generate_forward_code_single(self, instruction: str, positions: Dict) -> str:
        """Single-turn 코드 생성 (기존 방식)"""
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen

        not_found = [name for name, info in positions.items() if info is None]
        if not_found:
            raise ValueError(f"Required objects not detected: {not_found}")

        code, _ = lerobot_code_gen(
            instruction=instruction,
            object_positions=positions,
            use_detection=False,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
        )

        return code

    def _generate_forward_code_multi_turn(
        self,
        instruction: str,
        positions: Dict,
        image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
        canonical_point_labels: Dict[str, List[str]] = None,
    ) -> str:
        """Multi-turn 코드 생성 (4-turn LLM 대화)

        LLM이 직접 이미지를 보고 장면 이해 → bbox 검출 → grasp point → 코드 생성.
        positions는 fallback으로 사용 (LLM pixel→world 변환 실패 시).

        Args:
            skip_codegen: True면 T0~T2(검출)만 수행하고 코드 생성(T3) 스킵
            canonical_labels: 코드 재사용 시 T1에서 강제할 라벨 목록
            canonical_point_labels: 코드 재사용 시 T2에서 강제할 point 라벨 {obj: [labels]}
        """
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen_multi_turn

        if image_path is None:
            print("[MultiTurn] WARNING: No image_path provided, falling back to single-turn")
            return self._generate_forward_code_single(instruction, positions)

        # depth 기반 3D 좌표 변환을 위해 카메라 전달
        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("realsense")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        code, mt_positions, mt_info = lerobot_code_gen_multi_turn(
            instruction=instruction,
            image_path=image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            fallback_positions=positions,
            camera=active_camera,
            cad_image_dirs=self.cad_image_dirs,
            side_view_image=self.side_view_image,
            codegen_model=self.codegen_model,
            skip_codegen=skip_codegen,
            canonical_labels=canonical_labels,
            canonical_point_labels=canonical_point_labels,
        )

        # multi-turn 정보 저장
        self.multi_turn_info = mt_info

        # multi-turn으로 생성된 positions로 업데이트
        # (world 좌표 변환 성공한 것만)
        for name, info in mt_positions.items():
            if not info.get("_needs_world_coords", False):
                self.detected_positions[name] = info

        return code

    def _extract_skill_sequence(self, exec_globals: Dict = None):
        """LeRobotSkills._last_instance에서 스킬 시퀀스를 추출"""
        try:
            from skills.skills_lerobot import LeRobotSkills
            if LeRobotSkills._last_instance and hasattr(LeRobotSkills._last_instance, 'skill_sequence'):
                self._last_skill_sequence = LeRobotSkills._last_instance.skill_sequence
        except Exception:
            pass

    @staticmethod
    def _patch_reset_code_targets(code: str, new_targets: Dict) -> str:
        """Reset 코드 내 target_positions 블록의 좌표만 새 값으로 치환.

        target_positions = { ... } 블록을 찾아 그 안의 좌표만 교체.
        current_positions 등 다른 블록은 건드리지 않음.
        """
        import re

        block_match = re.search(r'(target_positions\s*=\s*\{)(.*?)(\})', code, re.DOTALL)
        if not block_match:
            return code

        block_body = block_match.group(2)
        for name, info in new_targets.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos is None or len(pos) < 3:
                continue
            pattern = rf'("{name}":\s*\[)[^\]]+(\])'
            replacement = rf'\g<1>{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}\2'
            block_body = re.sub(pattern, replacement, block_body)

        return code[:block_match.start()] + block_match.group(1) + block_body + block_match.group(3) + code[block_match.end():]

    def dry_run_code(self, code: str, positions: Dict) -> bool:
        """생성된 코드를 IK 검증만으로 가상 실행 (로봇 연결 없음).

        LeRobotSkills를 DryRunSkills로 교체하여 실행.
        모든 move/pick/place가 IK 체크로 대체됨.

        Returns:
            True if all IK checks passed
        """
        from skills.dry_run_skills import DryRunSkills

        # 코드 내 LeRobotSkills import를 DryRunSkills로 교체
        patched_code = code.replace(
            "from skills.skills_lerobot import LeRobotSkills",
            "from skills.dry_run_skills import DryRunSkills as LeRobotSkills",
        )

        try:
            exec_globals = {
                "__name__": "__generated__",
                "positions": positions,
            }
            exec(patched_code, exec_globals)
            if "execute_task" in exec_globals:
                exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                exec_globals["execute_reset_task"]()

            # DryRunSkills 인스턴스에서 결과 확인
            from skills.dry_run_skills import DryRunSkills as _DRS
            if _DRS._last_instance is not None:
                dry = _DRS._last_instance
                if not dry.success:
                    print(f"  [DryRun] IK failures: {dry.ik_failures}")
                return dry.success
            return True

        except Exception as e:
            print(f"  [DryRun] Code execution error: {e}")
            return False

    def execute_code(self, code: str, positions: Dict) -> bool:
        """생성된 코드 실행 (레코딩 모드 지원)

        NOTE: __name__을 "__generated__"로 설정하여 LLM 생성 코드의
        if __name__ == "__main__": 블록이 실행되지 않도록 합니다.
        이 블록에는 LLM이 하드코딩한 가짜 positions가 포함되어 있어,
        실행 시 올바른 pix2world 좌표를 덮어쓰는 버그가 발생합니다.
        대신 exec 후 execute_task()를 수동으로 호출합니다.
        """
        self._last_skill_sequence = []  # 초기화
        try:
            exec_globals = {
                "__name__": "__generated__",
                "positions": positions,
            }

            # 레코딩 모드: RecordingContext 설정
            # LeRobotSkills가 생성될 때 자동으로 콜백을 획득
            if self.record_dataset and self.dataset_recorder:
                success = self._execute_code_with_recording(code, exec_globals)
            else:
                exec(code, exec_globals)
                if "execute_task" in exec_globals:
                    exec_globals["execute_task"]()
                elif "execute_reset_task" in exec_globals:
                    exec_globals["execute_reset_task"]()
                success = True

            # 스킬 시퀀스 추출 (LeRobotSkills 인스턴스에서)
            self._extract_skill_sequence(exec_globals)
            return success

        except AssertionError as e:
            error_msg = str(e)
            # 모터 연결 실패 - 파이프라인 완전 종료
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                print(f"\n[Pipeline] FATAL: Robot connection failed - {e}")
                print(f"[Pipeline] Terminating pipeline...")
                raise  # 상위로 전파하여 파이프라인 종료
            # 그 외 AssertionError는 일반 에러로 처리
            print(f"[Pipeline] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"[Pipeline] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_code_with_recording(self, code: str, exec_globals: Dict) -> bool:
        """RecordingContext를 사용한 레코딩 포함 코드 실행 (멀티 카메라 지원)

        RecordingContext를 설정하면 LeRobotSkills가 생성될 때
        자동으로 콜백을 획득하여 제어 루프 내에서 (state, action) 쌍을 캡처합니다.

        멀티 카메라 모드:
        - camera_manager가 있으면 async_read_all()로 모든 카메라에서 캡처
        - record_frame_multi()로 멀티 카메라 프레임 저장
        """
        try:
            from record_dataset.context import RecordingContext

            # 레코딩용 카메라 결정 (멀티 카메라 우선)
            recording_camera = self.camera_manager if self.camera_manager else self.camera

            # RecordingContext 설정 (비동기 캡처 활성화!)
            # LeRobotSkills.__init__()에서 이 컨텍스트를 확인하고 콜백을 가져감
            RecordingContext.setup(
                recorder=self.dataset_recorder,
                camera_manager=recording_camera,  # MultiCameraManager
                target_fps=self.recording_fps,
                control_hz=50,  # 제어 루프 주파수 (skills_lerobot.py의 time.sleep(0.02))
                use_async_capture=True,  # ✨ 비동기 캡처로 50Hz 달성!
            )
            RecordingContext.reset_episode()
            print(f"[Recording] Context setup complete (callback will be injected to LeRobotSkills)")

            # 코드 실행 - LeRobotSkills가 자동으로 콜백 획득
            exec(code, exec_globals)
            if "execute_task" in exec_globals:
                exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                exec_globals["execute_reset_task"]()

            # 레코딩 통계 출력
            stats = RecordingContext.get_stats()
            print(f"[Recording] Recorded {stats['recorded_frames']} frames, effective FPS: {stats['effective_fps']:.1f}")
            if stats.get('cameras'):
                print(f"[Recording] Cameras: {stats['cameras']}")

            return True

        except ImportError as e:
            print(f"[Recording] Warning: RecordingContext not available: {e}")
            print(f"[Recording] Falling back to non-recording execution")
            exec(code, exec_globals)
            if "execute_task" in exec_globals:
                exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                exec_globals["execute_reset_task"]()
            return True

        except AssertionError as e:
            error_msg = str(e)
            # 모터 연결 실패 - 파이프라인 완전 종료
            if "Failed to connect to robot hardware" in error_msg or "No motors found" in error_msg:
                print(f"\n[Pipeline] FATAL: Robot connection failed - {e}")
                print(f"[Pipeline] Terminating pipeline...")
                raise  # 상위로 전파하여 파이프라인 종료
            # 그 외 AssertionError는 일반 에러로 처리
            print(f"[Pipeline] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        except Exception as e:
            print(f"[Pipeline] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # RecordingContext 정리
            try:
                from record_dataset.context import RecordingContext
                RecordingContext.clear()
            except:
                pass

    def capture_final_image(self) -> Optional[np.ndarray]:
        """최종 이미지 캡처 (해상도도 함께 저장)"""
        # camera_manager가 사용 가능하면 self.camera 초기화 불필요
        if not (self.camera_manager and self.camera_manager.is_connected):
            # camera_manager가 없으면 self.camera 필요
            if self.camera is None:
                if not self.initialize_camera():
                    return None
                time.sleep(0.5)

        self.final_image = self.capture_frame()
        if self.final_image is not None:
            # 해상도 저장 (width, height)
            self.final_image_resolution = (self.final_image.shape[1], self.final_image.shape[0])
        return self.final_image

    def run_judge(
        self,
        instruction: str,
        positions: Dict,
        executed_code: str,
    ) -> Dict:
        """Judge 실행"""
        if self.initial_image is None or self.final_image is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'Initial or final image not captured',
                'success': False,
            }

        from judge import TaskJudge

        # 이미지 해상도 결정 (final_image 우선, 없으면 initial_image에서)
        image_resolution = self.final_image_resolution or self.initial_image_resolution

        # 환경변수에서 서버 모드 확인
        use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
        judge = TaskJudge(model=self.judge_model, verbose=self.verbose, use_server=use_server)
        return judge.judge(
            instruction=instruction,
            initial_image=self.initial_image,
            final_image=self.final_image,
            object_positions=positions,
            executed_code=executed_code,
            image_resolution=image_resolution,
        )

    def run_reset_judge(
        self,
        reset_mode: str,
        current_positions: Dict,
        target_positions: Dict,
        executed_code: str,
        original_instruction: str = None,
    ) -> Dict:
        """Reset Judge 실행"""
        if self.reset_initial_image is None or self.reset_final_image is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'Reset initial or final image not captured',
                'success': False,
                'reset_mode': reset_mode,
            }

        from judge import ResetJudge

        # 이미지 해상도 결정 (reset_final 우선, 없으면 reset_initial에서)
        image_resolution = self.reset_final_resolution or self.reset_initial_resolution

        # 환경변수에서 서버 모드 확인
        use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
        judge = ResetJudge(model=self.judge_model, verbose=self.verbose, use_server=use_server)
        return judge.judge(
            reset_mode=reset_mode,
            current_positions=current_positions,
            target_positions=target_positions,
            initial_image=self.reset_initial_image,
            final_image=self.reset_final_image,
            executed_code=executed_code,
            original_instruction=original_instruction,
            image_resolution=image_resolution,
        )

    def show_judge_ui(
        self,
        instruction: str,
        prediction: str,
        reasoning: str,
        positions: Dict,
    ) -> Optional[np.ndarray]:
        """Judge 결과 UI 표시 (타임아웃 적용)"""
        if self.initial_image is None or self.final_image is None:
            return None

        from judge import show_judge_result

        result_image = show_judge_result(
            initial_image=self.initial_image,
            final_image=self.final_image,
            instruction=instruction,
            prediction=prediction,
            reasoning=reasoning,
            object_positions=positions,
            wait_key=True,
            timeout_ms=self.judge_timeout_ms,
        )

        return result_image

    def show_reset_judge_ui(
        self,
        reset_mode: str,
        prediction: str,
        reasoning: str,
        current_positions: Dict,
        target_positions: Dict,
    ) -> Optional[np.ndarray]:
        """Reset Judge 결과 UI 표시 (타임아웃 적용)"""
        if self.reset_initial_image is None or self.reset_final_image is None:
            return None

        from judge import show_reset_judge_result

        result_image = show_reset_judge_result(
            initial_image=self.reset_initial_image,
            final_image=self.reset_final_image,
            reset_mode=reset_mode,
            prediction=prediction,
            reasoning=reasoning,
            current_positions=current_positions,
            target_positions=target_positions,
            wait_key=True,
            timeout_ms=self.judge_timeout_ms,
        )

        return result_image

    def save_execution_context(
        self,
        instruction: str,
        positions: Dict,
        spec: Dict,
        code: str,
        success: bool,
        save_dir: str,
    ) -> str:
        """Execution Context 저장"""
        from code_gen_lerobot.execution_context import save_forward_context

        context = save_forward_context(
            instruction=instruction,
            object_positions=positions,
            generated_spec=spec,
            generated_code=code,
            execution_success=success,
            robot_id=self.robot_id,
            output_dir=save_dir,
        )

        self.execution_context = {
            'instruction': instruction,
            'object_positions': positions,
            'generated_spec': spec,
            'generated_code': code,
            'execution_success': success,
        }

        return save_dir

    def generate_reset_code(
        self,
        original_instruction: str,
        original_positions: Dict,
        forward_spec: Dict = None,
        forward_code: str = None,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        current_positions: Dict = None,
        current_state_image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """Reset 코드 생성

        multi_turn=True일 때 VLM multi-turn 파이프라인 사용,
        False일 때 기존 Grounding DINO + 단일 LLM 방식 사용.

        Args:
            current_positions: 미리 감지된 현재 위치 (multi-robot 공유 감지용)
                              제공되면 내부 detection을 건너뜀
            current_state_image_path: 현재 상태 이미지 경로 (multi-turn 모드용)
            skip_codegen: True면 검출만 수행, 코드 생성 스킵 (multi-turn 전용)
            canonical_labels: 코드 재사용 시 강제할 라벨 목록

        Returns:
            Tuple[str, Dict, Dict, Dict]: (코드, 원래 위치, 현재 위치, 타겟 위치)
        """
        if self.multi_turn:
            return self._generate_reset_code_multi_turn(
                original_instruction=original_instruction,
                original_positions=original_positions,
                current_state_image_path=current_state_image_path,
                skip_codegen=skip_codegen,
                canonical_labels=canonical_labels,
            )
        else:
            return self._generate_reset_code_single(
                original_instruction=original_instruction,
                original_positions=original_positions,
                forward_spec=forward_spec,
                forward_code=forward_code,
                detection_timeout=detection_timeout,
                visualize_detection=visualize_detection,
                current_positions=current_positions,
            )

    def _generate_reset_code_single(
        self,
        original_instruction: str,
        original_positions: Dict,
        forward_spec: Dict = None,
        forward_code: str = None,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        current_positions: Dict = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """기존 single-turn 방식 reset 코드 생성"""
        from code_gen_lerobot.reset_execution import lerobot_reset_code_gen

        # Recording용 카메라가 있으면 공유 (current_positions가 없을 때만)
        external_camera = None
        if current_positions is None and self.camera_manager and self.camera_manager.is_connected:
            try:
                external_camera = self.camera_manager.get_camera("realsense")
                print("  [Reset Detection] Using shared camera from recording")
            except KeyError:
                pass

        reset_code, orig_pos, current_pos, target_pos = lerobot_reset_code_gen(
            original_instruction=original_instruction,
            original_positions=original_positions,
            forward_spec=forward_spec,
            forward_code=forward_code,
            external_camera=external_camera,
            detection_timeout=detection_timeout,
            visualize_detection=visualize_detection,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            reset_mode="original",
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            current_positions=current_positions,
        )

        return reset_code, orig_pos, current_pos, target_pos

    def _generate_reset_code_multi_turn(
        self,
        original_instruction: str,
        original_positions: Dict,
        current_state_image_path: str = None,
        skip_codegen: bool = False,
        canonical_labels: List[str] = None,
    ) -> Tuple[str, Dict, Dict, Dict]:
        """VLM multi-turn 방식 reset 코드 생성

        Forward와 동일한 crop-then-point 파이프라인으로
        현재 물체 위치를 VLM이 직접 검출하고 reset 코드 생성.
        """
        from code_gen_lerobot.reset_execution import lerobot_reset_code_gen_multi_turn

        if current_state_image_path is None or self.forward_initial_image_path is None:
            print("  [Reset MultiTurn] WARNING: Missing images, falling back to single-turn")
            return self._generate_reset_code_single(
                original_instruction=original_instruction,
                original_positions=original_positions,
            )

        # depth 기반 3D 좌표 변환용 카메라
        active_camera = None
        if self.camera_manager and self.camera_manager.is_connected:
            try:
                active_camera = self.camera_manager.get_camera("realsense")
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        reset_code, current_pos, target_pos, grippable, obstacles = lerobot_reset_code_gen_multi_turn(
            original_instruction=original_instruction,
            original_positions=original_positions,
            current_state_image_path=current_state_image_path,
            initial_state_image_path=self.forward_initial_image_path,
            llm_model=self.llm_model,
            robot_id=self.robot_id,
            reset_mode="original",
            camera=active_camera,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            codegen_model=self.codegen_model,
            skip_codegen=skip_codegen,
            canonical_labels=canonical_labels,
        )

        return reset_code, original_positions, current_pos, target_pos

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
    ) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            instruction: 자연어 명령어
            objects: 검출할 객체 리스트
            detection_timeout: 검출 타임아웃
            visualize_detection: 검출 시각화 여부
            save_dir: 결과 저장 디렉토리
            skip_reset: Reset 단계 건너뛰기

        Returns:
            파이프라인 결과 딕셔너리
        """
        self.instruction = instruction

        # 결과 저장 디렉토리 설정
        if save_dir is None:
            save_dir = "results"

        if use_timestamp_subdir:
            # 단일 실행: timestamp 서브디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = str(Path(save_dir) / timestamp)
        else:
            # multi-episode 모드: save_dir을 직접 사용
            result_dir = str(Path(save_dir))

        Path(result_dir).mkdir(parents=True, exist_ok=True)

        # Forward/Reset 별도 디렉토리
        forward_dir = str(Path(result_dir) / "forward")
        reset_dir = str(Path(result_dir) / "reset")
        Path(forward_dir).mkdir(parents=True, exist_ok=True)
        Path(reset_dir).mkdir(parents=True, exist_ok=True)

        # 로거 초기화
        forward_logger = TeeLogger(str(Path(forward_dir) / "forward_log.txt"))
        reset_logger = TeeLogger(str(Path(reset_dir) / "reset_log.txt"))

        result = {
            'forward': {
                'positions': {},
                'code': '',
                'execution_success': False,
            },
            'judge': {
                'prediction': 'UNCERTAIN',
                'reasoning': '',
            },
            'reset': {
                'mode': 'original',
                'current_positions': {},
                'target_positions': {},
                'code': '',
                'execution_success': False,
            },
            'reset_judge': {
                'prediction': 'UNCERTAIN',
                'reasoning': '',
                'reset_mode': "original",
            },
            'saved_files': {},
        }

        # 색상 코드
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # Set phase for logging
        self.current_phase = "Forward"
        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"

        print("\n" + CYAN + "=" * 70 + RESET)
        print(CYAN + BOLD + f"[{ep_str}] Forward + Reset Pipeline".center(70) + RESET)
        print(CYAN + "=" * 70 + RESET)

        try:
            # Forward 로깅 시작
            forward_logger.start()

            # ================================================================
            # PHASE 1: FORWARD EXECUTION
            # ================================================================
            print(f"\n{GREEN}{BOLD}" + self._log("FORWARD EXECUTION") + f"{RESET}")
            print(GREEN + "-" * 70 + RESET)

            if self.multi_turn:
                # ============================================================
                # Multi-turn: Detection 스킵, 이미지만 캡처하여 VLM에 전달
                # VLM이 Turn 1에서 직접 물체를 식별함
                # ============================================================

                # Step 1: 이미지 캡처 (Detection 없이)
                print(f"\n{YELLOW}" + self._log("Capturing image for VLM (no detection)...", step="Step 1/5") + f"{RESET}")

                # 카메라 초기화
                if not self.camera and not (self.camera_manager and self.camera_manager.is_connected):
                    if not self.initialize_camera():
                        print(f"{RED}[Error] Camera initialization failed{RESET}")
                        return result

                self.initial_image = self.capture_frame()

                # 캡처 실패 시 카메라 재초기화 후 재시도
                if self.initial_image is None:
                    print(f"  {YELLOW}[Warning] Capture failed, reinitializing camera...{RESET}")
                    self.shutdown_camera()
                    time.sleep(1.0)
                    if self.initialize_camera():
                        time.sleep(0.5)
                        self.initial_image = self.capture_frame()

                if self.initial_image is None:
                    print(f"{RED}[Error] Failed to capture image{RESET}")
                    return result

                self.initial_image_resolution = (self.initial_image.shape[1], self.initial_image.shape[0])
                print(f"  Image captured ({self.initial_image_resolution[0]}x{self.initial_image_resolution[1]})")

                # [즉시 저장] Initial 이미지
                initial_path = Path(forward_dir) / "initial_state.jpg"
                cv2.imwrite(str(initial_path), self.initial_image)
                self.forward_initial_image_path = str(initial_path)
                print(f"  Image saved: {initial_path}")

                # Detection 없이 빈 positions
                self.detected_positions = {}
                result['forward']['positions'] = self.detected_positions

                # Step 2: (스킵 — Step 1에서 이미 캡처됨)

                # Step 3: Forward 코드 생성 (multi-turn)
                # 코드 재사용: 캐싱된 코드가 있으면 T0~T2(검출)만 수행, T3(코드생성) 스킵
                use_cached = self.cached_forward_code is not None
                if use_cached:
                    print(f"\n{YELLOW}" + self._log(f"Detection only (reusing cached code, T3 skipped)...", step="Step 3/5") + f"{RESET}")
                    # T0~T2만 수행 (positions 갱신) — point 라벨도 강제
                    self.generate_forward_code(
                        instruction, self.detected_positions,
                        image_path=str(initial_path),
                        skip_codegen=True,
                        canonical_labels=self.cached_forward_keys,
                        canonical_point_labels=getattr(self, '_cached_point_labels', None),
                    )
                    # key 일치 확인
                    if self._can_reuse_code(self.cached_forward_code, self.cached_forward_keys, self.detected_positions):
                        self.generated_code = self.cached_forward_code
                        print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET}")
                    else:
                        missing = set(self.cached_forward_keys) - set(self.detected_positions.keys())
                        print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET}")
                        self.cached_forward_code = None
                        self.cached_forward_keys = []
                        self.generated_code = self.generate_forward_code(
                            instruction, self.detected_positions,
                            image_path=str(initial_path),
                        )
                else:
                    print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, multi-turn)...", step="Step 3/5") + f"{RESET}")
                    self.generated_code = self.generate_forward_code(
                        instruction, self.detected_positions,
                        image_path=str(initial_path),
                    )

            else:
                # ============================================================
                # Single-turn: 기존 Grounding DINO Detection → 코드 생성
                # ============================================================

                # Step 1: 객체 검출
                # Note: Recording 카메라가 있으면 run_detection에서 자동 공유
                print(f"\n{YELLOW}" + self._log(f"Detecting objects: {objects}", step="Step 1/5") + f"{RESET}")
                if visualize_detection:
                    print("  (Visualization mode)")
                else:
                    if not self.initialize_camera():
                        print(f"{RED}[Error] Camera initialization failed{RESET}")
                        return result

                self.detected_positions = self.run_detection(
                    queries=objects,
                    timeout=detection_timeout,
                    visualize=visualize_detection,
                )
                result['forward']['positions'] = self.detected_positions

                # [즉시 저장] Detection 이미지
                if self.detection_image is not None:
                    detection_path = Path(forward_dir) / "detection_result.jpg"
                    cv2.imwrite(str(detection_path), self.detection_image)  # Already BGR
                    print(f"  Detection image saved: {detection_path}")

                # 검출 결과 검증 1: 객체 미발견 체크
                not_found = [k for k, v in self.detected_positions.items() if v is None]
                if not_found:
                    print(f"{RED}[Error] Objects not detected: {not_found}{RESET}")
                    return result

                # 첫 에피소드의 검출 위치 저장 (original reset mode용)
                if self.first_episode_positions is None:
                    import copy
                    self.first_episode_positions = copy.deepcopy(self.detected_positions)
                    print(f"  {GREEN}[First Episode] Initial positions saved for 'original' reset mode{RESET}")

                # 검출 결과 검증 2: Workspace 범위 체크
                print(f"\n{YELLOW}" + self._log("Checking workspace bounds...", tag="Validation") + f"{RESET}")
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from lerobot_cap.workspace import BaseWorkspace
                from lerobot_cap.transforms import FrameTransformer

                # FrameTransformer 생성 (robot_id에 맞는 config 로드)
                frame_config_path = PROJECT_ROOT / f"robot_configs/world2robot_matrices/robot{self.robot_id}_matrix.json"
                if frame_config_path.exists():
                    frame_transformer = FrameTransformer(str(frame_config_path))
                else:
                    frame_transformer = None
                    print(f"  {YELLOW}[Warning] Frame config not found: {frame_config_path}{RESET}")

                workspace = BaseWorkspace(frame_transformer=frame_transformer)
                print(f"  Workspace: reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

                critical_error = False
                for obj_name, obj_info in self.detected_positions.items():
                    if obj_info is None:
                        continue
                    pos = obj_info.get("position") if isinstance(obj_info, dict) else obj_info
                    if pos is None:
                        continue
                    position_m = np.array([pos[0], pos[1], pos[2]])

                    if not workspace.is_reachable(position_m):
                        print(f"{RED}[CRITICAL] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m{RESET}")
                        print(f"{RED}  Outside reach limits: [{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m{RESET}")
                        critical_error = True
                    else:
                        print(f"  {GREEN}✓ '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m - OK{RESET}")

                if critical_error:
                    print(f"\n{RED}[Error] Critical workspace violation detected. Terminating pipeline.{RESET}")
                    return result

                # Step 2: Initial 이미지 캡처
                print(f"\n{YELLOW}" + self._log("Capturing initial state...", step="Step 2/5") + f"{RESET}")
                if self.initial_image is None:
                    self.initial_image = self.capture_frame()
                if self.initial_image is not None:
                    # 해상도 저장 (Judge용)
                    self.initial_image_resolution = (self.initial_image.shape[1], self.initial_image.shape[0])
                    print(f"  Initial image captured ({self.initial_image_resolution[0]}x{self.initial_image_resolution[1]})")
                    # [즉시 저장] Initial 이미지
                    initial_path = Path(forward_dir) / "initial_state.jpg"
                    cv2.imwrite(str(initial_path), self.initial_image)  # Already BGR
                    self.forward_initial_image_path = str(initial_path)
                    print(f"  Initial image saved: {initial_path}")

                # Step 3: Forward 코드 생성 (single-turn)
                # 코드 재사용: 캐싱된 코드가 있고 key 일치하면 스킵
                if self._can_reuse_code(self.cached_forward_code, self.cached_forward_keys, self.detected_positions):
                    self.generated_code = self.cached_forward_code
                    print(f"\n{YELLOW}" + self._log(f"Reusing cached code (single-turn, T3 skipped)...", step="Step 3/5") + f"{RESET}")
                    print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET}")
                else:
                    if self.cached_forward_code is not None:
                        missing = set(self.cached_forward_keys) - set(self.detected_positions.keys())
                        print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET}")
                        self.cached_forward_code = None
                        self.cached_forward_keys = []
                    print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, single-turn)...", step="Step 3/5") + f"{RESET}")
                    self.generated_code = self.generate_forward_code(
                        instruction,
                        self.detected_positions,
                    )
            result['forward']['code'] = self.generated_code
            result['forward']['positions'] = self.detected_positions

            # 첫 에피소드의 검출 위치 저장 (original reset mode용)
            # multi-turn에서도 detected_positions가 갱신된 후 저장
            if self.first_episode_positions is None and self.detected_positions:
                import copy
                self.first_episode_positions = copy.deepcopy(self.detected_positions)
                GREEN_TMP = "\033[92m"
                RESET_TMP = "\033[0m"
                print(f"  {GREEN_TMP}[First Episode] Initial positions saved for 'original' reset mode{RESET_TMP}")
                for name, info in self.detected_positions.items():
                    if isinstance(info, dict) and "position" in info:
                        pos = info["position"]
                        print(f"    + {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

            # [즉시 저장] Generated code
            code_path = Path(forward_dir) / "generated_code.py"
            code_path.write_text(self.generated_code)
            print(f"  Generated code saved: {code_path}")

            # [즉시 저장] Multi-turn info (if available)
            if self.multi_turn and self.multi_turn_info:
                mt_info_path = Path(forward_dir) / "multi_turn_info.json"
                mt_save = {
                    "turn0_response": self.multi_turn_info.get("turn0_response", ""),
                    "turn1_response": self.multi_turn_info.get("turn1_response", ""),
                    "turn2_response": self.multi_turn_info.get("turn2_response", ""),
                    "turn3_response": self.multi_turn_info.get("turn3_response", ""),
                    "turn1_parsed": self.multi_turn_info.get("turn1_parsed"),
                    "turn1_sideview_parsed": self.multi_turn_info.get("turn1_sideview_parsed"),
                    "turn2_parsed": self.multi_turn_info.get("turn2_parsed"),
                    "detected_objects": self.multi_turn_info.get("detected_objects"),
                    "all_points": self.multi_turn_info.get("all_points"),
                    "crop_responses": self.multi_turn_info.get("crop_responses"),
                    "turn_test_response": self.multi_turn_info.get("turn_test_response", ""),
                    "turn_test_overhead_waypoints": self.multi_turn_info.get("turn_test_overhead_waypoints"),
                    "turn_test_sideview_waypoints": self.multi_turn_info.get("turn_test_sideview_waypoints"),
                }
                import json as _json
                with open(mt_info_path, 'w', encoding='utf-8') as f:
                    _json.dump(mt_save, f, indent=2, ensure_ascii=False, default=str)
                print(f"  Multi-turn info saved: {mt_info_path}")

                # [신규] Turn 시각화 이미지 저장
                if self.initial_image is not None:
                    t1_parsed = self.multi_turn_info.get("turn1_parsed")
                    t2_parsed = self.multi_turn_info.get("turn2_parsed")

                    if t1_parsed:
                        self._visualize_turn1(
                            self.initial_image.copy(), t1_parsed,
                            str(Path(forward_dir) / "turn1_detection.jpg"),
                            turn1_raw=self.multi_turn_info.get("turn0_response", ""),
                        )
                    if t2_parsed:
                        self._visualize_turn2(
                            self.initial_image.copy(), t1_parsed, t2_parsed,
                            str(Path(forward_dir) / "turn2_grasp_points.jpg")
                        )

                    # Side-view Turn 1 + Turn 2 시각화
                    sv_image_path = self.multi_turn_info.get("side_view_image")
                    if sv_image_path and os.path.isfile(sv_image_path):
                        sv_img_base = cv2.imread(sv_image_path)
                        if sv_img_base is not None:
                            sv_img_base = cv2.resize(sv_img_base, (640, 480))

                            # Turn 1 side-view bbox 시각화
                            t1_sv_parsed = self.multi_turn_info.get("turn1_sideview_parsed")
                            if t1_sv_parsed:
                                self._visualize_turn1(
                                    sv_img_base.copy(), t1_sv_parsed,
                                    str(Path(forward_dir) / "turn1_sideview_detection.jpg"),
                                )

                            # Turn 2 side-view grasp points 시각화
                            sv_grasp = t2_parsed.get("sv_grasp_points") if isinstance(t2_parsed, dict) else None
                            if sv_grasp:
                                sv_t2_compat = {"grasp_points": sv_grasp}
                                self._visualize_turn2(
                                    sv_img_base.copy(), t1_sv_parsed, sv_t2_compat,
                                    str(Path(forward_dir) / "turn2_sideview_grasp_points.jpg")
                                )

                    oh_waypoints = self.multi_turn_info.get("turn_test_overhead_waypoints")
                    if oh_waypoints:
                        self._visualize_turn_test(
                            self.initial_image.copy(), t2_parsed, oh_waypoints,
                            str(Path(forward_dir) / "turn_test_overhead_waypoints.jpg")
                        )

                    sv_waypoints = self.multi_turn_info.get("turn_test_sideview_waypoints")
                    if sv_waypoints and sv_image_path and os.path.isfile(sv_image_path):
                        sv_img = cv2.imread(sv_image_path)
                        if sv_img is not None:
                            sv_img = cv2.resize(sv_img, (640, 480))
                            self._visualize_turn_test(
                                sv_img, None, sv_waypoints,
                                str(Path(forward_dir) / "turn_test_sideview_waypoints.jpg")
                            )

                # [신규] Turn 2 crop 이미지 저장
                crop_dir = self.multi_turn_info.get("crop_dir")
                if crop_dir and os.path.isdir(crop_dir):
                    import shutil
                    for fname in sorted(os.listdir(crop_dir)):
                        if fname.endswith(('.jpg', '.png')):
                            src = os.path.join(crop_dir, fname)
                            dst = os.path.join(forward_dir, fname)
                            shutil.copy2(src, dst)
                    print(f"  Crop images saved to: {forward_dir}")

                # [신규] Turn description 로그 저장
                self._save_turn_logs(forward_dir, self.multi_turn_info)

            print("\n" + "-" * 40)
            print("Generated Forward Code (preview):")
            print("-" * 40)
            code_preview = self.generated_code[:600]
            if len(self.generated_code) > 600:
                code_preview += "\n... (truncated)"
            print(code_preview)
            print("-" * 40)

            # Step 4: Forward 코드 실행
            print(f"\n{YELLOW}" + self._log(f"Executing forward code on Robot {self.robot_id}...", step="Step 4/5") + f"{RESET}")

            # 레코딩 모드: 에피소드 시작
            if self.record_dataset:
                self._start_episode_recording(task=instruction)

            forward_success = self.execute_code(self.generated_code, self.detected_positions)
            result['forward']['execution_success'] = forward_success

            # 후처리: 스킬 라벨 생성
            if hasattr(self, '_last_skill_sequence') and self._last_skill_sequence:
                try:
                    from record_dataset.postprocess import generate_skill_labels
                    skill_labels_path = str(Path(forward_dir) / "skill_labels.json")
                    generate_skill_labels(
                        instruction=instruction,
                        skill_sequence=self._last_skill_sequence,
                        llm_model=self.judge_model,  # 빠른 모델 사용
                        save_path=skill_labels_path,
                    )
                except Exception as e:
                    print(f"  [SkillLabeler] Warning: {e}")

            if forward_success:
                print(f"  {GREEN}Forward execution SUCCESS{RESET}")
            else:
                print(f"  {RED}Forward execution FAILED{RESET}")

            # Step 5: Context 저장
            print(f"\n{YELLOW}" + self._log("Saving execution context...", step="Step 5/5") + f"{RESET}")
            self.generated_spec = {}
            self.save_execution_context(
                instruction=instruction,
                positions=self.detected_positions,
                spec=self.generated_spec,
                code=self.generated_code,
                success=forward_success,
                save_dir=forward_dir,
            )
            print(f"  Context saved to: {forward_dir}")

            # ================================================================
            # PHASE 2: JUDGE (EVALUATION)
            # ================================================================
            print(f"\n{MAGENTA}{BOLD}" + self._log("JUDGE (Forward Evaluation)") + f"{RESET}")
            print(MAGENTA + "-" * 70 + RESET)

            # Step 1: Final 이미지 캡처
            print(f"\n{YELLOW}" + self._log("Capturing final state...", step="Step 1/3", tag="Judge") + f"{RESET}")
            time.sleep(1.0)
            self.capture_final_image()
            if self.final_image is not None:
                print("  Final image captured")
                final_path = Path(forward_dir) / "final_state.jpg"
                cv2.imwrite(str(final_path), self.final_image)
                print(f"  Final image saved: {final_path}")

            # Step 2: Judge 실행
            judge_prediction = "UNCERTAIN"
            print(f"\n{YELLOW}" + self._log(f"Running VLM Judge ({self.judge_model})...", step="Step 2/3", tag="Judge") + f"{RESET}")
            if self.initial_image is not None and self.final_image is not None:
                judge_result = self.run_judge(
                    instruction=instruction,
                    positions=self.detected_positions,
                    executed_code=self.generated_code,
                )
                result['judge'] = judge_result

                judge_prediction = judge_result.get('prediction', 'UNCERTAIN')
                reasoning = judge_result.get('reasoning', '')

                pred_color = GREEN if judge_prediction == "TRUE" else RED if judge_prediction == "FALSE" else YELLOW
                print(f"  Prediction: {pred_color}{judge_prediction}{RESET}")
                print(f"  Reasoning: {reasoning[:100]}...")
            else:
                print(f"  {YELLOW}Skipped (missing images){RESET}")

            # 레코딩 모드: Judge 결과에 따라 에피소드 저장/폐기
            if self.record_dataset:
                should_discard = judge_prediction != "TRUE"
                self._end_episode_recording(discard=should_discard)

                if not should_discard:
                    # Skill recording 시각화 저장 (성공한 에피소드만)
                    try:
                        from record_dataset.visualize_skills import generate_skill_visualizations
                        dataset = self.dataset_recorder._dataset
                        dataset._ensure_hf_dataset_loaded()
                        episode_df = dataset.hf_dataset.to_pandas()
                        saved_viz = generate_skill_visualizations(
                            dataframe=episode_df,
                            save_dir=forward_dir,
                            episode_index=self.dataset_recorder.episode_count - 1,
                        )
                        if saved_viz:
                            print(f"  Skill visualizations saved: {len(saved_viz)} files")
                    except Exception as e:
                        import traceback
                        print(f"  Warning: Skill visualization failed: {e}")
                        traceback.print_exc()

                # Step 3: Judge UI 표시 (타임아웃 적용)
                print(f"\n{YELLOW}" + self._log(f"Displaying result ({self.judge_timeout_ms/1000:.1f}s timeout)...", step="Step 3/3", tag="Judge") + f"{RESET}")
                if self.initial_image is not None and self.final_image is not None:
                    from judge import save_judge_log

                    result_image = self.show_judge_ui(
                        instruction=instruction,
                        prediction=result['judge'].get('prediction', 'UNCERTAIN'),
                        reasoning=result['judge'].get('reasoning', ''),
                        positions=self.detected_positions,
                    )

                    # Judge 로그 저장 (forward 폴더에)
                    result['saved_files'] = save_judge_log(
                        save_dir=forward_dir,
                        initial_image=self.initial_image,
                        final_image=self.final_image,
                        instruction=instruction,
                        prediction=result['judge'].get('prediction', 'UNCERTAIN'),
                        reasoning=result['judge'].get('reasoning', ''),
                        object_positions=self.detected_positions,
                        executed_code=self.generated_code,
                        result_image=result_image,
                        detection_image=self.detection_image,
                    )
            # 코드 캐시 갱신: 실행 성공이면 캐싱 (Judge FALSE면 무효화)
            judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
            should_cache = forward_success and judge_pred != 'FALSE'
            if should_cache:
                if self.cached_forward_code is None:
                    self.cached_forward_code = self.generated_code
                    self.cached_forward_keys = self._extract_position_keys(self.generated_code)
                    # point labels 캐시: {obj: [label1, label2, ...]}
                    self._cached_point_labels = {}
                    for name, info in self.detected_positions.items():
                        if isinstance(info, dict) and "points" in info:
                            self._cached_point_labels[name] = list(info["points"].keys())
                    print(f"  {GREEN}[CodeReuse] Forward code cached (keys: {self.cached_forward_keys}){RESET}")
                    if self._cached_point_labels:
                        print(f"  {GREEN}[CodeReuse] Point labels cached: {self._cached_point_labels}{RESET}")
            elif not forward_success or judge_pred == 'FALSE':
                if self.cached_forward_code is not None:
                    reason = "execution failed" if not forward_success else "Judge=FALSE"
                    print(f"  {YELLOW}[CodeReuse] Cache invalidated ({reason}){RESET}")
                self.cached_forward_code = None
                self.cached_forward_keys = []
                self._cached_point_labels = None

            # Forward 로깅 종료
            forward_log_path = forward_logger.stop()
            print(f"\n  Forward log saved to: {forward_log_path}")

            # Forward 완료 후 콜백: seed 생성 등 (reset_target 갱신 가능)
            if pre_reset_callback is not None:
                new_target = pre_reset_callback()
                if new_target is not None:
                    reset_target_positions = new_target

            # ================================================================
            # PHASE 3: RESET EXECUTION
            # ================================================================
            if not skip_reset:
                # Switch phase for logging
                self.current_phase = "Reset"

                # Reset 로깅 시작
                reset_logger.start()

                print(f"\n{CYAN}{BOLD}" + self._log("RESET EXECUTION") + f"{RESET}")
                print(CYAN + "-" * 70 + RESET)

                # 카메라 종료 (Forward에서 사용하던 별도 카메라)
                self.shutdown_camera()

                # Reset target 결정: 외부 지정 > first_episode_positions > detected_positions
                if reset_target_positions is not None:
                    reset_original_positions = reset_target_positions
                    print(f"  Reset target: seed positions")
                elif self.first_episode_positions is not None:
                    reset_original_positions = self.first_episode_positions
                    print(f"  Reset target: first episode positions")
                else:
                    reset_original_positions = self.detected_positions
                    print(f"  Reset target: current detected positions")

                # Step 1 & 2: Reset 코드 생성
                multi_turn_str = "multi-turn VLM" if self.multi_turn else "single-turn"
                print(f"\n{YELLOW}" + self._log(f"Generating reset code ({multi_turn_str})...", step="Step 1/4") + f"{RESET}")
                try:

                    # Multi-turn 모드: 코드 생성 전에 current_state 이미지 캡처
                    reset_current_state_image_path = None
                    if self.multi_turn:
                        print(f"  [MultiTurn] Capturing current state for VLM...")
                        if not (self.camera_manager and self.camera_manager.is_connected):
                            if not self.initialize_camera():
                                print(f"  {RED}Failed to initialize camera for reset VLM{RESET}")
                        time.sleep(0.3)
                        reset_current_frame = self.capture_frame()
                        if reset_current_frame is not None:
                            reset_current_state_image_path = str(Path(reset_dir) / "current_state.jpg")
                            cv2.imwrite(reset_current_state_image_path, reset_current_frame)
                            print(f"  Current state captured: {reset_current_state_image_path}")
                            # 이 이미지를 reset_initial_image로도 사용 (Judge용)
                            self.reset_initial_image = reset_current_frame
                            self.reset_initial_resolution = (reset_current_frame.shape[1], reset_current_frame.shape[0])

                    # Reset 코드는 매번 좌표가 달라지므로(현재위치→타겟위치 하드코딩) 항상 새로 생성
                    reset_code, _, current_positions, target_positions = self.generate_reset_code(
                        original_instruction=instruction,
                        original_positions=reset_original_positions,
                        forward_spec=self.generated_spec,
                        forward_code=self.generated_code,
                        detection_timeout=detection_timeout,
                        visualize_detection=visualize_detection,
                        current_state_image_path=reset_current_state_image_path,
                    )

                    result['reset']['current_positions'] = current_positions
                    result['reset']['target_positions'] = target_positions
                    result['reset']['code'] = reset_code

                    # [즉시 저장] Reset generated code
                    reset_code_path = Path(reset_dir) / "generated_code.py"
                    reset_code_path.write_text(reset_code)
                    print(f"  Reset code saved: {reset_code_path}")

                    # [즉시 저장] Reset positions (current & target)
                    reset_positions_path = Path(reset_dir) / "positions.json"
                    reset_positions_data = {
                        "current_positions": current_positions,
                        "target_positions": target_positions,
                        "reset_mode": "original",
                    }
                    with open(reset_positions_path, 'w') as f:
                        json.dump(reset_positions_data, f, indent=2, default=str)
                    print(f"  Reset positions saved: {reset_positions_path}")

                    print("\n" + "-" * 40)
                    print("Generated Reset Code (preview):")
                    print("-" * 40)
                    reset_preview = reset_code[:600]
                    if len(reset_code) > 600:
                        reset_preview += "\n... (truncated)"
                    print(reset_preview)
                    print("-" * 40)

                    # Step 2: Reset 초기 이미지 캡처
                    # multi-turn 모드에서는 이미 current_state로 캡처됨
                    if self.reset_initial_image is not None and self.multi_turn:
                        print(f"\n{YELLOW}" + self._log("Reset initial image already captured (multi-turn)", step="Step 2/4") + f"{RESET}")
                        # [즉시 저장] Reset initial 이미지 (이미 current_state.jpg로 저장됨, initial_state.jpg로도 복사)
                        reset_initial_path = Path(reset_dir) / "initial_state.jpg"
                        cv2.imwrite(str(reset_initial_path), self.reset_initial_image)
                        print(f"  Reset initial image saved: {reset_initial_path}")
                    else:
                        print(f"\n{YELLOW}" + self._log("Capturing reset initial state...", step="Step 2/4") + f"{RESET}")
                        # camera_manager가 있으면 재사용, 없으면 새로 생성
                        if not (self.camera_manager and self.camera_manager.is_connected):
                            if not self.initialize_camera():
                                print(f"  {RED}Failed to initialize camera{RESET}")
                        time.sleep(0.3)
                        self.reset_initial_image = self.capture_frame()
                        if self.reset_initial_image is not None:
                            # 해상도 저장 (Judge용)
                            self.reset_initial_resolution = (self.reset_initial_image.shape[1], self.reset_initial_image.shape[0])
                            print(f"  {GREEN}Reset initial image captured ({self.reset_initial_resolution[0]}x{self.reset_initial_resolution[1]}){RESET}")
                            # [즉시 저장] Reset initial 이미지
                            reset_initial_path = Path(reset_dir) / "initial_state.jpg"
                            cv2.imwrite(str(reset_initial_path), self.reset_initial_image)  # Already BGR
                            print(f"  Reset initial image saved: {reset_initial_path}")
                        else:
                            print(f"  {YELLOW}Warning: Failed to capture reset initial image{RESET}")

                    # Step 3: Reset 코드 실행
                    print(f"\n{YELLOW}" + self._log("Executing reset code...", step="Step 3/4") + f"{RESET}")
                    self.reset_code = reset_code
                    reset_success = self.execute_code(reset_code, current_positions)
                    result['reset']['execution_success'] = reset_success

                    if reset_success:
                        print(f"  {GREEN}Reset execution SUCCESS{RESET}")
                    else:
                        print(f"  {RED}Reset execution FAILED{RESET}")

                    # Step 4: Reset 최종 이미지 캡처
                    print(f"\n{YELLOW}" + self._log("Capturing reset final state...", step="Step 4/4") + f"{RESET}")
                    time.sleep(0.5)  # 로봇 정지 대기
                    # camera_manager가 있으면 재사용, 없으면 새로 생성
                    if not (self.camera_manager and self.camera_manager.is_connected):
                        if self.camera is None:
                            self.initialize_camera()
                            time.sleep(0.3)
                    self.reset_final_image = self.capture_frame()
                    if self.reset_final_image is not None:
                        # 해상도 저장 (Judge용)
                        self.reset_final_resolution = (self.reset_final_image.shape[1], self.reset_final_image.shape[0])
                        print(f"  {GREEN}Reset final image captured ({self.reset_final_resolution[0]}x{self.reset_final_resolution[1]}){RESET}")
                        # [즉시 저장] Reset final 이미지
                        reset_final_path = Path(reset_dir) / "final_state.jpg"
                        cv2.imwrite(str(reset_final_path), self.reset_final_image)  # Already BGR
                        print(f"  Reset final image saved: {reset_final_path}")
                    else:
                        print(f"  {YELLOW}Warning: Failed to capture reset final image{RESET}")

                    # Step 5: Reset Judge 실행
                    print(f"\n{CYAN}" + self._log("Evaluating reset result...", tag="Judge") + f"{RESET}")
                    reset_judge_result = self.run_reset_judge(
                        reset_mode="original",
                        current_positions=current_positions,
                        target_positions=target_positions,
                        executed_code=reset_code,
                        original_instruction=instruction,
                    )
                    result['reset_judge'] = reset_judge_result

                    rj_pred = reset_judge_result.get('prediction', 'UNCERTAIN')
                    pred_color = GREEN if rj_pred == "TRUE" else RED if rj_pred == "FALSE" else YELLOW
                    print(f"  Prediction: {pred_color}{rj_pred}{RESET}")
                    rj_reasoning = reset_judge_result.get('reasoning', '')
                    if rj_reasoning:
                        reasoning_preview = rj_reasoning[:200]
                        if len(rj_reasoning) > 200:
                            reasoning_preview += "..."
                        print(f"  Reasoning: {reasoning_preview}")

                    # Reset Judge UI 표시
                    if self.reset_initial_image is not None and self.reset_final_image is not None:
                        self.show_reset_judge_ui(
                            reset_mode="original",
                            prediction=rj_pred,
                            reasoning=rj_reasoning,
                            current_positions=current_positions,
                            target_positions=target_positions,
                        )

                    # Reset judge 로그 저장
                    reset_log = {
                        'reset_mode': "original",
                        'current_positions': current_positions,
                        'target_positions': target_positions,
                        'prediction': rj_pred,
                        'reasoning': rj_reasoning,
                        'execution_success': result['reset']['execution_success'],
                    }
                    reset_log_path = Path(reset_dir) / "reset_judge_result.json"
                    with open(reset_log_path, 'w') as f:
                        json.dump(reset_log, f, indent=2, default=str)
                    print(f"  Reset judge result saved to: {reset_log_path}")

                    # Reset 코드를 dry_run 검증용으로 캐싱 (재사용은 하지 않음)
                    if result['reset']['execution_success'] and result['reset']['code']:
                        self.cached_reset_code = result['reset']['code']

                except Exception as e:
                    print(f"{RED}[Error] Reset failed: {e}{RESET}")
                    import traceback
                    traceback.print_exc()

                # Reset 로깅 종료
                reset_txt_log_path = reset_logger.stop()
                print(f"\n  Reset log saved to: {reset_txt_log_path}")
            else:
                print(f"\n{YELLOW}[PHASE 3] RESET EXECUTION - Skipped{RESET}")

            # ================================================================
            # SUMMARY
            # ================================================================
            self._print_summary(result, skip_reset)

            return result

        finally:
            self.shutdown_camera()

    def _visualize_turn1(
        self,
        image: np.ndarray,
        turn1_parsed,
        save_path: str,
        turn1_raw: str = "",
    ) -> None:
        """Turn 1 bbox 시각화 — test6 스타일 (녹색 bbox + 라벨)"""
        obj_list = None
        if isinstance(turn1_parsed, list):
            obj_list = turn1_parsed
        elif isinstance(turn1_parsed, dict) and "objects" in turn1_parsed:
            obj_list = turn1_parsed["objects"]

        if not obj_list:
            print(f"  [Visualize] Turn 1: No objects to draw")
            return

        img_h, img_w = image.shape[:2]

        for obj in obj_list:
            label = obj.get("label") or obj.get("name", "?")
            box = obj.get("box_2d") or obj.get("bbox_pixel")
            if box and len(box) == 4:
                ymin, xmin, ymax, xmax = box
                x1 = int(xmin * img_w / 1000)
                y1 = int(ymin * img_h / 1000)
                x2 = int(xmax * img_w / 1000)
                y2 = int(ymax * img_h / 1000)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1 + 2, y1 - 4), font, 0.5, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn 1 visualization saved: {save_path}")

    def _visualize_turn2(
        self,
        image: np.ndarray,
        turn1_parsed,
        turn2_parsed,
        save_path: str,
    ) -> None:
        """Turn 2 결과 시각화: bbox + grasp/interaction 포인트 (test6 스타일)

        Args:
            image: BGR 이미지 (copy)
            turn1_parsed: Turn 1 parsed 데이터 (bbox 리스트)
            turn2_parsed: Turn 2 parsed 데이터 ({"grasp_points": [...]})
            save_path: 저장 경로
        """
        img_h, img_w = image.shape[:2]

        # Turn 1 bbox 그리기
        obj_list = None
        if isinstance(turn1_parsed, list):
            obj_list = turn1_parsed
        elif isinstance(turn1_parsed, dict) and "objects" in turn1_parsed:
            obj_list = turn1_parsed["objects"]

        if obj_list:
            for obj in obj_list:
                box = obj.get("box_2d") or obj.get("bbox_pixel")
                label = obj.get("label", "")
                if box and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * img_w / 1000)
                    y1 = int(ymin * img_h / 1000)
                    x2 = int(xmax * img_w / 1000)
                    y2 = int(ymax * img_h / 1000)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Critical points 그리기 (grasp: 녹색 원, interaction: 빨간 X)
        ROLE_COLORS = {
            "grasp": (0, 255, 0),       # green
            "pick": (0, 255, 0),         # green (하위호환)
            "interaction": (0, 0, 255),  # red
            "place": (255, 0, 0),        # blue (하위호환)
        }

        grasp_points = []
        if isinstance(turn2_parsed, dict) and "grasp_points" in turn2_parsed:
            grasp_points = turn2_parsed["grasp_points"]

        if not grasp_points:
            print(f"  [Visualize] Turn 2: No points to draw")
            return

        for i, gp in enumerate(grasp_points):
            name = gp.get("object_name", "unknown")
            sub_label = gp.get("label", "")
            role = gp.get("role", "grasp")
            pixel = gp.get("point_pixel")

            if not pixel or len(pixel) != 2:
                continue

            px = int(pixel[1] * img_w / 1000)
            py = int(pixel[0] * img_h / 1000)
            color = ROLE_COLORS.get(role, (255, 255, 255))

            # 마커: grasp → green dot, interaction → red dot
            cv2.circle(image, (px, py), 3, color, -1)
            cv2.circle(image, (px, py), 3, (0, 0, 0), 1)

            # 라벨
            marker_label = f"{name}: {sub_label}" if sub_label else name
            (tw, th), _ = cv2.getTextSize(marker_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            # 짝수/홀수로 텍스트 위치 교차 (겹침 방지)
            if i % 2 == 0:
                text_x = min(px + 15, img_w - tw - 5)
                text_y = max(py - 8, th + 5)
            else:
                text_x = max(px - tw - 15, 2)
                text_y = min(py + 12, img_h - 5)

            cv2.rectangle(image, (text_x - 2, text_y - th - 4),
                          (text_x + tw + 2, text_y + 4), color, -1)
            cv2.putText(image, marker_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn 2 visualization saved: {save_path}")

    def _visualize_turn_test(
        self,
        image: np.ndarray,
        turn2_parsed,
        waypoints: list,
        save_path: str,
    ) -> None:
        """Turn Test 결과 시각화: interaction points + waypoint trajectory

        Args:
            image: BGR 이미지 (copy)
            turn2_parsed: Turn 2 parsed 데이터 (interaction point 표시용)
            waypoints: turn_test_waypoints 리스트 [{"py", "px", "label", ...}, ...]
            save_path: 저장 경로
        """
        img_h, img_w = image.shape[:2]

        WAYPOINT_COLOR = (255, 165, 0)  # orange (BGR)
        LINE_COLOR = (255, 200, 100)    # light blue-ish line
        INTERACTION_COLOR = (0, 0, 255) # red

        # Draw interaction points from Turn 2 (for reference)
        if isinstance(turn2_parsed, dict) and "grasp_points" in turn2_parsed:
            for gp in turn2_parsed["grasp_points"]:
                if gp.get("role") != "interaction":
                    continue
                pixel = gp.get("point_pixel")
                if not pixel or len(pixel) != 2:
                    continue
                ipx = int(pixel[1] * img_w / 1000)
                ipy = int(pixel[0] * img_h / 1000)
                cv2.circle(image, (ipx, ipy), 5, INTERACTION_COLOR, -1)
                cv2.circle(image, (ipx, ipy), 5, (0, 0, 0), 1)

        if not waypoints:
            print(f"  [Visualize] Turn Test: No waypoints to draw")
            return

        # Collect pixel coords for line drawing
        wp_pixels = []
        for wp in waypoints:
            wy = wp.get("py", 0)
            wx = wp.get("px", 0)
            wp_pixels.append((wx, wy))

        # Draw lines between consecutive waypoints
        for i in range(len(wp_pixels) - 1):
            cv2.line(image, wp_pixels[i], wp_pixels[i + 1], LINE_COLOR, 2)

        # Draw waypoint dots and labels
        for i, (wp, (wx, wy)) in enumerate(zip(waypoints, wp_pixels)):
            label = wp.get("label", f"wp{i}")

            cv2.circle(image, (wx, wy), 4, WAYPOINT_COLOR, -1)
            cv2.circle(image, (wx, wy), 4, (0, 0, 0), 1)

            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            text_x = min(wx + 8, img_w - tw - 5)
            text_y = max(wy - 6, th + 5)
            cv2.rectangle(image, (text_x - 2, text_y - th - 2),
                          (text_x + tw + 2, text_y + 2), WAYPOINT_COLOR, -1)
            cv2.putText(image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Turn Test visualization saved: {save_path}")

    def _save_turn_logs(self, forward_dir: str, multi_turn_info: Dict) -> None:
        """턴별 description 로그를 텍스트 파일로 저장

        Args:
            forward_dir: forward 결과 저장 디렉토리
            multi_turn_info: multi-turn 정보 dict
        """
        if not multi_turn_info:
            return

        forward_path = Path(forward_dir)

        # Turn 0 로그 (Scene Understanding)
        turn0_raw = multi_turn_info.get("turn0_response", "")
        if turn0_raw:
            lines = ["=" * 60, "Turn 0: Scene Understanding", "=" * 60, ""]
            lines.append("[Raw Response]")
            lines.append(turn0_raw)

            (forward_path / "turn0_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 0 log saved: {forward_path / 'turn0_log.txt'}")

        # Turn 1 로그 (Bounding Box Detection)
        turn1_raw = multi_turn_info.get("turn1_response", "")
        turn1_parsed = multi_turn_info.get("turn1_parsed")
        if turn1_raw:
            lines = ["=" * 60, "Turn 1: Bounding Box Detection", "=" * 60, ""]
            lines.append("[Raw Response]")
            lines.append(turn1_raw)
            lines.append("")

            if turn1_parsed:
                lines.append("[Parsed Summary]")
                obj_list = None
                if isinstance(turn1_parsed, list):
                    obj_list = turn1_parsed
                elif isinstance(turn1_parsed, dict) and "objects" in turn1_parsed:
                    obj_list = turn1_parsed["objects"]

                if obj_list:
                    for obj in obj_list:
                        name = obj.get("label") or obj.get("name", "?")
                        box = obj.get("box_2d") or obj.get("bbox_pixel", "N/A")
                        size = obj.get("estimated_size_cm", "N/A")
                        lines.append(f"  - {name}: bbox={box}, size_cm={size}")

            (forward_path / "turn1_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 1 log saved: {forward_path / 'turn1_log.txt'}")

        # Turn 2+ 로그 (Crop-then-Point)
        all_points = multi_turn_info.get("all_points", [])
        crop_responses = multi_turn_info.get("crop_responses", [])
        if all_points or crop_responses:
            lines = ["=" * 60, "Turn 2+: Crop-then-Point", "=" * 60, ""]

            # Crop별 응답
            for cr in crop_responses:
                lines.append(f"--- Crop: {cr.get('label', '?')} ---")
                lines.append(cr.get("response", ""))
                lines.append("")

            # Parsed 포인트 요약
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

            (forward_path / "turn2_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 2+ log saved: {forward_path / 'turn2_log.txt'}")

        # Turn 3 로그 (Code Generation)
        turn3_raw = multi_turn_info.get("turn3_response", "")
        if turn3_raw:
            lines = ["=" * 60, "Turn 3: Code Generation", "=" * 60, ""]
            lines.append("[Raw Response]")
            lines.append(turn3_raw)

            (forward_path / "turn3_log.txt").write_text("\n".join(lines), encoding="utf-8")
            print(f"  Turn 3 log saved: {forward_path / 'turn3_log.txt'}")

    def _update_results(self, all_results: Dict, result: Dict, episode_num: int, skip_reset: bool) -> None:
        """에피소드 결과를 all_results에 추가"""
        all_results['episodes'].append({
            'episode': episode_num, 'result': result, 'success': True, 'error': None,
        })
        judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
        if result['forward']['execution_success']:
            all_results['summary']['forward_success'] += 1
        if judge_pred == 'TRUE':
            all_results['summary']['forward_judge_true'] += 1
        elif judge_pred == 'FALSE':
            all_results['summary']['forward_judge_false'] += 1
        if not skip_reset:
            if result['reset']['execution_success']:
                all_results['summary']['reset_success'] += 1
            rj_pred = result['reset_judge'].get('prediction', 'UNCERTAIN')
            if rj_pred == 'TRUE':
                all_results['summary']['reset_judge_true'] += 1
            elif rj_pred == 'FALSE':
                all_results['summary']['reset_judge_false'] += 1

    def _print_summary(self, result: Dict, skip_reset: bool) -> None:
        """결과 요약 출력"""
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print("\n" + CYAN + "=" * 70 + RESET)
        print(CYAN + BOLD + "PIPELINE COMPLETE".center(70) + RESET)
        print(CYAN + "=" * 70 + RESET)

        forward_status = result['forward']['execution_success']
        forward_color = GREEN if forward_status else RED
        print(f"  Forward: {forward_color}{'SUCCESS' if forward_status else 'FAILED'}{RESET}")

        judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
        judge_color = GREEN if judge_pred == "TRUE" else RED if judge_pred == "FALSE" else YELLOW
        print(f"  Judge:   {judge_color}{judge_pred}{RESET}")

        if not skip_reset:
            reset_status = result['reset']['execution_success']
            reset_color = GREEN if reset_status else RED
            print(f"  Reset:   {reset_color}{'SUCCESS' if reset_status else 'FAILED'}{RESET}")

            rj_pred = result['reset_judge'].get('prediction', 'UNCERTAIN')
            rj_color = GREEN if rj_pred == "TRUE" else RED if rj_pred == "FALSE" else YELLOW
            print(f"  Reset Judge: {rj_color}{rj_pred}{RESET}")
        else:
            print(f"  Reset:   {YELLOW}SKIPPED{RESET}")

        print(CYAN + "=" * 70 + RESET)

    def _generate_seed_positions(self, session_dir: str, seed_index: int) -> Optional[Dict]:
        """
        새 seed 위치 생성: 랜덤 위치 생성 → IK dry_run 검증.

        first_episode_positions를 기반으로 랜덤 위치를 생성합니다.
        first_episode_positions는 변경하지 않습니다.

        Returns:
            성공 시 새 positions dict, 실패 시 None
        """
        from code_gen_lerobot.reset_execution.workspace import (
            generate_random_positions, classify_objects, ResetWorkspace,
        )

        if self.first_episode_positions is None:
            print("  [SeedGen] No first_episode_positions, cannot generate")
            return None

        save_dir = str(Path(session_dir) / f"seed_{seed_index:02d}_setup")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        grippable, obstacles = classify_objects(self.first_episode_positions)

        # pix2robot 로드
        pix2robot = None
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            calib_path = Path(__file__).parent / "robot_configs" / "pix2robot_matrices" / f"robot{self.robot_id}_pix2robot_data.npz"
            if calib_path.exists():
                pix2robot = Pix2RobotCalibrator(robot_id=self.robot_id)
                if not pix2robot.load(str(calib_path)):
                    pix2robot = None
        except Exception:
            pass

        # Workspace
        kin_engine = None
        try:
            from lerobot_cap.kinematics.engine import KinematicsEngine
            urdf_path = Path(__file__).parent / "assets" / "urdf" / f"so101_robot{self.robot_id}.urdf"
            if urdf_path.exists():
                kin_engine = KinematicsEngine(str(urdf_path))
        except Exception:
            pass
        workspace = ResetWorkspace(kinematics_engine=kin_engine)

        # 과거 모든 시드 위치 + 현재 초기 위치를 합쳐서 겹침 방지
        all_initial = dict(self.first_episode_positions)
        for i, prev_positions in enumerate(self._all_previous_seed_positions):
            for name, info in prev_positions.items():
                all_initial[f"{name}_seed{i}"] = info

        # 랜덤 위치 생성 + dry_run 검증 (최대 10회 재시도)
        reset_code = self.cached_reset_code
        accepted_positions = None

        for attempt in range(10):
            random_targets = generate_random_positions(
                grippable_objects=grippable,
                obstacle_objects=obstacles,
                initial_positions=all_initial,
                workspace=workspace,
                pix2robot=pix2robot,
            )
            if not random_targets:
                print(f"  [SeedGen] Attempt {attempt+1}: position generation failed, retrying...")
                continue

            candidate = self._build_batch_positions(random_targets, obstacles, pix2robot=pix2robot)

            # dry_run 검증: 캐싱된 reset 코드의 target 좌표를 새 seed로 치환 후 IK 가상 실행
            if reset_code is not None:
                print(f"  [SeedGen] Attempt {attempt+1}: dry-run validating...")
                patched_code = self._patch_reset_code_targets(reset_code, candidate)
                if self.dry_run_code(patched_code, candidate):
                    print(f"  [SeedGen] Attempt {attempt+1}: dry-run PASSED")
                    accepted_positions = candidate
                    break
                else:
                    print(f"  [SeedGen] Attempt {attempt+1}: dry-run FAILED, retrying...")
            else:
                # reset 코드 없으면 기본 IK 검증만으로 채택
                accepted_positions = candidate
                break
        else:
            print(f"  [SeedGen] All 10 attempts failed")
            return None

        new_positions = accepted_positions

        print(f"  [SeedGen] seed_{seed_index} positions:")
        for name, info in new_positions.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos and len(pos) >= 3:
                print(f"    {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        # 과거 시드 위치에 추가 (다음 시드 생성 시 겹침 방지) — grippable만
        from code_gen_lerobot.reset_execution.workspace import is_grippable as _is_grip
        self._all_previous_seed_positions.append(
            {name: info for name, info in new_positions.items()
             if isinstance(info, dict) and _is_grip(info.get("bbox_px"))}
        )

        # 시각화: workspace + 과거 시드 bbox + 새 시드 bbox
        self._visualize_seed_positions(
            save_dir=save_dir,
            new_positions=new_positions,
            seed_index=seed_index,
            pix2robot=pix2robot,
        )

        # 로그 저장
        with open(str(Path(save_dir) / "seed_positions.json"), 'w') as f:
            json.dump({"seed_index": seed_index, "positions": new_positions}, f, indent=2, default=str)

        return new_positions

    def _visualize_seed_positions(
        self,
        save_dir: str,
        new_positions: Dict,
        seed_index: int,
        pix2robot=None,
    ):
        """
        시드 위치 시각화: workspace 위에 과거/현재 시드 bbox를 그림.

        - workspace 도넛 마스크 + 가장자리 마진
        - 과거 시드 (초기 포함): 회색 계열 bbox (seed 번호 라벨)
        - 장애물: 빨간색 bbox
        - 새 시드: 초록색 bbox (굵게)
        """
        import cv2
        from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image, _get_bbox_px, is_grippable as _is_grippable

        # 초기 이미지 로드
        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
            base_img = cv2.imread(self.forward_initial_image_path)
        else:
            base_img = np.zeros((480, 640, 3), dtype=np.uint8) + 60

        # workspace 시각화 베이스
        result = draw_workspace_on_image(base_img, robot_id=self.robot_id, pix2robot_calibrator=pix2robot)

        def _draw_bbox(img, center_px, bbox_px, color, thickness, label=""):
            hw, hh = bbox_px[0] // 2, bbox_px[1] // 2
            cu, cv = int(center_px[0]), int(center_px[1])
            cv2.rectangle(img, (cu - hw, cv - hh), (cu + hw, cv + hh), color, thickness)
            if label:
                cv2.putText(img, label, (cu - hw, cv - hh - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # 장애물 (빨간색)
        if self.first_episode_positions:
            for name, info in self.first_episode_positions.items():
                if not isinstance(info, dict):
                    continue
                if not _is_grippable(info.get("bbox_px")):
                    pos = info.get("position")
                    bbox = _get_bbox_px(info)
                    if pos and pix2robot:
                        try:
                            px = pix2robot.robot_to_pixel(pos[0], pos[1])
                            _draw_bbox(result, px, bbox, (0, 0, 255), 2, f"obs:{name}")
                        except Exception:
                            pass

        # 과거 시드 — 모두 회색 계열 (초기 위치 = s0)
        PAST_COLORS = [
            (150, 150, 150),  # 회색
            (180, 130, 180),  # 보라
            (130, 180, 180),  # 청록
            (180, 180, 130),  # 올리브
            (130, 130, 180),  # 남색
        ]
        for i, prev in enumerate(self._all_previous_seed_positions):
            color = PAST_COLORS[i % len(PAST_COLORS)]
            for name, info in prev.items():
                if not isinstance(info, dict):
                    continue
                pos = info.get("position")
                bbox = _get_bbox_px(info)
                if pos and pix2robot:
                    try:
                        px = pix2robot.robot_to_pixel(pos[0], pos[1])
                        _draw_bbox(result, px, bbox, color, 1, f"s{i}:{name}")
                    except Exception:
                        pass

        # 새 시드 (초록색, 굵게) — grippable만
        for name, info in new_positions.items():
            if not isinstance(info, dict):
                continue
            if not _is_grippable(info.get("bbox_px")):
                continue
            pos = info.get("position")
            bbox = _get_bbox_px(info)
            if pos and pix2robot:
                try:
                    px = pix2robot.robot_to_pixel(pos[0], pos[1])
                    _draw_bbox(result, px, bbox, (0, 255, 0), 3, f"NEW s{seed_index}:{name}")
                except Exception:
                    pass

        # 범례
        img_h = result.shape[0]
        cv2.putText(result, f"Seed {seed_index} | Past: {len(self._all_previous_seed_positions)} seeds",
                    (10, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = str(Path(save_dir) / "seed_visualization.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [SeedGen] Visualization saved: {out_path}")

    def _build_batch_positions(self, random_targets: Dict, obstacles: Dict, pix2robot=None) -> Dict:
        """랜덤 타겟과 obstacle을 합쳐 positions dict 구성. pixel 필드도 갱신."""
        new_positions = {}
        for name, pos in random_targets.items():
            orig = self.first_episode_positions.get(name, {})
            if isinstance(orig, dict):
                new_positions[name] = {**orig, "position": pos}
                # pixel 필드를 새 position에 맞게 갱신
                if pix2robot is not None:
                    try:
                        new_positions[name]["pixel"] = list(pix2robot.robot_to_pixel(pos[0], pos[1]))
                    except Exception:
                        pass
                if "points" in orig:
                    new_positions[name]["points"] = {
                        pt_name: pos for pt_name in orig["points"]
                    }
            else:
                new_positions[name] = pos

        for name, info in obstacles.items():
            new_positions[name] = self.first_episode_positions.get(name, info)

        return new_positions

    def _load_resume_state(self, session_dir: str) -> Tuple[List[int], List[Optional[Dict]]]:
        """이전 세션에서 상태 복원.

        Returns:
            (batch_successes, seed_positions):
                batch_successes[i] = 배치 i의 성공 에피소드 수
                seed_positions[i] = 배치 i의 seed 위치 (없으면 None)
        """
        import copy
        episodes_per_seed = max(1, self.total_episodes // self.num_random_seeds)
        batch_successes = [0] * self.num_random_seeds
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # 1. first_episode_positions 복원 (ep_01 execution_context)
        ctx_path = Path(session_dir) / "episode_01" / "forward" / "execution_context.json"
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            self.first_episode_positions = ctx.get("object_positions", {})
            seed_positions[0] = copy.deepcopy(self.first_episode_positions)
            print(f"  [Resume] first_episode_positions restored from {ctx_path}")

        # 2. seed positions 복원
        for i in range(1, self.num_random_seeds):
            sp_path = Path(session_dir) / f"seed_{i:02d}_setup" / "seed_positions.json"
            if sp_path.exists():
                with open(sp_path) as f:
                    sp = json.load(f)
                seed_positions[i] = sp.get("positions", None)
                print(f"  [Resume] seed_{i} restored from {sp_path}")

        # 3. 배치별 성공 에피소드 카운트
        for ep_dir in sorted(Path(session_dir).glob("episode_*")):
            ep_num = int(ep_dir.name.replace("episode_", ""))
            batch_idx = min((ep_num - 1) // episodes_per_seed, self.num_random_seeds - 1)

            judge_file = ep_dir / "forward" / "judge_result.json"
            if judge_file.exists():
                with open(judge_file) as f:
                    jr = json.load(f)
                pred = jr.get("judge_result", {}).get("prediction", jr.get("prediction", ""))
                if pred == "TRUE":
                    batch_successes[batch_idx] += 1

        # 4. 복원된 seed positions를 _all_previous_seed_positions에 등록 (겹침 방지)
        for i, sp in enumerate(seed_positions):
            if sp is not None:
                self._all_previous_seed_positions.append(sp)

        print(f"  [Resume] Batch successes: {batch_successes} (target: {episodes_per_seed} each)")
        print(f"  [Resume] Previous seeds registered: {len(self._all_previous_seed_positions)}")
        return batch_successes, seed_positions

    def _restore_to_seed(
        self,
        target_positions: Dict,
        instruction: str,
        detection_timeout: float = 10.0,
    ) -> bool:
        """물체를 seed 위치로 복원 (레코딩 없음).

        현재 물체 위치를 검출하고, target_positions로 이동하는 Reset 코드를 생성/실행.
        """
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET_C = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET_C}")
        print(f"{CYAN}{BOLD}  RESTORE TO SEED POSITION (no recording){RESET_C}")
        print(f"{CYAN}{BOLD}{'=' * 60}{RESET_C}")

        for name, info in target_positions.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos and len(pos) >= 3:
                print(f"  Target: {name} → [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        try:
            # 카메라로 현재 상태 캡처
            if not self.camera and not (self.camera_manager and self.camera_manager.is_connected):
                self.initialize_camera()
            time.sleep(0.5)
            current_frame = self.capture_frame()
            if current_frame is None:
                print(f"  {RED}Failed to capture frame{RESET_C}")
                return False

            # 이미지 저장 (임시)
            import tempfile
            tmp_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(tmp_path, current_frame)

            # Reset 코드 생성
            reset_code, _, current_pos, _ = self.generate_reset_code(
                original_instruction=instruction,
                original_positions=target_positions,
                current_state_image_path=tmp_path,
            )

            # Reset 코드 실행 (레코딩 임시 비활성화)
            self.shutdown_camera()
            saved_record = self.record_dataset
            self.record_dataset = False
            try:
                success = self.execute_code(reset_code, current_pos)
            finally:
                self.record_dataset = saved_record

            if success:
                print(f"  {GREEN}Restore to seed: SUCCESS{RESET_C}")
            else:
                print(f"  {RED}Restore to seed: FAILED{RESET_C}")

            return success

        except Exception as e:
            print(f"  {RED}Restore to seed error: {e}{RESET_C}")
            import traceback
            traceback.print_exc()
            return False

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
        resume_session_dir: Optional[str] = None,
    ) -> Dict:
        """
        여러 에피소드 연속 실행

        Args:
            num_episodes: 실행할 에피소드 수
            instruction: 자연어 명령어
            objects: 검출할 객체 리스트
            detection_timeout: 검출 타임아웃
            visualize_detection: 검출 시각화 여부
            save_dir: 결과 저장 디렉토리
            skip_reset: Reset 단계 건너뛰기

        Returns:
            전체 에피소드 결과 딕셔너리
        """
        # 색상 코드
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        # 결과 저장 디렉토리 설정
        if save_dir is None:
            save_dir = "results"

        # Resume 모드: 이전 세션 디렉토리 재사용
        if resume_session_dir:
            session_dir = str(resume_session_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = str(Path(save_dir) / f"session_{timestamp}")
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        # Set total episodes for logging
        self.total_episodes = num_episodes

        # 배치 계산
        episodes_per_seed = max(1, num_episodes // self.num_random_seeds)

        # 배치별 seed positions 초기화
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds
        batch_successes = [0] * self.num_random_seeds

        # Resume: 이전 세션에서 상태 복원
        if resume_session_dir:
            batch_successes, seed_positions = self._load_resume_state(session_dir)

        # 전체 결과 저장
        all_results = {
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'session_dir': session_dir,
            'episodes': [],
            'summary': {
                'forward_success': 0,
                'forward_judge_true': 0,
                'forward_judge_false': 0,
                'reset_success': 0,
                'reset_judge_true': 0,
                'reset_judge_false': 0,
            },
        }

        print("\n" + MAGENTA + "=" * 70 + RESET)
        if resume_session_dir:
            print(MAGENTA + BOLD + f"  RESUME SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        else:
            print(MAGENTA + BOLD + f"  MULTI-EPISODE SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        if resume_session_dir:
            print(f"  Resume from: {session_dir}")
            for i, (s, sp) in enumerate(zip(batch_successes, seed_positions)):
                status = "DONE" if s >= episodes_per_seed else f"{s}/{episodes_per_seed}"
                seed_str = "loaded" if sp else "to generate"
                print(f"    Batch {i}: {status} ({seed_str})")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        # ================================================================
        # 에피소드 루프
        # ================================================================
        # 새 세션: 고정 스케줄 (NUM_EPISODES개 순차 실행, 실패해도 스킵 없이 진행)
        # Resume: 미완료 배치만 실행 (성공할 때까지 재시도)
        # ================================================================

        if resume_session_dir:
            # Resume 모드: 첫 미완료 배치의 seed로 복원 후 재시도 루프
            first_incomplete = None
            for i in range(self.num_random_seeds):
                if batch_successes[i] < episodes_per_seed:
                    first_incomplete = i
                    break

            if first_incomplete is not None:
                # 복원: 해당 seed 위치로 이동 (1회만)
                if seed_positions[first_incomplete] is None and first_incomplete > 0:
                    print(f"\n{MAGENTA}{BOLD}  [Resume] Generating seed_{first_incomplete}...{RESET}")
                    seed_positions[first_incomplete] = self._generate_seed_positions(session_dir, first_incomplete)
                if seed_positions[first_incomplete] is not None:
                    print(f"\n{CYAN}{BOLD}  [Resume] Restoring to seed_{first_incomplete}...{RESET}")
                    self._restore_to_seed(seed_positions[first_incomplete], instruction, detection_timeout)

            global_episode_num = sum(batch_successes)

            for batch_index in range(self.num_random_seeds):
                needed = episodes_per_seed - batch_successes[batch_index]
                if needed <= 0:
                    print(f"\n{GREEN}  [Batch {batch_index}] Already complete ({batch_successes[batch_index]}/{episodes_per_seed}), skipping{RESET}")
                    continue

                # Seed 위치 확보 (resume 복원 이후 배치)
                if seed_positions[batch_index] is None and batch_index > 0:
                    print(f"\n{MAGENTA}{BOLD}  [Batch {batch_index}] Generating seed_{batch_index}...{RESET}")
                    seed_positions[batch_index] = self._generate_seed_positions(session_dir, batch_index)
                    if seed_positions[batch_index] is None:
                        print(f"  {RED}[Batch {batch_index}] Seed generation failed, skipping batch{RESET}")
                        continue

                success_count = 0
                max_attempts = needed * 3

                for attempt in range(max_attempts):
                    if success_count >= needed:
                        break
                    global_episode_num += 1
                    self.current_episode = global_episode_num

                    print("\n" + CYAN + "=" * 70 + RESET)
                    print(CYAN + BOLD + f"  [Resume Batch {batch_index}] {success_count+1}/{needed} (attempt {attempt+1})  ".center(70) + RESET)
                    print(CYAN + "=" * 70 + RESET)

                    episode_dir = str(Path(session_dir) / f"episode_{global_episode_num:02d}")

                    # 마지막 필요 에피소드면 다음 seed로 전환
                    is_last = (success_count == needed - 1)
                    next_batch = batch_index + 1
                    if is_last and next_batch < self.num_random_seeds:
                        if seed_positions[next_batch] is None:
                            seed_positions[next_batch] = self._generate_seed_positions(session_dir, next_batch)
                        reset_target = seed_positions[next_batch] if seed_positions[next_batch] else seed_positions[batch_index]
                    else:
                        reset_target = seed_positions[batch_index]

                    try:
                        result = self.run(
                            instruction=instruction, objects=objects,
                            detection_timeout=detection_timeout,
                            visualize_detection=visualize_detection,
                            save_dir=episode_dir, use_timestamp_subdir=False,
                            skip_reset=skip_reset, reset_target_positions=reset_target,
                        )
                        judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                        if judge_pred == 'TRUE':
                            success_count += 1
                        self._update_results(all_results, result, global_episode_num, skip_reset)
                    except Exception as e:
                        print(f"\n{RED}[Resume Batch {batch_index}] Error: {e}{RESET}")
                        import traceback; traceback.print_exc()
                        all_results['episodes'].append({'episode': global_episode_num, 'result': None, 'success': False, 'error': str(e)})

                    time.sleep(2)

        else:
            # 새 세션: 고정 스케줄 (NUM_EPISODES개, 실패해도 계속 진행)
            for episode_idx in range(num_episodes):
                episode_num = episode_idx + 1
                batch_index = min(episode_idx // episodes_per_seed, self.num_random_seeds - 1)
                is_batch_last = (episode_idx % episodes_per_seed == episodes_per_seed - 1) or (episode_idx == num_episodes - 1)
                next_batch_index = batch_index + 1

                self.current_episode = episode_num

                print("\n" + CYAN + "=" * 70 + RESET)
                print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index})  ".center(70) + RESET)
                print(CYAN + "=" * 70 + RESET)

                episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")

                # Reset target: 기본은 현재 seed, 배치 전환 시 콜백으로 갱신
                reset_target = seed_positions[batch_index]

                # 배치 마지막이면: Forward 후 다음 seed 생성 → Reset target 갱신 콜백
                pre_reset_cb = None
                if is_batch_last and next_batch_index < self.num_random_seeds:
                    _next_idx = next_batch_index
                    _sp = seed_positions
                    _sd = session_dir
                    def _make_next_seed(next_idx=_next_idx, sp=_sp, sd=_sd):
                        # first_episode_positions 초기 설정 (아직 안 됐으면)
                        if sp[0] is None and self.first_episode_positions is not None:
                            import copy
                            sp[0] = copy.deepcopy(self.first_episode_positions)
                            self._all_previous_seed_positions.append(sp[0])
                        # 다음 seed 생성
                        if sp[next_idx] is None:
                            print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{next_idx}...{RESET}")
                            sp[next_idx] = self._generate_seed_positions(sd, next_idx)
                        return sp[next_idx]
                    pre_reset_cb = _make_next_seed

                try:
                    result = self.run(
                        instruction=instruction, objects=objects,
                        detection_timeout=detection_timeout,
                        visualize_detection=visualize_detection,
                        save_dir=episode_dir, use_timestamp_subdir=False,
                        skip_reset=skip_reset, reset_target_positions=reset_target,
                        pre_reset_callback=pre_reset_cb,
                    )

                    # first_episode_positions 초기 설정 (콜백에서 안 됐을 수 있음)
                    if seed_positions[0] is None and self.first_episode_positions is not None:
                        import copy
                        seed_positions[0] = copy.deepcopy(self.first_episode_positions)
                        self._all_previous_seed_positions.append(seed_positions[0])

                    self._update_results(all_results, result, episode_num, skip_reset)

                except Exception as e:
                    print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                    import traceback; traceback.print_exc()
                    all_results['episodes'].append({'episode': episode_num, 'result': None, 'success': False, 'error': str(e)})

                time.sleep(2)

        # 최종 요약 출력
        self._print_final_summary(all_results, skip_reset)

        # 세션 요약 JSON 저장
        summary_path = Path(session_dir) / "session_summary.json"
        summary_data = {
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'summary': all_results['summary'],
            'episodes': [
                {
                    'episode': ep['episode'],
                    'success': ep['success'],
                    'error': ep['error'],
                    'forward_success': ep['result']['forward']['execution_success'] if ep['result'] else False,
                    'forward_judge': ep['result']['judge'].get('prediction', 'N/A') if ep['result'] else 'N/A',
                    'reset_success': ep['result']['reset']['execution_success'] if ep['result'] and not skip_reset else 'N/A',
                    'reset_judge': ep['result']['reset_judge'].get('prediction', 'N/A') if ep['result'] and not skip_reset else 'N/A',
                }
                for ep in all_results['episodes']
            ],
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Session Summary] Saved to: {summary_path}")

        # 레코딩 모드: 세션 종료 시 데이터셋 finalize
        if self.record_dataset:
            self._finalize_recording()

        return all_results

    def _print_final_summary(self, all_results: Dict, skip_reset: bool) -> None:
        """전체 에피소드 최종 요약 출력"""
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        n = all_results['num_episodes']
        s = all_results['summary']

        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  FINAL SUMMARY ({n} Episodes)  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)

        # Forward 결과
        fwd_rate = s['forward_success'] / n * 100 if n > 0 else 0
        fwd_color = GREEN if fwd_rate >= 80 else YELLOW if fwd_rate >= 50 else RED
        print(f"  Forward Success:    {fwd_color}{s['forward_success']}/{n} ({fwd_rate:.1f}%){RESET}")

        # Forward Judge 결과
        judge_true_rate = s['forward_judge_true'] / n * 100 if n > 0 else 0
        judge_color = GREEN if judge_true_rate >= 80 else YELLOW if judge_true_rate >= 50 else RED
        print(f"  Forward Judge TRUE: {judge_color}{s['forward_judge_true']}/{n} ({judge_true_rate:.1f}%){RESET}")

        if not skip_reset:
            # Reset 결과
            reset_rate = s['reset_success'] / n * 100 if n > 0 else 0
            reset_color = GREEN if reset_rate >= 80 else YELLOW if reset_rate >= 50 else RED
            print(f"  Reset Success:      {reset_color}{s['reset_success']}/{n} ({reset_rate:.1f}%){RESET}")

            # Reset Judge 결과
            rj_true_rate = s['reset_judge_true'] / n * 100 if n > 0 else 0
            rj_color = GREEN if rj_true_rate >= 80 else YELLOW if rj_true_rate >= 50 else RED
            print(f"  Reset Judge TRUE:   {rj_color}{s['reset_judge_true']}/{n} ({rj_true_rate:.1f}%){RESET}")

        print(MAGENTA + "=" * 70 + RESET)


def main():
    parser = argparse.ArgumentParser(
        description="Forward + Reset Integrated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수 인자
    parser.add_argument(
        "--instruction", "-i",
        type=str,
        required=True,
        help="Natural language instruction"
    )

    parser.add_argument(
        "--objects", "-o",
        type=str,
        nargs="+",
        required=True,
        help="Objects to detect"
    )

    # 선택 인자
    parser.add_argument(
        "--robot", "-r",
        type=int,
        default=3,
        help="Robot number (default: 3)"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for code generation (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge VLM model (default: gpt-4o)"
    )

    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=10.0,
        help="Detection timeout in seconds (default: 10)"
    )

    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=5.0,
        help="Judge UI timeout in seconds (default: 5.0)"
    )

    parser.add_argument(
        "--visualize-detection",
        action="store_true",
        help="Show real-time detection visualization"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous session directory (e.g., results/session_20260319_174942)"
    )

    parser.add_argument(
        "--save", "-s",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )

    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip reset execution phase"
    )

    parser.add_argument(
        "--num-random-seeds",
        type=int,
        default=1,
        help="Number of random position batches (1=keep initial positions, N>1=N different random layouts)"
    )

    parser.add_argument(
        "--use-server",
        action="store_true",
        help="Use vLLM server for LLM/VLM inference instead of API"
    )

    # CodeGen LLM 서버 설정
    parser.add_argument(
        "--codegen-server-url",
        type=str,
        default="http://localhost:8001/v1",
        help="CodeGen LLM server URL (default: http://localhost:8001/v1)"
    )
    parser.add_argument(
        "--codegen-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="CodeGen LLM model name (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    )

    # Judge VLM 서버 설정
    parser.add_argument(
        "--judge-server-url",
        type=str,
        default="http://localhost:8002/v1",
        help="Judge VLM server URL (default: http://localhost:8002/v1)"
    )
    parser.add_argument(
        "--judge-server-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Judge VLM model name (default: Qwen/Qwen2-VL-2B-Instruct)"
    )

    parser.add_argument(
        "--num-episodes", "-n",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )

    # LeRobot 데이터셋 레코딩 옵션
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable LeRobot dataset recording during forward execution"
    )

    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="LeRobot dataset repository ID (e.g., 'user/my_dataset'). If not specified, auto-generated."
    )

    parser.add_argument(
        "--recording-fps",
        type=int,
        default=30,
        help="Recording FPS for LeRobot dataset (default: 30)"
    )

    # Multi-turn 옵션
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Use crop-then-point multi-turn LLM code generation (requires Gemini model)"
    )

    parser.add_argument(
        "--cad-image-dirs",
        type=str,
        nargs="*",
        default=None,
        help="CAD reference image directories for Turn 0 scene understanding"
    )

    parser.add_argument(
        "--codegen-session2-model",
        type=str,
        default=None,
        help="Model for code generation Session 2 (context handoff). If not set, uses --llm model."
    )

    parser.add_argument(
        "--side-view-image",
        type=str,
        default=None,
        help="Side-view image path for Turn Test waypoint trajectory prediction"
    )

    args = parser.parse_args()

    # 서버 모드 설정 (환경변수로 전달)
    if args.use_server:
        import os
        os.environ["USE_LLM_SERVER"] = "1"
        os.environ["USE_VLM_SERVER"] = "1"  # Judge용 VLM 서버 모드 활성화
        # CodeGen LLM 서버 설정
        os.environ["VLLM_SERVER_URL"] = args.codegen_server_url
        os.environ["VLLM_MODEL_NAME"] = args.codegen_model
        # Judge VLM 서버 설정
        os.environ["JUDGE_SERVER_URL"] = args.judge_server_url
        os.environ["JUDGE_MODEL_NAME"] = args.judge_server_model

    # 파이프라인 실행
    pipeline = ForwardAndResetPipeline(
        robot_id=args.robot,
        llm_model=args.llm,
        judge_model=args.judge_model,
        judge_timeout_ms=int(args.judge_timeout * 1000),  # 초 → 밀리초 변환
        num_random_seeds=args.num_random_seeds,
        verbose=True,
        # LeRobot 데이터셋 레코딩 옵션
        record_dataset=args.record,
        dataset_repo_id=args.dataset_repo_id,
        resume_recording=bool(args.resume),
        # Multi-turn 옵션
        multi_turn=args.multi_turn,
        cad_image_dirs=args.cad_image_dirs,
        side_view_image=args.side_view_image,
        recording_fps=args.recording_fps,
        codegen_model=args.codegen_session2_model,
    )

    # 에피소드 실행 (항상 session 구조 사용)
    all_results = pipeline.run_multiple_episodes(
        num_episodes=args.num_episodes,
        instruction=args.instruction,
        objects=args.objects,
        detection_timeout=args.timeout,
        visualize_detection=args.visualize_detection,
        save_dir=args.save,
        skip_reset=args.skip_reset,
        resume_session_dir=args.resume,
    )

    # 종료 코드 결정 (성공률 기반)
    n = all_results['num_episodes']
    s = all_results['summary']
    success_rate = s['forward_judge_true'] / n if n > 0 else 0

    if success_rate >= 0.5:  # 50% 이상 성공
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
