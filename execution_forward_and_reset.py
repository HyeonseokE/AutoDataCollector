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
        reset_mode: str = "original",
        verbose: bool = True,
        # Recording options
        record_dataset: bool = False,
        dataset_repo_id: Optional[str] = None,
        recording_fps: int = 30,
    ):
        """
        초기화

        Args:
            robot_id: 로봇 번호 (2 또는 3)
            llm_model: 코드 생성용 LLM 모델 (예: "gpt-4o-mini", "gemini-1.5-flash")
            judge_model: Judge VLM 모델
            judge_timeout_ms: Judge UI 타임아웃 (밀리초)
            reset_mode: Reset 모드 ("original": 초기 위치, "random": 랜덤 위치)
            verbose: 상세 출력 여부
            record_dataset: LeRobot 데이터셋 레코딩 활성화
            dataset_repo_id: 데이터셋 저장 경로 (예: "user/my_dataset")
            recording_fps: 레코딩 FPS (기본: 30)
        """
        self.robot_id = robot_id
        self.llm_model = llm_model
        self.judge_model = judge_model
        self.judge_timeout_ms = judge_timeout_ms
        self.reset_mode = reset_mode
        self.verbose = verbose

        # Recording options
        self.record_dataset = record_dataset
        self.dataset_repo_id = dataset_repo_id
        self.recording_fps = recording_fps
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

        # Context 저장용
        self.execution_context: Dict = {}

        # Episode tracking for logging
        self.current_episode: int = 1
        self.total_episodes: int = 1
        self.current_phase: str = "Forward"  # "Forward" or "Reset"

        # First episode positions for "original" reset mode
        # Stores the first episode's detection result to avoid cumulative drift
        self.first_episode_positions: Optional[Dict] = None

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

    def initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            from object_detection.camera import RealSenseD435

            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            if self.verbose:
                print("[Pipeline] Camera initialized")
            return True
        except Exception as e:
            print(f"[Pipeline] Camera initialization failed: {e}")
            return False

    def shutdown_camera(self) -> None:
        """카메라 종료"""
        if self.camera:
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
                # robot_type은 config.py에서 자동으로 "so101_follower" 사용
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

        # Return extended format: {name: {"position": [x,y,z], "gripper_offset": float, ...}}
        # This includes gripper_offset calculated from bbox size
        return extended_results

    def generate_forward_code(self, instruction: str, positions: Dict) -> str:
        """Forward 코드 생성

        Args:
            instruction: 자연어 목표
            positions: Extended format {name: {"position": [x,y,z], "gripper_offset": float, ...}}
        """
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

    def execute_code(self, code: str, positions: Dict) -> bool:
        """생성된 코드 실행 (레코딩 모드 지원)"""
        try:
            exec_globals = {
                "__name__": "__main__",
                "positions": positions,
            }

            # 레코딩 모드: RecordingContext 설정
            # LeRobotSkills가 생성될 때 자동으로 콜백을 획득
            if self.record_dataset and self.dataset_recorder:
                return self._execute_code_with_recording(code, exec_globals)
            else:
                exec(code, exec_globals)
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
    ) -> Tuple[str, Dict, Dict, Dict]:
        """Reset 코드 생성

        Recording 카메라가 있으면 detection에 공유.

        Args:
            current_positions: 미리 감지된 현재 위치 (multi-robot 공유 감지용)
                              제공되면 내부 detection을 건너뜀

        Returns:
            Tuple[str, Dict, Dict, Dict]: (코드, 원래 위치, 현재 위치, 타겟 위치)
        """
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
            reset_mode=self.reset_mode,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            current_positions=current_positions,
        )

        return reset_code, orig_pos, current_pos, target_pos

    def run(
        self,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        use_timestamp_subdir: bool = True,
        skip_reset: bool = False,
        skip_judge: bool = False,
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
            skip_judge: Judge 단계 건너뛰기

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
                'mode': self.reset_mode,
                'current_positions': {},
                'target_positions': {},
                'code': '',
                'execution_success': False,
            },
            'reset_judge': {
                'prediction': 'UNCERTAIN',
                'reasoning': '',
                'reset_mode': self.reset_mode,
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
            print(f"  Workspace: x_min_world={workspace.x_min_world:.2f}m, reach=[{workspace.min_reach:.2f}, {workspace.max_reach:.2f}]m")

            # 경계값 정의
            X_WORKSPACE_MIN = workspace.x_min_world  # workspace 최소값 (기본 0.12m)
            X_WARNING_MAX = 0.14  # 이 값 미만이면 경고 (경계 근처)

            critical_error = False
            for obj_name, obj_info in self.detected_positions.items():
                if obj_info is None:
                    continue
                pos = obj_info.get("position") if isinstance(obj_info, dict) else obj_info
                if pos is None:
                    continue
                position_m = np.array([pos[0], pos[1], pos[2]])
                x_pos = pos[0]

                # Case 1: x < x_min_world (12cm) → workspace 밖, 파이프라인 종료
                if x_pos < X_WORKSPACE_MIN:
                    print(f"{RED}[CRITICAL] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m{RESET}")
                    print(f"{RED}  x={x_pos:.3f}m < {X_WORKSPACE_MIN}m (workspace limit) - OUTSIDE WORKSPACE!{RESET}")
                    print(f"{RED}  Pipeline will be terminated.{RESET}")
                    critical_error = True
                # Case 2: 12cm <= x < 14cm → 경계 근처, 경고만 출력
                elif x_pos < X_WARNING_MAX:
                    print(f"{YELLOW}[WARNING] Object '{obj_name}' at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m{RESET}")
                    print(f"{YELLOW}  x={x_pos:.3f}m is near workspace boundary ({X_WORKSPACE_MIN}m), continuing...{RESET}")
                # Case 3: 기타 workspace 검사 (reach limits 등)
                elif not workspace.is_reachable(position_m):
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
                print(f"  Initial image saved: {initial_path}")

            # Step 3: Forward 코드 생성
            print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model})...", step="Step 3/5") + f"{RESET}")
            self.generated_code = self.generate_forward_code(instruction, self.detected_positions)
            result['forward']['code'] = self.generated_code

            # [즉시 저장] Generated code
            code_path = Path(forward_dir) / "generated_code.py"
            code_path.write_text(self.generated_code)
            print(f"  Generated code saved: {code_path}")

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

            # 레코딩 모드: 에피소드 종료 (실패 시 버림)
            if self.record_dataset:
                # Judge 결과에 따라 버릴지 결정하기 위해 여기서는 종료하지 않음
                # Judge 후에 종료
                pass

            if forward_success:
                print(f"  {GREEN}Forward execution SUCCESS{RESET}")
            else:
                print(f"  {RED}Forward execution FAILED{RESET}")

            # Step 5: Context 저장
            print(f"\n{YELLOW}" + self._log("Saving execution context...", step="Step 5/5") + f"{RESET}")
            # Note: spec은 현재 code_gen에서 직접 생성하므로 빈 dict 사용
            # 향후 spec_gen을 사용하는 경우 여기서 spec도 저장
            self.generated_spec = {}  # TODO: spec_gen 통합 시 채우기
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
            time.sleep(1.0)  # 로봇 동작 완료 대기
            self.capture_final_image()
            if self.final_image is not None:
                print("  Final image captured")
                # [즉시 저장] Final 이미지
                final_path = Path(forward_dir) / "final_state.jpg"
                cv2.imwrite(str(final_path), self.final_image)  # Already BGR
                print(f"  Final image saved: {final_path}")

            # Step 2: Judge 실행
            if not skip_judge:
                print(f"\n{YELLOW}" + self._log(f"Running VLM Judge ({self.judge_model})...", step="Step 2/3", tag="Judge") + f"{RESET}")
                if self.initial_image is not None and self.final_image is not None:
                    judge_result = self.run_judge(
                        instruction=instruction,
                        positions=self.detected_positions,
                        executed_code=self.generated_code,
                    )
                    result['judge'] = judge_result

                    prediction = judge_result.get('prediction', 'UNCERTAIN')
                    reasoning = judge_result.get('reasoning', '')

                    # 색상 코딩
                    pred_color = GREEN if prediction == "TRUE" else RED if prediction == "FALSE" else YELLOW
                    print(f"  Prediction: {pred_color}{prediction}{RESET}")
                    print(f"  Reasoning: {reasoning[:100]}...")
                else:
                    print(f"  {YELLOW}Skipped (missing images){RESET}")

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
            else:
                print(f"\n{YELLOW}" + self._log("Judge skipped (--skip-judge)", step="Step 2/3", tag="Judge") + f"{RESET}")

            # 레코딩 모드: 에피소드 종료
            if self.record_dataset:
                judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                # Judge 결과가 FALSE면 에피소드 버림 (옵션)
                # 현재는 모든 에피소드 저장
                discard = False  # judge_pred == 'FALSE'
                self._end_episode_recording(discard=discard)

            # Forward 로깅 종료
            forward_log_path = forward_logger.stop()
            print(f"\n  Forward log saved to: {forward_log_path}")

            # ================================================================
            # PHASE 3: RESET EXECUTION
            # ================================================================
            if not skip_reset:
                # Switch phase for logging
                self.current_phase = "Reset"

                # Reset 로깅 시작
                reset_logger.start()

                mode_str = "RANDOM RESET" if self.reset_mode == "random" else "RESET TO ORIGINAL"
                print(f"\n{CYAN}{BOLD}" + self._log(mode_str) + f"{RESET}")
                print(CYAN + "-" * 70 + RESET)

                # 카메라 종료 (Forward에서 사용하던 별도 카메라)
                self.shutdown_camera()

                # Step 1 & 2: Reset 코드 생성 (내부에서 detection 수행)
                # Note: Recording 카메라가 있으면 reset_code_gen에서 자동 공유
                print(f"\n{YELLOW}" + self._log(f"Generating reset code ({self.reset_mode} mode)...", step="Step 1/4") + f"{RESET}")
                try:
                    # original 모드: 첫 에피소드의 위치를 사용 (누적 오차 방지)
                    # random 모드: 현재 에피소드의 검출 위치 사용
                    if self.reset_mode == "original" and self.first_episode_positions is not None:
                        reset_original_positions = self.first_episode_positions
                        print(f"  Using first episode positions for 'original' reset")
                    else:
                        reset_original_positions = self.detected_positions

                    reset_code, _, current_positions, target_positions = self.generate_reset_code(
                        original_instruction=instruction,
                        original_positions=reset_original_positions,
                        forward_spec=self.generated_spec,
                        forward_code=self.generated_code,
                        detection_timeout=detection_timeout,
                        visualize_detection=visualize_detection,
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
                        "reset_mode": self.reset_mode,
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
                    if not skip_judge:
                        print(f"\n{CYAN}" + self._log("Evaluating reset result...", tag="Judge") + f"{RESET}")
                        reset_judge_result = self.run_reset_judge(
                            reset_mode=self.reset_mode,
                            current_positions=current_positions,
                            target_positions=target_positions,
                            executed_code=reset_code,
                            original_instruction=instruction,
                        )
                        result['reset_judge'] = reset_judge_result

                        # Reset Judge 결과 출력
                        rj_pred = reset_judge_result.get('prediction', 'UNCERTAIN')
                        if rj_pred == "TRUE":
                            pred_color = GREEN
                        elif rj_pred == "FALSE":
                            pred_color = RED
                        else:
                            pred_color = YELLOW
                        print(f"  Prediction: {pred_color}{rj_pred}{RESET}")
                        rj_reasoning = reset_judge_result.get('reasoning', '')
                        if rj_reasoning:
                            # 첫 200자만 표시
                            reasoning_preview = rj_reasoning[:200]
                            if len(rj_reasoning) > 200:
                                reasoning_preview += "..."
                            print(f"  Reasoning: {reasoning_preview}")

                        # Reset Judge UI 표시
                        if self.reset_initial_image is not None and self.reset_final_image is not None:
                            self.show_reset_judge_ui(
                                reset_mode=self.reset_mode,
                                prediction=rj_pred,
                                reasoning=rj_reasoning,
                                current_positions=current_positions,
                                target_positions=target_positions,
                            )

                        # Reset judge 로그 저장 (judge 완료 후)
                        reset_log = {
                            'reset_mode': self.reset_mode,
                            'current_positions': current_positions,
                            'target_positions': target_positions,
                            'prediction': result['reset_judge'].get('prediction', 'UNCERTAIN'),
                            'reasoning': result['reset_judge'].get('reasoning', ''),
                            'execution_success': result['reset']['execution_success'],
                        }
                        reset_log_path = Path(reset_dir) / "reset_judge_result.json"
                        with open(reset_log_path, 'w') as f:
                            json.dump(reset_log, f, indent=2, default=str)
                        print(f"  Reset judge result saved to: {reset_log_path}")
                    else:
                        print(f"\n{CYAN}" + self._log("Reset Judge skipped (--skip-judge)", tag="Judge") + f"{RESET}")

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
            self._print_summary(result, skip_reset, skip_judge)

            return result

        finally:
            self.shutdown_camera()

    def _print_summary(self, result: Dict, skip_reset: bool, skip_judge: bool = False) -> None:
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

        # Forward 결과
        forward_status = result['forward']['execution_success']
        forward_color = GREEN if forward_status else RED
        print(f"  Forward: {forward_color}{'SUCCESS' if forward_status else 'FAILED'}{RESET}")

        # Judge 결과
        if not skip_judge:
            judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
            judge_color = GREEN if judge_pred == "TRUE" else RED if judge_pred == "FALSE" else YELLOW
            print(f"  Judge:   {judge_color}{judge_pred}{RESET}")
        else:
            print(f"  Judge:   {YELLOW}SKIPPED{RESET}")

        # Reset 결과
        if not skip_reset:
            reset_status = result['reset']['execution_success']
            reset_color = GREEN if reset_status else RED
            print(f"  Reset:   {reset_color}{'SUCCESS' if reset_status else 'FAILED'}{RESET}")

            # Reset Judge 결과
            if not skip_judge:
                rj_pred = result['reset_judge'].get('prediction', 'UNCERTAIN')
                rj_color = GREEN if rj_pred == "TRUE" else RED if rj_pred == "FALSE" else YELLOW
                print(f"  Reset Judge: {rj_color}{rj_pred}{RESET}")
            else:
                print(f"  Reset Judge: {YELLOW}SKIPPED{RESET}")
        else:
            print(f"  Reset:   {YELLOW}SKIPPED{RESET}")

        print(CYAN + "=" * 70 + RESET)

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
        skip_judge: bool = False,
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
            skip_judge: Judge 단계 건너뛰기

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path(save_dir) / f"session_{timestamp}")
        Path(session_dir).mkdir(parents=True, exist_ok=True)

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

        # Set total episodes for logging
        self.total_episodes = num_episodes

        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  MULTI-EPISODE SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        for episode_idx in range(num_episodes):
            episode_num = episode_idx + 1

            # Set current episode for logging
            self.current_episode = episode_num

            print("\n" + CYAN + "=" * 70 + RESET)
            print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Starting Episode  ".center(70) + RESET)
            print(CYAN + "=" * 70 + RESET)

            # 에피소드별 저장 디렉토리
            episode_dir = str(Path(session_dir) / f"episode_{episode_num:02d}")

            try:
                # 단일 에피소드 실행
                result = self.run(
                    instruction=instruction,
                    objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir,
                    use_timestamp_subdir=False,  # episode 폴더 안에 직접 forward/reset 생성
                    skip_reset=skip_reset,
                    skip_judge=skip_judge,
                )

                # 결과 저장
                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': result,
                    'success': True,
                    'error': None,
                })

                # 통계 업데이트
                if result['forward']['execution_success']:
                    all_results['summary']['forward_success'] += 1

                judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
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

            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                import traceback
                traceback.print_exc()

                all_results['episodes'].append({
                    'episode': episode_num,
                    'result': None,
                    'success': False,
                    'error': str(e),
                })

            # 에피소드 간 잠시 대기 (카메라 안정화)
            if episode_idx < num_episodes - 1:
                next_ep = episode_num + 1
                print(f"\n{YELLOW}[{next_ep:02d}/{num_episodes:02d}] Starting in 2 seconds...{RESET}")
                time.sleep(2)

        # 최종 요약 출력
        self._print_final_summary(all_results, skip_reset)

        # 세션 요약 JSON 저장
        summary_path = Path(session_dir) / "session_summary.json"
        summary_data = {
            'num_episodes': num_episodes,
            'instruction': instruction,
            'objects': objects,
            'timestamp': timestamp,
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
        choices=[2, 3],
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
        "--skip-judge",
        action="store_true",
        help="Skip all judge evaluation phases"
    )

    parser.add_argument(
        "--reset-mode",
        type=str,
        default="original",
        choices=["original", "random"],
        help="Reset mode: 'original' (restore to initial) or 'random' (shuffle to new positions)"
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
        reset_mode=args.reset_mode,
        verbose=True,
        # LeRobot 데이터셋 레코딩 옵션
        record_dataset=args.record,
        dataset_repo_id=args.dataset_repo_id,
        recording_fps=args.recording_fps,
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
        skip_judge=args.skip_judge,
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
