#!/usr/bin/env python3
"""
Multi-Robot Pipeline

여러 로봇을 동시에 제어하면서 데이터셋 레코딩을 수행하는 파이프라인

파이프라인 흐름:
    [Detection] 한 번 실행 → 두 로봇 프레임으로 변환
         ↓
    [Code Gen] 병렬로 두 로봇 코드 생성
         ↓
    [Execution] 병렬로 두 로봇 실행 (각각 레코딩)
         ↓
    [Judge] 병렬로 두 로봇 평가
         ↓
    [Reset] 병렬로 두 로봇 리셋

Usage:
    python pipeline_multi/multi_robot_pipeline.py \\
        --robot-ids 2 3 \\
        --instruction "pick up the red cup" \\
        --objects "red cup" "blue box" \\
        --num-episodes 5 \\
        --record
"""

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .multi_robot_config import MultiRobotConfig, RobotConfig, create_default_multi_robot_config
from .detection_wrapper import SharedDetectionManager
from .execution_wrapper import ParallelExecutionManager, ParallelExecutionResults, SynchronizedExecutionManager, SyncBarrier
from .recording_wrapper import MultiRobotRecordingManager, create_recording_manager
from .results_saver import MultiRobotResultsSaver, create_task_name_from_instruction

# Multi-robot 전용 코드 생성 (towel folding context 포함)
from .code_gen_lerobot_multi.forward_execution.code_gen import single_robot_code_gen
from .code_gen_lerobot_multi.reset_execution.code_gen import single_robot_reset_code_gen

# 기존 파이프라인 import (execution, reset용)
from execution_forward_and_reset import ForwardAndResetPipeline


# 색상 코드
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class MultiRobotPipeline:
    """
    멀티로봇 파이프라인

    기존 ForwardAndResetPipeline 인스턴스를 래핑하여
    여러 로봇을 병렬로 제어합니다.
    """

    def __init__(
        self,
        config: MultiRobotConfig,
        verbose: bool = True,
    ):
        """
        Args:
            config: 멀티로봇 설정
            verbose: 상세 로그 출력
        """
        self.config = config
        self.verbose = verbose

        # 로봇별 파이프라인 인스턴스
        self.pipelines: Dict[int, ForwardAndResetPipeline] = {}

        # 공유 리소스
        self.detection_manager: Optional[SharedDetectionManager] = None
        self.execution_manager: Optional[ParallelExecutionManager] = None
        self.synchronized_manager: Optional[SynchronizedExecutionManager] = None
        self.recording_manager: Optional[MultiRobotRecordingManager] = None
        self.shared_camera: Optional[Any] = None  # 공유 카메라 인스턴스 (detect_objects용)

        # 상태
        self.current_episode = 0
        self.total_episodes = 1
        self.instruction = ""  # 현재 태스크 명령어

        # 세션 타임스탬프
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 결과 저장 관리자 (run_episode에서 초기화)
        self.results_saver: Optional[MultiRobotResultsSaver] = None

        # 초기화
        self._initialize()

    def _initialize(self):
        """파이프라인 초기화"""
        robot_ids = self.config.get_robot_ids()

        if self.verbose:
            print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}Multi-Robot Pipeline Initialization{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"  Robots: {robot_ids}")
            print(f"  LLM Model: {self.config.llm_model}")
            print(f"  Judge Model: {self.config.judge_model}")
            print(f"  Reset Mode: {self.config.reset_mode}")
            print(f"  Record Dataset: {self.config.record_dataset}")

        # 공유 Detection Manager 생성
        self.detection_manager = SharedDetectionManager(
            robot_ids=robot_ids,
            verbose=self.verbose,
        )

        # 병렬 실행 Manager 생성
        self.execution_manager = ParallelExecutionManager(
            robot_ids=robot_ids,
            verbose=self.verbose,
        )

        # 동기화된 실행 Manager 생성 (sync_barrier 지원)
        self.synchronized_manager = SynchronizedExecutionManager(
            robot_ids=robot_ids,
            verbose=self.verbose,
        )

        # 레코딩 Manager 생성 (옵션)
        if self.config.record_dataset:
            self.recording_manager = create_recording_manager(
                robot_configs=self.config.get_enabled_robots(),
                fps=self.config.recording_fps,
                config_yaml=self.config.camera_config_yaml,
                auto_initialize=True,
                verbose=self.verbose,
            )

        # 로봇별 ForwardAndResetPipeline 생성
        for robot_config in self.config.get_enabled_robots():
            robot_id = robot_config.robot_id

            # 레코딩 옵션 설정
            if self.config.record_dataset and self.recording_manager:
                recorder = self.recording_manager.get_recorder(robot_id)
                dataset_repo_id = recorder.repo_id if recorder else None
            else:
                dataset_repo_id = None

            pipeline = ForwardAndResetPipeline(
                robot_id=robot_id,
                llm_model=self.config.llm_model,
                judge_model=self.config.judge_model,
                reset_mode=self.config.reset_mode,
                verbose=self.verbose,
                record_dataset=False,  # 레코딩은 MultiRobotRecordingManager에서 관리
                dataset_repo_id=dataset_repo_id,
                recording_fps=self.config.recording_fps,
            )

            self.pipelines[robot_id] = pipeline

            if self.verbose:
                print(f"  [Robot {robot_id}] Pipeline created")

    def _get_or_create_shared_camera(self):
        """공유 카메라 인스턴스를 가져오거나 생성합니다."""
        if self.shared_camera is None:
            try:
                import sys
                sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))
                from object_detection.camera import RealSenseD435

                if self.verbose:
                    print(f"  {Colors.CYAN}[SharedCamera] Creating shared RealSense camera...{Colors.RESET}")

                self.shared_camera = RealSenseD435(width=640, height=480, fps=30)
                self.shared_camera.start()

                # 카메라 안정화 대기
                import time
                time.sleep(1.5)

                if self.verbose:
                    print(f"  {Colors.GREEN}[SharedCamera] Camera ready{Colors.RESET}")

            except Exception as e:
                print(f"  {Colors.RED}[SharedCamera] Failed to create camera: {e}{Colors.RESET}")
                self.shared_camera = None

        return self.shared_camera

    def _cleanup_shared_camera(self):
        """공유 카메라 정리"""
        if self.shared_camera is not None:
            try:
                if self.verbose:
                    print(f"  {Colors.CYAN}[SharedCamera] Stopping shared camera...{Colors.RESET}")
                self.shared_camera.stop()
                self.shared_camera = None
            except Exception as e:
                print(f"  {Colors.RED}[SharedCamera] Error stopping camera: {e}{Colors.RESET}")

    def run_shared_detection(
        self,
        objects: List[str],
        timeout: float = 10.0,
        visualize: bool = False,
    ) -> Dict[str, Dict]:
        """
        공유 감지 실행

        Args:
            objects: 검출할 객체 목록
            timeout: 감지 타임아웃
            visualize: 시각화 여부

        Returns:
            감지 결과 (world frame)
        """
        if self.verbose:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}[PHASE 1] SHARED DETECTION{Colors.RESET}")
            print(f"{Colors.YELLOW}{'-'*70}{Colors.RESET}")

        detected = self.detection_manager.run_shared_detection(
            queries=objects,
            timeout=timeout,
            visualize=visualize,
        )

        # 감지 결과 검증
        not_found = [k for k, v in detected.items() if v is None]
        if not_found:
            print(f"{Colors.RED}[Error] Objects not detected: {not_found}{Colors.RESET}")
            return {}

        return detected

    def run_parallel_code_generation(
        self,
        instruction: str,
        detected_positions: Dict[str, Dict],
    ) -> Dict[int, str]:
        """
        병렬 코드 생성

        Args:
            instruction: 자연어 명령어
            detected_positions: 감지된 위치 (world frame)

        Returns:
            {robot_id: generated_code}
        """
        if self.verbose:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}[PHASE 2] PARALLEL CODE GENERATION{Colors.RESET}")
            print(f"{Colors.YELLOW}{'-'*70}{Colors.RESET}")

        # 물체 할당: 각 로봇에게 가장 가까운 물체 할당
        object_assignments = self.detection_manager.assign_objects_to_robots(detected_positions)

        # 역할 결정: 물체 이름 기반으로 역할 자동 할당
        # Door hinge assembly: red → holder, pink → inserter
        role_assignments = {}
        for rid, obj_name in object_assignments.items():
            role = None
            if obj_name:
                obj_lower = obj_name.lower()
                if "red" in obj_lower:
                    role = "holder"
                elif "pink" in obj_lower:
                    role = "inserter"
            role_assignments[rid] = role

        if self.verbose:
            print(f"\n{Colors.CYAN}[Object & Role Assignments]{Colors.RESET}")
            for rid, obj_name in object_assignments.items():
                role = role_assignments.get(rid, "N/A")
                print(f"  Robot {rid} -> '{obj_name}' (role: {role})")
            print()

        def generate_code(robot_id: int) -> str:
            pipeline = self.pipelines[robot_id]

            # 에피소드 정보 설정
            pipeline.current_episode = self.current_episode
            pipeline.total_episodes = self.total_episodes

            # 위치 정보 설정
            pipeline.detected_positions = copy.deepcopy(detected_positions)

            # 할당된 물체 및 역할 정보
            assigned_object = object_assignments.get(robot_id)
            assigned_role = role_assignments.get(robot_id)

            # Multi-robot 전용 코드 생성 (towel folding context 포함)
            # 기존 ForwardAndResetPipeline.generate_forward_code() 대신 사용
            code, _ = single_robot_code_gen(
                instruction=instruction,
                object_positions=detected_positions,
                robot_id=robot_id,
                llm_model=self.config.llm_model,
                task_type=None,  # instruction에서 자동 감지
                assigned_object=assigned_object,
                assigned_role=assigned_role,
            )
            pipeline.generated_code = code

            return code

        results = self.execution_manager.execute_parallel(
            func=generate_code,
            timeout=60.0,
            description="code generation",
        )

        # 결과 추출 및 즉시 저장
        codes = {}
        for robot_id, result in results.results.items():
            if result.success:
                codes[robot_id] = result.result
                # 디버깅: 생성된 코드 정보 출력
                code_len = len(result.result) if result.result else 0
                has_sync = "sync_barrier" in (result.result or "")
                has_rotate = "rotate_90degree" in (result.result or "")
                print(f"  {Colors.GREEN}[Robot {robot_id}] Code generated: {code_len} chars, "
                      f"has sync_barrier: {has_sync}, has rotate: {has_rotate}{Colors.RESET}")

                # 코드 생성 직후 즉시 저장
                if self.results_saver and result.result:
                    save_path = self.results_saver.save_generated_code(
                        episode=self.current_episode,
                        robot_id=robot_id,
                        code=result.result,
                        phase="forward",
                    )
                    if save_path:
                        print(f"  {Colors.CYAN}[Robot {robot_id}] Code saved immediately: {save_path}{Colors.RESET}")
            else:
                codes[robot_id] = ""
                print(f"{Colors.RED}  [Robot {robot_id}] Code generation failed: {result.error}{Colors.RESET}")
                if result.traceback:
                    print(f"    Traceback: {result.traceback[:500]}")

        return codes

    def run_parallel_execution(
        self,
        instruction: str,
        detected_positions: Dict[str, Dict],
        generated_codes: Dict[int, str],
    ) -> ParallelExecutionResults:
        """
        동기화된 병렬 실행

        sync_barrier를 사용하여 로봇 간 정확한 동기화를 보장합니다.

        Args:
            instruction: 태스크 설명
            detected_positions: 감지된 위치
            generated_codes: 생성된 코드

        Returns:
            실행 결과
        """
        if self.verbose:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}[PHASE 3] SYNCHRONIZED PARALLEL EXECUTION{Colors.RESET}")
            print(f"{Colors.YELLOW}{'-'*70}{Colors.RESET}")
            print(f"  Using sync_barrier for exact synchronization")

        # 레코딩 시작 (활성화된 경우)
        if self.recording_manager:
            self.recording_manager.start_episodes(task=instruction)

        # sync_barrier 리셋
        self.synchronized_manager.sync_barrier.reset()

        def execute_robot_synchronized(robot_id: int) -> bool:
            pipeline = self.pipelines[robot_id]
            code = generated_codes.get(robot_id, "")

            if not code:
                return False

            # sync_barrier를 주입하여 코드 실행
            success = self._execute_code_with_sync(
                robot_id=robot_id,
                code=code,
                positions=detected_positions,
                sync_barrier=self.synchronized_manager.sync_barrier,
            )
            return success

        results = self.execution_manager.execute_parallel(
            func=execute_robot_synchronized,
            timeout=self.config.parallel_timeout,
            description="synchronized code execution",
        )

        return results

    def _execute_code_with_sync(
        self,
        robot_id: int,
        code: str,
        positions: Dict[str, Dict],
        sync_barrier: SyncBarrier,
    ) -> bool:
        """
        sync_barrier를 주입하여 코드 실행

        Args:
            robot_id: 로봇 ID
            code: 실행할 코드
            positions: 객체 위치
            sync_barrier: 동기화 배리어

        Returns:
            실행 성공 여부
        """
        import traceback
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        pipeline = self.pipelines[robot_id]

        # 로그 캡처용 StringIO
        log_buffer = io.StringIO()

        # Tee 클래스: stdout과 buffer 둘 다에 출력
        class TeeWriter:
            def __init__(self, original, buffer):
                self.original = original
                self.buffer = buffer

            def write(self, text):
                self.original.write(text)
                self.buffer.write(text)

            def flush(self):
                self.original.flush()
                self.buffer.flush()

        # 원본 stdout/stderr 저장
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Tee writer 설정 (콘솔 + 버퍼 동시 출력)
        tee_stdout = TeeWriter(original_stdout, log_buffer)
        tee_stderr = TeeWriter(original_stderr, log_buffer)

        success = False

        try:
            # stdout/stderr를 Tee로 리다이렉트
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            print(f"  {Colors.CYAN}[Robot {robot_id}] Starting code execution...{Colors.RESET}")

            # Initial 이미지 캡처 (메서드가 있는 경우에만)
            if hasattr(pipeline, 'capture_initial_image'):
                pipeline.capture_initial_image()
                print(f"    [Robot {robot_id}] Initial image captured")

            # 코드 길이 확인
            if not code or len(code.strip()) == 0:
                print(f"  {Colors.RED}[Robot {robot_id}] ERROR: Empty code!{Colors.RESET}")
                return False

            print(f"    [Robot {robot_id}] Code length: {len(code)} chars")

            # 공유 카메라를 builtins에 주입 (detect_objects 스킬용)
            import builtins
            shared_camera = self._get_or_create_shared_camera()
            if shared_camera is not None:
                builtins.shared_camera = shared_camera
                print(f"    [Robot {robot_id}] Shared camera injected into builtins")

            # 실행 컨텍스트 구성 (sync_barrier 및 multi_skills 함수 주입)
            # NOTE: __name__을 "__multi_pipeline__"으로 설정하여
            #       if __name__ == "__main__": 블록이 실행되지 않도록 함
            from pipeline_multi.multi_skills import (
                execute_multi_pick_object,
                execute_multi_place_object,
                execute_pause_for_sync,
            )
            exec_globals = {
                "sync_barrier": sync_barrier,
                "robot_id": robot_id,
                "positions": positions,
                "__name__": "__multi_pipeline__",
                # Multi-robot skill functions (pre-injected for reliability)
                "execute_multi_pick_object": execute_multi_pick_object,
                "execute_multi_place_object": execute_multi_place_object,
                "execute_pause_for_sync": execute_pause_for_sync,
            }

            # Debug: sync_barrier 상태 확인
            print(f"    [Robot {robot_id}] sync_barrier type: {type(sync_barrier)}, is None: {sync_barrier is None}")

            # 코드 실행 (함수 정의)
            print(f"    [Robot {robot_id}] Defining execute_task function...")
            exec(code, exec_globals)

            # Debug: exec 후 sync_barrier 상태 확인
            print(f"    [Robot {robot_id}] After exec - sync_barrier in globals: {'sync_barrier' in exec_globals}, value is None: {exec_globals.get('sync_barrier') is None}")

            # execute_task 함수 호출
            if "execute_task" in exec_globals:
                print(f"    [Robot {robot_id}] Calling execute_task()...")
                exec_globals["execute_task"]()
                print(f"  {Colors.GREEN}[Robot {robot_id}] execute_task() completed successfully{Colors.RESET}")
                success = True
            elif "execute_reset_task" in exec_globals:
                print(f"    [Robot {robot_id}] Calling execute_reset_task()...")
                exec_globals["execute_reset_task"]()
                print(f"  {Colors.GREEN}[Robot {robot_id}] execute_reset_task() completed successfully{Colors.RESET}")
                success = True
            else:
                print(f"  {Colors.RED}[Robot {robot_id}] ERROR: No execute_task/execute_reset_task found in code!{Colors.RESET}")
                # 코드 내용 일부 출력
                print(f"    Code preview (first 500 chars):\n{code[:500]}")
                success = False

        except Exception as e:
            print(f"  {Colors.RED}[Robot {robot_id}] Execution error: {type(e).__name__}: {e}{Colors.RESET}")
            traceback.print_exc()
            success = False

        finally:
            # stdout/stderr 복원
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # 로그 저장
            log_content = log_buffer.getvalue()
            if self.results_saver and log_content:
                try:
                    log_dir = self.results_saver.get_episode_dir(
                        self.current_episode, robot_id, "forward"
                    )
                    log_path = log_dir / "execution_log.txt"
                    log_path.write_text(log_content, encoding='utf-8')
                    print(f"    [Robot {robot_id}] Execution log saved: {log_path}")
                except Exception as log_err:
                    print(f"    [Robot {robot_id}] Warning: Failed to save log: {log_err}")

        return success

    def run_parallel_judge(
        self,
        instruction: str,
        detected_positions: Dict[str, Dict],
        generated_codes: Dict[int, str],
    ) -> Dict[int, Dict]:
        """
        병렬 Judge 실행

        Args:
            instruction: 태스크 설명
            detected_positions: 감지된 위치
            generated_codes: 생성된 코드

        Returns:
            {robot_id: judge_result}
        """
        if self.config.skip_judge:
            if self.verbose:
                print(f"\n{Colors.YELLOW}[PHASE 4] JUDGE - SKIPPED{Colors.RESET}")
            return {}

        if self.verbose:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}[PHASE 4] PARALLEL JUDGE{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'-'*70}{Colors.RESET}")

        def run_judge(robot_id: int) -> Dict:
            pipeline = self.pipelines[robot_id]
            code = generated_codes.get(robot_id, "")

            # Final 이미지 캡처
            time.sleep(0.5)
            pipeline.capture_final_image()

            if pipeline.initial_image is None or pipeline.final_image is None:
                return {"prediction": "UNCERTAIN", "reasoning": "Missing images"}

            result = pipeline.run_judge(
                instruction=instruction,
                positions=detected_positions,
                executed_code=code,
            )
            return result

        results = self.execution_manager.execute_parallel(
            func=run_judge,
            timeout=30.0,
            description="judge evaluation",
        )

        # 결과 추출
        judge_results = {}
        for robot_id, result in results.results.items():
            if result.success:
                judge_results[robot_id] = result.result
            else:
                judge_results[robot_id] = {
                    "prediction": "ERROR",
                    "reasoning": result.error,
                }

        return judge_results

    def run_parallel_reset(
        self,
        instruction: str,
        detected_positions: Dict[str, Dict],
        objects: List[str],
    ) -> ParallelExecutionResults:
        """
        병렬 Reset 실행

        공유 Detection → 병렬 코드 생성 → 병렬 실행 순서로 진행

        Args:
            instruction: 원본 명령어
            detected_positions: 원본 감지 위치 (Forward 시작 시)
            objects: 검출할 객체 목록

        Returns:
            실행 결과
        """
        if self.config.skip_reset:
            if self.verbose:
                print(f"\n{Colors.YELLOW}[PHASE 5] RESET - SKIPPED{Colors.RESET}")
            return ParallelExecutionResults()

        if self.verbose:
            print(f"\n{Colors.CYAN}{Colors.BOLD}[PHASE 5] PARALLEL RESET{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*70}{Colors.RESET}")

        # Step 1: 공유 Detection (현재 객체 위치 감지)
        if self.verbose:
            print(f"\n{Colors.YELLOW}[Reset] Running shared detection for current positions...{Colors.RESET}")

        current_detected = self.detection_manager.run_shared_detection(
            queries=objects,
            timeout=self.config.detection_timeout,
            visualize=False,
        )

        if not current_detected:
            if self.verbose:
                print(f"{Colors.RED}[Reset] Detection failed - skipping reset{Colors.RESET}")
            return ParallelExecutionResults()

        # Step 2: 병렬 코드 생성 및 실행
        def run_reset(robot_id: int) -> bool:
            pipeline = self.pipelines[robot_id]

            # Multi-robot 전용 리셋 코드 생성 (towel reset context 포함)
            # 기존 ForwardAndResetPipeline.generate_reset_code() 대신 사용
            reset_code, orig_pos, curr_pos, target_pos = single_robot_reset_code_gen(
                original_instruction=instruction,
                target_positions=detected_positions,  # 원래 위치로 복귀
                current_positions=current_detected,   # 현재 위치
                robot_id=robot_id,
                llm_model=self.config.llm_model,
                forward_spec=pipeline.generated_spec,
                forward_code=pipeline.generated_code,
                is_random_reset=False,
                task_type=None,  # instruction에서 자동 감지
            )

            if not reset_code:
                return False

            # Reset 코드 실행
            success = pipeline.execute_code(reset_code, target_pos)
            return success

        results = self.execution_manager.execute_parallel(
            func=run_reset,
            timeout=self.config.parallel_timeout,
            description="reset execution",
        )

        return results

    def run_episode(
        self,
        instruction: str,
        objects: List[str],
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        단일 에피소드 실행

        Args:
            instruction: 자연어 명령어
            objects: 검출할 객체 목록
            detection_timeout: 감지 타임아웃
            visualize_detection: 시각화 여부
            save_dir: 결과 저장 디렉토리

        Returns:
            에피소드 결과
        """
        robot_ids = self.config.get_robot_ids()
        self.instruction = instruction

        # 결과 저장 관리자 초기화 (첫 에피소드에서만)
        if self.results_saver is None:
            task_name = create_task_name_from_instruction(instruction)
            # 기본 저장 경로: pipeline_multi/results_multi/
            base_results_dir = Path(__file__).parent / "results_multi"
            self.results_saver = MultiRobotResultsSaver(
                base_dir=str(base_results_dir),
                task_name=task_name,
                session_timestamp=self.session_timestamp,
            )
            if self.verbose:
                print(f"  Results Dir: {self.results_saver.get_session_dir()}")

        ep_str = f"Episode {self.current_episode}/{self.total_episodes}"

        if self.verbose:
            print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{ep_str} - Multi-Robot Pipeline{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"  Instruction: {instruction}")
            print(f"  Objects: {objects}")
            print(f"  Robots: {robot_ids}")
            print(f"  Save Dir: {self.results_saver.get_session_dir()}")

        result = {
            "episode": self.current_episode,
            "instruction": instruction,
            "objects": objects,
            "robots": {},
            "detection": {},
            "summary": {
                "total_robots": len(robot_ids),
                "successful_executions": 0,
                "failed_executions": 0,
            },
        }

        try:
            # Phase 1: Shared Detection
            detected_positions = self.run_shared_detection(
                objects=objects,
                timeout=detection_timeout,
                visualize=visualize_detection,
            )

            if not detected_positions:
                result["summary"]["error"] = "Detection failed"
                return result

            result["detection"] = detected_positions

            # 첫 에피소드 위치 저장 (original reset용)
            for robot_id, pipeline in self.pipelines.items():
                if pipeline.first_episode_positions is None:
                    pipeline.first_episode_positions = copy.deepcopy(detected_positions)

            # Phase 2: Parallel Code Generation
            generated_codes = self.run_parallel_code_generation(
                instruction=instruction,
                detected_positions=detected_positions,
            )

            # Phase 3: Parallel Execution
            execution_results = self.run_parallel_execution(
                instruction=instruction,
                detected_positions=detected_positions,
                generated_codes=generated_codes,
            )

            # Phase 4: Parallel Judge
            judge_results = self.run_parallel_judge(
                instruction=instruction,
                detected_positions=detected_positions,
                generated_codes=generated_codes,
            )

            # 레코딩 에피소드 종료
            if self.recording_manager:
                # Judge 결과에 따라 discard 결정
                discard_dict = {}
                for robot_id in robot_ids:
                    exec_result = execution_results.get_result(robot_id)
                    judge_result = judge_results.get(robot_id, {})
                    prediction = judge_result.get("prediction", "UNCERTAIN")

                    # 실행 실패 또는 Judge FALSE면 버림
                    discard = not (exec_result and exec_result.success) or prediction == "FALSE"
                    discard_dict[robot_id] = discard

                self.recording_manager.end_episodes(discard_dict=discard_dict)

            # Phase 5: Parallel Reset
            if not self.config.skip_reset:
                reset_results = self.run_parallel_reset(
                    instruction=instruction,
                    detected_positions=detected_positions,
                    objects=objects,
                )

            # 결과 집계 및 저장
            for robot_id in robot_ids:
                exec_result = execution_results.get_result(robot_id)
                judge_result = judge_results.get(robot_id, {})
                code = generated_codes.get(robot_id, "")
                exec_success = exec_result.success if exec_result else False

                # 디버깅: 각 로봇 결과 상태 출력
                print(f"  [Robot {robot_id}] Code length: {len(code) if code else 0}, "
                      f"Exec result exists: {exec_result is not None}, "
                      f"Success: {exec_success}")

                robot_result = {
                    "code": code,
                    "execution_success": exec_success,
                    "execution_error": exec_result.error if exec_result and not exec_result.success else None,
                    "judge": judge_result,
                }

                result["robots"][robot_id] = robot_result

                if exec_result and exec_result.success:
                    result["summary"]["successful_executions"] += 1
                else:
                    result["summary"]["failed_executions"] += 1

                # === 결과 저장 (ALWAYS, regardless of success) ===
                if self.results_saver:
                    pipeline = self.pipelines.get(robot_id)

                    # 디렉토리 먼저 생성 보장
                    save_dir = self.results_saver.get_episode_dir(
                        self.current_episode, robot_id, "forward"
                    )
                    print(f"    [Robot {robot_id}] Saving results to: {save_dir}")

                    # Forward 결과 저장
                    if code:
                        self.results_saver.save_generated_code(
                            episode=self.current_episode,
                            robot_id=robot_id,
                            code=code,
                            phase="forward",
                        )
                    else:
                        # 빈 코드라도 기록
                        empty_code_path = save_dir / "generated_code_EMPTY.txt"
                        empty_code_path.write_text("# Code generation failed or returned empty")
                        print(f"    {Colors.YELLOW}[Robot {robot_id}] Warning: Empty code, saved marker file{Colors.RESET}")

                    self.results_saver.save_execution_context(
                        episode=self.current_episode,
                        robot_id=robot_id,
                        instruction=instruction,
                        positions=detected_positions,
                        code=code if code else "# EMPTY",
                        success=exec_success,
                        phase="forward",
                    )

                    # 이미지 저장 (pipeline에서 가져오기)
                    if pipeline:
                        if hasattr(pipeline, 'detection_image') and pipeline.detection_image is not None:
                            self.results_saver.save_detection_image(
                                episode=self.current_episode,
                                robot_id=robot_id,
                                image=pipeline.detection_image,
                                phase="forward",
                            )
                        if hasattr(pipeline, 'initial_image') and pipeline.initial_image is not None:
                            self.results_saver.save_initial_image(
                                episode=self.current_episode,
                                robot_id=robot_id,
                                image=pipeline.initial_image,
                                phase="forward",
                            )
                        if hasattr(pipeline, 'final_image') and pipeline.final_image is not None:
                            self.results_saver.save_final_image(
                                episode=self.current_episode,
                                robot_id=robot_id,
                                image=pipeline.final_image,
                                phase="forward",
                            )

                    # Judge 결과 저장
                    if judge_result:
                        self.results_saver.save_judge_result(
                            episode=self.current_episode,
                            robot_id=robot_id,
                            prediction=judge_result.get("prediction", "UNCERTAIN"),
                            reasoning=judge_result.get("reasoning", ""),
                            phase="forward",
                        )

            # 에피소드 요약 저장
            if self.results_saver:
                self.results_saver.save_episode_summary(
                    episode=self.current_episode,
                    robot_results=result["robots"],
                    detection_results=detected_positions,
                    instruction=instruction,
                )

        except Exception as e:
            result["summary"]["error"] = str(e)
            import traceback
            result["summary"]["traceback"] = traceback.format_exc()
            print(f"{Colors.RED}[Error] Episode failed: {e}{Colors.RESET}")

        # 결과 요약 출력
        if self.verbose:
            print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}{ep_str} - Summary{Colors.RESET}")
            print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"  Success: {result['summary']['successful_executions']}/{result['summary']['total_robots']}")
            for robot_id, robot_result in result["robots"].items():
                status = "SUCCESS" if robot_result["execution_success"] else "FAILED"
                judge = robot_result.get("judge", {}).get("prediction", "N/A")
                print(f"  [Robot {robot_id}] Execution: {status}, Judge: {judge}")

        return result

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: List[str],
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        여러 에피소드 실행

        Args:
            num_episodes: 실행할 에피소드 수
            instruction: 자연어 명령어
            objects: 검출할 객체 목록
            detection_timeout: 감지 타임아웃
            visualize_detection: 시각화 여부
            save_dir: 결과 저장 디렉토리

        Returns:
            전체 결과
        """
        self.total_episodes = num_episodes

        # 저장 디렉토리 설정
        if save_dir is None:
            save_dir = Path(self.config.save_dir) / self.session_timestamp
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}Multi-Robot Pipeline - {num_episodes} Episodes{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"  Save Directory: {save_dir}")

        all_results = {
            "session_timestamp": self.session_timestamp,
            "config": {
                "robot_ids": self.config.get_robot_ids(),
                "num_episodes": num_episodes,
                "instruction": instruction,
                "objects": objects,
            },
            "episodes": [],
            "summary": {
                "total_episodes": num_episodes,
                "completed_episodes": 0,
                "total_robot_executions": 0,
                "successful_robot_executions": 0,
            },
        }

        try:
            for episode in range(1, num_episodes + 1):
                self.current_episode = episode

                episode_result = self.run_episode(
                    instruction=instruction,
                    objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=str(save_dir),
                )

                all_results["episodes"].append(episode_result)
                all_results["summary"]["completed_episodes"] += 1
                all_results["summary"]["total_robot_executions"] += episode_result["summary"]["total_robots"]
                all_results["summary"]["successful_robot_executions"] += episode_result["summary"]["successful_executions"]

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}[Interrupted] Stopping pipeline...{Colors.RESET}")

        finally:
            # 레코딩 Manager 종료
            if self.recording_manager:
                self.recording_manager.finalize_all()

            # 세션 요약 저장
            if self.results_saver:
                summary_path = self.results_saver.save_session_summary(
                    total_episodes=num_episodes,
                    completed_episodes=all_results["summary"]["completed_episodes"],
                    robot_ids=self.config.get_robot_ids(),
                    instruction=instruction,
                    all_results=all_results,
                )
                if self.verbose:
                    print(f"  Session summary saved: {summary_path}")

        # 최종 요약
        if self.verbose:
            summary = all_results["summary"]
            print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}Final Summary{Colors.RESET}")
            print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"  Completed Episodes: {summary['completed_episodes']}/{summary['total_episodes']}")
            print(f"  Robot Executions: {summary['successful_robot_executions']}/{summary['total_robot_executions']}")
            print(f"  Success Rate: {summary['successful_robot_executions']/max(1, summary['total_robot_executions'])*100:.1f}%")
            if self.results_saver:
                print(f"  Results saved to: {self.results_saver.get_session_dir()}")

        return all_results

    def cleanup(self):
        """리소스 정리"""
        if self.verbose:
            print(f"\n{Colors.CYAN}[Cleanup] Releasing resources...{Colors.RESET}")

        # 레코딩 Manager 종료
        if self.recording_manager and not self.recording_manager._finalized:
            self.recording_manager.finalize_all()

        # 공유 카메라 정리
        self._cleanup_shared_camera()

        # builtins에서 shared_camera 제거
        import builtins
        if hasattr(builtins, 'shared_camera'):
            delattr(builtins, 'shared_camera')

        # 파이프라인 정리
        for robot_id, pipeline in self.pipelines.items():
            if hasattr(pipeline, 'camera') and pipeline.camera:
                try:
                    pipeline.camera.stop()
                except Exception:
                    pass


def create_multi_robot_pipeline(
    robot_ids: List[int] = [2, 3],
    llm_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    reset_mode: str = "original",
    record_dataset: bool = True,
    recording_fps: int = 30,
    skip_judge: bool = False,
    skip_reset: bool = False,
    verbose: bool = True,
) -> MultiRobotPipeline:
    """
    MultiRobotPipeline 생성 헬퍼 함수

    Args:
        robot_ids: 로봇 ID 목록
        llm_model: LLM 모델
        judge_model: Judge 모델
        reset_mode: Reset 모드
        record_dataset: 레코딩 활성화
        recording_fps: 레코딩 FPS
        skip_judge: Judge 건너뛰기
        skip_reset: Reset 건너뛰기
        verbose: 상세 로그

    Returns:
        초기화된 MultiRobotPipeline
    """
    config = create_default_multi_robot_config(
        robot_ids=robot_ids,
        llm_model=llm_model,
        judge_model=judge_model,
        reset_mode=reset_mode,
        record_dataset=record_dataset,
        recording_fps=recording_fps,
        skip_judge=skip_judge,
        skip_reset=skip_reset,
    )

    return MultiRobotPipeline(config=config, verbose=verbose)


def parse_args():
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Multi-Robot Pipeline for parallel robot control and dataset recording"
    )

    # 필수 인자
    parser.add_argument(
        "-i", "--instruction",
        type=str,
        required=True,
        help="Natural language instruction (e.g., 'pick up the red cup')",
    )
    parser.add_argument(
        "-o", "--objects",
        type=str,
        nargs="+",
        required=True,
        help="Objects to detect (e.g., 'red cup' 'blue box')",
    )

    # 로봇 설정
    parser.add_argument(
        "--robot-ids",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Robot IDs to control (default: 2 3)",
    )

    # 에피소드 설정
    parser.add_argument(
        "-n", "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )

    # LLM 설정
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for code generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="VLM model for judge (default: gpt-4o)",
    )

    # 실행 옵션
    parser.add_argument(
        "--reset-mode",
        type=str,
        choices=["original", "random"],
        default="original",
        help="Reset mode (default: original)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip judge phase",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip reset phase",
    )

    # 감지 설정
    parser.add_argument(
        "--detection-timeout",
        type=float,
        default=10.0,
        help="Detection timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--visualize-detection",
        action="store_true",
        help="Visualize detection window",
    )

    # 레코딩 설정
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable dataset recording",
    )
    parser.add_argument(
        "--recording-fps",
        type=int,
        default=30,
        help="Recording FPS (default: 30)",
    )

    # 저장 설정
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./pipeline_multi/results_multi",
        help="Save directory for results (default: ./pipeline_multi/results_multi)",
    )

    # 기타
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def main():
    """CLI 진입점"""
    args = parse_args()

    # 설정 생성
    config = create_default_multi_robot_config(
        robot_ids=args.robot_ids,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        reset_mode=args.reset_mode,
        record_dataset=args.record,
        recording_fps=args.recording_fps,
        skip_judge=args.skip_judge,
        skip_reset=args.skip_reset,
        save_dir=args.save_dir,
    )

    # 파이프라인 생성 및 실행
    pipeline = MultiRobotPipeline(
        config=config,
        verbose=not args.quiet,
    )

    try:
        if args.num_episodes == 1:
            # 단일 에피소드
            result = pipeline.run_episode(
                instruction=args.instruction,
                objects=args.objects,
                detection_timeout=args.detection_timeout,
                visualize_detection=args.visualize_detection,
                save_dir=args.save_dir,
            )
        else:
            # 멀티 에피소드
            result = pipeline.run_multiple_episodes(
                num_episodes=args.num_episodes,
                instruction=args.instruction,
                objects=args.objects,
                detection_timeout=args.detection_timeout,
                visualize_detection=args.visualize_detection,
                save_dir=args.save_dir,
            )

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[Interrupted] Cleaning up...{Colors.RESET}")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
