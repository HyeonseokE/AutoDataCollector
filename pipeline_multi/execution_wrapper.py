"""
Parallel Execution Manager for Multi-Robot Pipeline

ThreadPoolExecutor를 사용하여 여러 로봇의 작업을 병렬로 실행합니다.
"""

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar
import threading

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


T = TypeVar('T')


@dataclass
class ExecutionResult:
    """단일 로봇 실행 결과"""
    robot_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class ParallelExecutionResults:
    """병렬 실행 전체 결과"""
    results: Dict[int, ExecutionResult] = field(default_factory=dict)

    @property
    def all_success(self) -> bool:
        """모든 로봇 성공 여부"""
        return all(r.success for r in self.results.values())

    @property
    def any_success(self) -> bool:
        """하나 이상 성공 여부"""
        return any(r.success for r in self.results.values())

    @property
    def success_count(self) -> int:
        """성공한 로봇 수"""
        return sum(1 for r in self.results.values() if r.success)

    @property
    def failure_count(self) -> int:
        """실패한 로봇 수"""
        return sum(1 for r in self.results.values() if not r.success)

    def get_successful_robots(self) -> List[int]:
        """성공한 로봇 ID 목록"""
        return [rid for rid, r in self.results.items() if r.success]

    def get_failed_robots(self) -> List[int]:
        """실패한 로봇 ID 목록"""
        return [rid for rid, r in self.results.items() if not r.success]

    def get_result(self, robot_id: int) -> Optional[ExecutionResult]:
        """특정 로봇 결과 반환"""
        return self.results.get(robot_id)


class ParallelExecutionManager:
    """
    병렬 실행 관리자

    - ThreadPoolExecutor로 병렬 실행
    - 에러 격리: 한 로봇 실패해도 다른 로봇 계속
    - 타임아웃 지원
    """

    def __init__(
        self,
        robot_ids: List[int],
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            robot_ids: 로봇 ID 목록
            max_workers: 최대 동시 실행 수 (None이면 로봇 수)
            verbose: 상세 로그 출력
        """
        self.robot_ids = robot_ids
        self.max_workers = max_workers or len(robot_ids)
        self.verbose = verbose

        # 로봇별 상태 추적
        self._lock = threading.Lock()
        self._execution_status: Dict[int, str] = {}

    def execute_parallel(
        self,
        func: Callable[[int], T],
        timeout: Optional[float] = None,
        description: str = "task",
    ) -> ParallelExecutionResults:
        """
        병렬 실행

        Args:
            func: 로봇 ID를 인자로 받는 실행 함수
            timeout: 각 태스크의 최대 실행 시간 (초)
            description: 로그용 작업 설명

        Returns:
            ParallelExecutionResults
        """
        results = ParallelExecutionResults()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[ParallelExecution] Starting {description} for {len(self.robot_ids)} robots")
            print(f"  Robots: {self.robot_ids}")
            if timeout:
                print(f"  Timeout: {timeout}s per robot")
            print(f"{'='*60}")

        # 상태 초기화
        with self._lock:
            for robot_id in self.robot_ids:
                self._execution_status[robot_id] = "pending"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 로봇에 대해 작업 제출
            futures: Dict[Future, int] = {}

            for robot_id in self.robot_ids:
                future = executor.submit(self._execute_single, func, robot_id, description)
                futures[future] = robot_id

            # 결과 수집
            for future in as_completed(futures, timeout=timeout):
                robot_id = futures[future]

                try:
                    exec_result = future.result(timeout=0)  # 이미 완료됨
                    results.results[robot_id] = exec_result
                except Exception as e:
                    # Future 자체에서 예외 발생 (타임아웃 등)
                    results.results[robot_id] = ExecutionResult(
                        robot_id=robot_id,
                        success=False,
                        error=f"Future exception: {str(e)}",
                        traceback=traceback.format_exc(),
                    )

        # 결과 요약 출력
        if self.verbose:
            print(f"\n[ParallelExecution] {description} completed:")
            print(f"  Success: {results.success_count}/{len(self.robot_ids)}")
            if results.failure_count > 0:
                print(f"  Failed robots: {results.get_failed_robots()}")

        return results

    def _execute_single(
        self,
        func: Callable[[int], T],
        robot_id: int,
        description: str,
    ) -> ExecutionResult:
        """단일 로봇 실행 (스레드 내에서 호출)"""
        with self._lock:
            self._execution_status[robot_id] = "running"

        if self.verbose:
            print(f"  [Robot {robot_id}] Starting {description}...")

        try:
            result = func(robot_id)

            # Check if result indicates failure (e.g., False from _execute_code_with_sync)
            is_success = result is not False and result is not None

            with self._lock:
                self._execution_status[robot_id] = "completed" if is_success else "failed"

            if self.verbose:
                status = "SUCCESS" if is_success else "FAILED (returned False)"
                print(f"  [Robot {robot_id}] {description} {status}")

            return ExecutionResult(
                robot_id=robot_id,
                success=is_success,
                result=result,
            )

        except Exception as e:
            with self._lock:
                self._execution_status[robot_id] = "failed"

            error_msg = str(e)
            tb = traceback.format_exc()

            if self.verbose:
                print(f"  [Robot {robot_id}] {description} FAILED: {error_msg}")

            return ExecutionResult(
                robot_id=robot_id,
                success=False,
                error=error_msg,
                traceback=tb,
            )

    def execute_with_args(
        self,
        func: Callable[[int, Any], T],
        robot_args: Dict[int, Any],
        timeout: Optional[float] = None,
        description: str = "task",
    ) -> ParallelExecutionResults:
        """
        로봇별 다른 인자로 병렬 실행

        Args:
            func: (robot_id, args) -> result
            robot_args: {robot_id: args}
            timeout: 타임아웃 (초)
            description: 작업 설명

        Returns:
            ParallelExecutionResults
        """
        def wrapper(robot_id: int) -> T:
            args = robot_args.get(robot_id)
            return func(robot_id, args)

        return self.execute_parallel(wrapper, timeout=timeout, description=description)

    def get_execution_status(self) -> Dict[int, str]:
        """현재 실행 상태 반환"""
        with self._lock:
            return self._execution_status.copy()

    def barrier_wait(
        self,
        results: ParallelExecutionResults,
        require_all: bool = False,
    ) -> bool:
        """
        결과 확인 후 다음 단계 진행 여부 결정

        Args:
            results: 이전 단계 결과
            require_all: True면 모든 로봇 성공 필요, False면 하나라도 성공

        Returns:
            진행 가능 여부
        """
        if require_all:
            return results.all_success
        else:
            return results.any_success


def create_execution_manager(
    robot_ids: List[int],
    verbose: bool = True,
) -> ParallelExecutionManager:
    """ParallelExecutionManager 생성 헬퍼 함수"""
    return ParallelExecutionManager(robot_ids=robot_ids, verbose=verbose)


class SyncBarrier:
    """
    로봇 간 동기화를 위한 Barrier 래퍼

    생성된 코드에서 sync_barrier.wait("phase_name")을 호출하여 동기화 포인트 설정
    """

    def __init__(self, num_robots: int, verbose: bool = True):
        """
        Args:
            num_robots: 동기화에 참여할 로봇 수
            verbose: 상세 로그 출력
        """
        self.num_robots = num_robots
        self.verbose = verbose
        self._barriers: Dict[str, threading.Barrier] = {}
        self._lock = threading.Lock()

    def wait(self, phase_name: str = "sync") -> None:
        """
        동기화 포인트에서 대기

        Args:
            phase_name: 동기화 포인트 이름 (같은 이름끼리 동기화)
        """
        with self._lock:
            if phase_name not in self._barriers:
                self._barriers[phase_name] = threading.Barrier(self.num_robots)

        barrier = self._barriers[phase_name]

        if self.verbose:
            thread_name = threading.current_thread().name
            print(f"    [{thread_name}] Waiting at sync point: {phase_name}")

        barrier.wait()

        if self.verbose:
            thread_name = threading.current_thread().name
            print(f"    [{thread_name}] Passed sync point: {phase_name}")

    def reset(self, phase_name: str = None) -> None:
        """
        Barrier 리셋

        Args:
            phase_name: 특정 phase만 리셋 (None이면 전체)
        """
        with self._lock:
            if phase_name:
                if phase_name in self._barriers:
                    del self._barriers[phase_name]
            else:
                self._barriers.clear()


class SynchronizedExecutionManager:
    """
    동기화된 병렬 실행 관리자

    - 여러 로봇이 동시에 실행되면서 특정 지점에서 동기화
    - threading.Barrier를 사용하여 정확한 동기화 보장
    - 생성된 코드에 sync_barrier를 주입하여 사용
    """

    def __init__(
        self,
        robot_ids: List[int],
        verbose: bool = True,
    ):
        """
        Args:
            robot_ids: 로봇 ID 목록
            verbose: 상세 로그 출력
        """
        self.robot_ids = robot_ids
        self.verbose = verbose
        self.sync_barrier = SyncBarrier(len(robot_ids), verbose=verbose)
        self._parallel_manager = ParallelExecutionManager(robot_ids, verbose=verbose)

    def execute_code_synchronized(
        self,
        robot_codes: Dict[int, str],
        timeout: Optional[float] = None,
        description: str = "synchronized task",
        extra_globals: Dict[str, Any] = None,
    ) -> ParallelExecutionResults:
        """
        동기화 포인트가 포함된 코드를 병렬 실행

        생성된 코드에서 `sync_barrier.wait("phase_name")`을 호출하면
        모든 로봇이 해당 지점에 도달할 때까지 대기합니다.

        Args:
            robot_codes: {robot_id: code_string}
            timeout: 전체 타임아웃 (초)
            description: 작업 설명
            extra_globals: 코드 실행 시 추가할 전역 변수

        Returns:
            ParallelExecutionResults
        """
        # Barrier 리셋
        self.sync_barrier.reset()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[SynchronizedExecution] {description}")
            print(f"  Robots: {self.robot_ids}")
            print(f"  Sync barrier enabled for {len(self.robot_ids)} robots")
            print(f"{'='*60}")

        def execute_for_robot(robot_id: int) -> Any:
            code = robot_codes.get(robot_id, "")
            if not code:
                raise ValueError(f"No code provided for robot {robot_id}")

            # 실행 컨텍스트 구성
            exec_globals = {
                "sync_barrier": self.sync_barrier,
                "robot_id": robot_id,
                "__name__": "__main__",
            }

            # 추가 전역 변수
            if extra_globals:
                exec_globals.update(extra_globals)

            # 코드 실행
            exec(code, exec_globals)

            # execute_task 함수 호출 (있는 경우)
            if "execute_task" in exec_globals:
                return exec_globals["execute_task"]()
            elif "execute_reset_task" in exec_globals:
                return exec_globals["execute_reset_task"]()

            return None

        return self._parallel_manager.execute_with_args(
            func=lambda rid, _: execute_for_robot(rid),
            robot_args={rid: None for rid in self.robot_ids},
            timeout=timeout,
            description=description,
        )

    def get_sync_barrier(self) -> SyncBarrier:
        """현재 sync_barrier 반환 (외부에서 접근 필요시)"""
        return self.sync_barrier


def create_synchronized_manager(
    robot_ids: List[int],
    verbose: bool = True,
) -> SynchronizedExecutionManager:
    """SynchronizedExecutionManager 생성 헬퍼 함수"""
    return SynchronizedExecutionManager(robot_ids=robot_ids, verbose=verbose)
