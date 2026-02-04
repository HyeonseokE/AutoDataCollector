"""
Multi-Robot Results Saver

Saves execution results for the multi-robot pipeline including:
- Detection images
- Generated code
- Execution context
- Initial/final state images
- Log files
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MultiRobotResultsSaver:
    """
    멀티 로봇 파이프라인 결과 저장 관리자

    폴더 구조:
        results_multi/<task_name>/episode_XX/robot_2/forward/
            - detection_result.jpg
            - initial_state.jpg
            - generated_code.py
            - execution_context.json
            - final_state.jpg
        results_multi/<task_name>/episode_XX/robot_2/reset/
            - generated_code.py
            - positions.json
            - initial_state.jpg
            - final_state.jpg
    """

    def __init__(
        self,
        base_dir: str = "./pipeline_multi/results_multi",
        task_name: str = None,
        session_timestamp: str = None,
    ):
        """
        Args:
            base_dir: 기본 저장 디렉토리
            task_name: 태스크 이름 (예: "towel_fold")
            session_timestamp: 세션 타임스탬프
        """
        self.base_dir = Path(base_dir)
        self.task_name = task_name or "unnamed_task"
        self.session_timestamp = session_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 세션 디렉토리 생성
        self.session_dir = self.base_dir / f"{self.task_name}_{self.session_timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def get_episode_dir(self, episode: int, robot_id: int, phase: str = "forward") -> Path:
        """에피소드별 로봇별 디렉토리 경로 반환"""
        episode_dir = self.session_dir / f"episode_{episode:02d}" / f"robot_{robot_id}" / phase
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def save_detection_image(
        self,
        episode: int,
        robot_id: int,
        image: np.ndarray,
        phase: str = "forward",
    ) -> str:
        """검출 결과 이미지 저장"""
        if image is None or cv2 is None:
            return ""

        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "detection_result.jpg"
        cv2.imwrite(str(path), image)
        return str(path)

    def save_initial_image(
        self,
        episode: int,
        robot_id: int,
        image: np.ndarray,
        phase: str = "forward",
    ) -> str:
        """초기 상태 이미지 저장"""
        if image is None or cv2 is None:
            return ""

        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "initial_state.jpg"
        cv2.imwrite(str(path), image)
        return str(path)

    def save_final_image(
        self,
        episode: int,
        robot_id: int,
        image: np.ndarray,
        phase: str = "forward",
    ) -> str:
        """최종 상태 이미지 저장"""
        if image is None or cv2 is None:
            return ""

        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "final_state.jpg"
        cv2.imwrite(str(path), image)
        return str(path)

    def save_generated_code(
        self,
        episode: int,
        robot_id: int,
        code: str,
        phase: str = "forward",
    ) -> str:
        """생성된 코드 저장"""
        if not code:
            return ""

        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "generated_code.py"
        path.write_text(code)
        return str(path)

    def save_execution_context(
        self,
        episode: int,
        robot_id: int,
        instruction: str,
        positions: Dict,
        code: str,
        success: bool,
        spec: Dict = None,
        phase: str = "forward",
    ) -> str:
        """실행 컨텍스트 저장"""
        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "execution_context.json"

        # numpy 타입 변환
        def serialize(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        context = {
            "timestamp": datetime.now().isoformat(),
            "robot_id": robot_id,
            "episode": episode,
            "phase": phase,
            "instruction": instruction,
            "object_positions": serialize(positions),
            "generated_spec": serialize(spec) if spec else {},
            "generated_code": code,
            "execution_success": success,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        return str(path)

    def save_positions(
        self,
        episode: int,
        robot_id: int,
        original_positions: Dict,
        current_positions: Dict,
        target_positions: Dict,
        phase: str = "reset",
    ) -> str:
        """리셋 위치 정보 저장"""
        save_dir = self.get_episode_dir(episode, robot_id, phase)
        path = save_dir / "positions.json"

        def serialize(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        positions_data = {
            "original_positions": serialize(original_positions),
            "current_positions": serialize(current_positions),
            "target_positions": serialize(target_positions),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(positions_data, f, indent=2, ensure_ascii=False)

        return str(path)

    def save_log(
        self,
        episode: int,
        robot_id: int,
        log_content: str,
        phase: str = "forward",
    ) -> str:
        """로그 파일 저장"""
        save_dir = self.get_episode_dir(episode, robot_id, phase)
        filename = f"{phase}_log.txt"
        path = save_dir / filename
        path.write_text(log_content)
        return str(path)

    def save_judge_result(
        self,
        episode: int,
        robot_id: int,
        prediction: str,
        reasoning: str,
        initial_image: np.ndarray = None,
        final_image: np.ndarray = None,
        phase: str = "forward",
    ) -> Dict[str, str]:
        """Judge 결과 저장"""
        save_dir = self.get_episode_dir(episode, robot_id, phase)
        saved_files = {}

        # Judge 결과 JSON
        judge_data = {
            "prediction": prediction,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        }
        judge_path = save_dir / "judge_result.json"
        with open(judge_path, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, indent=2, ensure_ascii=False)
        saved_files['judge_result'] = str(judge_path)

        # 이미지 저장
        if cv2 is not None:
            if initial_image is not None:
                init_path = save_dir / "initial_state.jpg"
                cv2.imwrite(str(init_path), initial_image)
                saved_files['initial_image'] = str(init_path)

            if final_image is not None:
                final_path = save_dir / "final_state.jpg"
                cv2.imwrite(str(final_path), final_image)
                saved_files['final_image'] = str(final_path)

        return saved_files

    def save_episode_summary(
        self,
        episode: int,
        robot_results: Dict[int, Dict],
        detection_results: Dict,
        instruction: str,
    ) -> str:
        """에피소드 요약 저장"""
        episode_dir = self.session_dir / f"episode_{episode:02d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        def serialize(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        summary = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "detection_results": serialize(detection_results),
            "robots": {},
        }

        for robot_id, result in robot_results.items():
            summary["robots"][str(robot_id)] = serialize(result)

        path = episode_dir / "episode_summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return str(path)

    def save_session_summary(
        self,
        total_episodes: int,
        completed_episodes: int,
        robot_ids: List[int],
        instruction: str,
        all_results: Dict,
    ) -> str:
        """세션 전체 요약 저장"""
        def serialize(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        summary = {
            "session_timestamp": self.session_timestamp,
            "task_name": self.task_name,
            "instruction": instruction,
            "robot_ids": robot_ids,
            "total_episodes": total_episodes,
            "completed_episodes": completed_episodes,
            "summary": serialize(all_results.get("summary", {})),
        }

        path = self.session_dir / "session_summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return str(path)

    def get_session_dir(self) -> Path:
        """세션 디렉토리 경로 반환"""
        return self.session_dir


def create_task_name_from_instruction(instruction: str) -> str:
    """Instruction에서 태스크 이름 생성"""
    # 간단한 정규화: 소문자, 공백을 _로, 특수문자 제거
    import re
    name = instruction.lower()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    # 길이 제한
    if len(name) > 30:
        name = name[:30]
    return name or "unnamed_task"
