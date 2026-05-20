"""
BasePipeline — 싱글/멀티암 공통 파이프라인 기반 클래스.

공통 메서드를 여기에 모아서 중복 제거.
서브클래스: ForwardAndResetPipeline (싱글), UnifiedMultiArmPipeline (멀티)
"""

import signal
from typing import Dict, Optional


class BasePipeline:
    """싱글/멀티암 공통 파이프라인 기반 클래스."""

    def _log(self, message: str, step: str = None, tag: str = None) -> str:
        """통일된 로그 포맷 생성.

        Args:
            message: 출력할 메시지
            step: 단계 정보 (예: "Step 1/5")
            tag: 추가 태그 (예: "Validation", "Recording")
        """
        ep_str = f"{self.current_episode:02d}/{self.total_episodes:02d}"
        prefix = f"[{self.current_phase}][{ep_str}]"

        if step:
            prefix += f"[{step}]"
        if tag:
            prefix += f"[{tag}]"

        return f"{prefix} {message}"

    def _start_episode_recording(self, task: str) -> None:
        """에피소드 레코딩 시작."""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.start_episode(task=task)
                print(f"[Recording] Episode started: {task}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to start episode: {e}")

    def _end_episode_recording(self, discard: bool = False):
        """에피소드 레코딩 종료.

        Returns:
            pd.DataFrame | None: discard=False일 때 직전 에피소드의 frame DataFrame.
                                  visualization 등 후처리용. discard=True거나 실패 시 None.
        """
        if not (self.dataset_recorder and self.record_dataset):
            return None
        try:
            # save_episode() 전에 buffer snapshot 캡처 (visualization 용)
            buffer_df = None
            if not discard:
                buffer_df = self._snapshot_episode_buffer()

            info = self.dataset_recorder.end_episode(discard=discard)
            if not discard:
                print(f"[Recording] Episode saved: {info.get('num_frames', 0)} frames")
            else:
                print(f"[Recording] Episode discarded")
            return buffer_df
        except Exception as e:
            print(f"[Recording] Warning: Failed to end episode: {e}")
            return None

    def _snapshot_episode_buffer(self):
        """현재 에피소드 buffer를 pandas DataFrame으로 변환 (save_episode 전 호출).

        신 LeRobot 아키텍처는 save_episode() 후 parquet footer가 안 써져서
        외부에서 다시 읽으면 ArrowInvalid가 발생함. 따라서 buffer가 살아있는
        시점에 in-memory snapshot을 떠두는 것이 가장 안전.
        """
        try:
            import pandas as pd
            ds = self.dataset_recorder._dataset
            # LeRobot v0.5.1: the in-progress episode buffer lives directly on
            # the dataset (`ds.episode_buffer`). Older builds exposed it via
            # `ds.writer.episode_buffer`, but `ds.writer` is now a ParquetWriter
            # with no such attribute — accessing it raised AttributeError and
            # silently dropped every episode's skill visualization.
            buf = getattr(ds, "episode_buffer", None)
            if buf is None:
                return None
            n = buf.get("size", 0)
            if n <= 0:
                return None
            # buf는 dict[key -> list], 모든 list가 길이 n인 컬럼들 + 메타키 (size, episode_index 등)
            cols = {}
            for k, v in buf.items():
                if isinstance(v, list) and len(v) == n:
                    cols[k] = v
            # episode_index 컬럼 보정 (visualize_skills가 필터링에 사용)
            if "episode_index" not in cols and "episode_index" in buf:
                cols["episode_index"] = [buf["episode_index"]] * n
            return pd.DataFrame(cols)
        except Exception as e:
            print(f"[Recording] Buffer snapshot failed: {e}")
            return None

    def _install_recording_signal_handler(self) -> None:
        """Ctrl+C 시 데이터셋 finalize() 호출을 보장하는 signal handler 등록."""
        original_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            print(f"\n[Recording] SIGINT received — finalizing dataset...")
            self._finalize_recording()
            signal.signal(signal.SIGINT, original_handler)
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_sigint)
        print(f"[Recording] Signal handler installed (Ctrl+C will finalize dataset)")

    # ─────────────────────────────────────────────
    # Judge
    # ─────────────────────────────────────────────

    def _run_forward_judge(
        self,
        instruction: str,
        initial_image,
        final_image,
        object_positions: Dict = None,
        executed_code: str = "",
        image_resolution=None,
    ) -> Dict:
        """Forward judge 실행 — 싱글/멀티 공용."""
        if initial_image is None or final_image is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'Initial or final image not captured',
                'success': False,
            }
        try:
            import os
            from judge import TaskJudge

            use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
            judge = TaskJudge(
                model=self.judge_model,
                verbose=self.verbose,
                use_server=use_server,
            )
            return judge.judge(
                instruction=instruction,
                initial_image=initial_image,
                final_image=final_image,
                object_positions=object_positions or {},
                executed_code=executed_code,
                image_resolution=image_resolution,
            )
        except Exception as e:
            print(f"[Judge] Forward judge error: {e}")
            return {"prediction": "UNCERTAIN", "reasoning": str(e)}

    def _run_reset_judge(
        self,
        reset_mode: str,
        current_positions: Dict,
        target_positions: Dict,
        initial_image,
        final_image,
        executed_code: str = "",
        original_instruction: str = None,
        image_resolution=None,
    ) -> Dict:
        """Reset judge 실행 — 싱글/멀티 공용."""
        if initial_image is None or final_image is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'Reset initial or final image not captured',
                'success': False,
                'reset_mode': reset_mode,
            }
        try:
            import os
            from judge import ResetJudge

            use_server = os.getenv("USE_VLM_SERVER", "").lower() in ("1", "true", "yes")
            judge = ResetJudge(
                model=self.judge_model,
                verbose=self.verbose,
                use_server=use_server,
            )
            return judge.judge(
                reset_mode=reset_mode,
                current_positions=current_positions,
                target_positions=target_positions,
                initial_image=initial_image,
                final_image=final_image,
                executed_code=executed_code,
                original_instruction=original_instruction,
                image_resolution=image_resolution,
            )
        except Exception as e:
            print(f"[Judge] Reset judge error: {e}")
            return {"prediction": "UNCERTAIN", "reasoning": str(e), "reset_mode": reset_mode}
