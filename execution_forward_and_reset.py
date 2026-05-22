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


# ============================================================
# Phase-aware episode dir helpers (method3 paradigm — phase1/, phase2/ subdir).
# ============================================================
def _phase_subdir(method3_phase: str | None) -> str:
    """Return ``"phase2"`` if phase=phase2 (chain 이후 reorg 된 session 의 새 episode
    저장 위치). 그 외 (phase1 cycle 또는 미지정) 는 ``""`` (legacy episode_*/ 위치).

    chain 진입 시 phase1_end_prompt 가 기존 episode_* → phase1/ 으로 mv 하므로
    phase1 cycle 의 *추가* 데이터 수집 (resume / more) 은 *legacy 위치* 유지가
    안전. phase2 cycle 시작 시 self.method3_phase=="phase2" → phase2/ subdir.
    """
    return "phase2" if str(method3_phase or "").lower() == "phase2" else ""


def _episode_dir(session_dir: str | Path, episode_num: int, method3_phase: str | None) -> Path:
    """Resolve episode artifact dir based on phase. phase2 → session/phase2/episode_*."""
    sub = _phase_subdir(method3_phase)
    base = Path(session_dir) / sub if sub else Path(session_dir)
    return base / f"episode_{episode_num:02d}"


def _iter_episode_dirs(session_dir: str | Path) -> list[Path]:
    """Return all existing episode_*/ dirs across legacy + phase1/ + phase2/.

    Used by resume mode (incomplete detection) and any analysis that must scan
    *all* historical episodes regardless of reorg state.
    """
    sd = Path(session_dir)
    out: list[Path] = []
    out.extend(sorted(sd.glob("episode_*")))
    out.extend(sorted(sd.glob("phase1/episode_*")))
    out.extend(sorted(sd.glob("phase2/episode_*")))
    return [p for p in out if p.is_dir()]
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


from pipeline.base_pipeline import BasePipeline


class ForwardAndResetPipeline(BasePipeline):
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
        reset_instruction: str = None,
        skip_turn_test: bool = False,
        detect_model: str = None,
        resetspace: str = None,
        recording_config: str = None,
        # Method3 phase mode (final_method3_spec)
        method3_phase: str = "phase1",
        phase1_trained_vla_path: Optional[str] = None,
        phase1_dataset_path: Optional[str] = None,
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
            recording_config: recording config YAML 경로 (None이면 기본 recording_config.yaml)
            method3_phase: "phase1" (subgoal seeding, 기본) | "phase2" (MI selection).
                phase1 → buffer-aware subgoal selector active.
                phase2 → Phase2MISelector + P_phase1 vector DB load (§6/§7).
            phase1_trained_vla_path: Phase2 시 §6 frozen VLA encoder 체크포인트
                경로. 아직 학습 안 됐을 땐 임시로 pretrained 모델 경로를 넣어도 됨.
                phase=phase1 일 땐 무시.
            phase1_dataset_path: Phase2 시 §6 re-embedding 의 raw dataset 경로
                (LeRobot repo_id 또는 local path). 캐시된 P_phase1 이 있으면 무시.
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
        self.reset_instruction = reset_instruction or "move objects to certain position"
        self.skip_turn_test = skip_turn_test
        self.detect_model = detect_model
        self.resetspace = resetspace
        self.recording_config = recording_config
        self.multi_turn_info: Dict = {}
        self.reset_multi_turn_info: Dict = {}

        # Method3 phase mode (final_method3_spec §2)
        if method3_phase not in ("phase1", "phase2"):
            raise ValueError(
                f"method3_phase must be 'phase1' or 'phase2', got {method3_phase!r}")
        self.method3_phase = method3_phase
        self.phase1_trained_vla_path = phase1_trained_vla_path
        self.phase1_dataset_path = phase1_dataset_path
        self._phase2_vector_db = None  # P_phase1 — _setup_phase2_session 이 채움
        self._phase2_selector = None   # Phase2MISelector — _setup_phase2_session 이 채움
        self._phase2_vla_scorer = None  # U_VLA scorer — yaml.vla_informativeness.enabled=true 일 때만

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

        # Reset recording
        self.reset_dataset_recorder = None

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

        # 레코딩 모드 초기화 (resume 모드는 cleanup 후 초기화)
        if self.record_dataset and not self.resume_recording:
            self._init_recording()

    @staticmethod
    def _extract_position_keys(code: str) -> List[str]:
        """캐싱된 코드에서 positions['xxx'] 패턴의 key 추출"""
        import re
        pattern = r'positions\[["\'](.+?)["\']\]'
        return list(dict.fromkeys(re.findall(pattern, code)))

    def _can_reuse_code(self, cached_code: Optional[str], cached_keys: List[str],
                        new_positions: Dict) -> bool:
        """캐싱된 코드를 재사용할 수 있는지 확인 (key + points 일치 검사)

        cached_keys가 비어있으면 코드가 positions를 참조하지 않는 것이므로
        (좌표 하드코딩 등) 무조건 재사용 가능.
        """
        if cached_code is None:
            return False
        if not cached_keys:
            return True
        if not set(cached_keys) <= set(new_positions.keys()):
            return False
        # 코드가 "points"를 참조하면, 해당 object에 points sub-dict가 있어야 함
        if '["points"]' in cached_code or "['points']" in cached_code:
            for key in cached_keys:
                info = new_positions.get(key)
                if isinstance(info, dict) and "points" not in info:
                    return False
        return True

    def _create_skills(self):
        """LeRobotSkills 인스턴스 생성 (exec_globals 주입용).

        LLM 코드가 직접 import/생성하지 않고, 파이프라인이 미리 생성하여 주입.
        """
        from skills.skills_lerobot import LeRobotSkills

        robot_config = f"robot_configs/robot/so101_robot{self.robot_id}.yaml"
        kwargs = {
            "robot_config": robot_config,
            "frame": "base_link",
            "recording_fps": self.recording_fps,
        }
        if self.detect_model:
            kwargs["detect_model"] = self.detect_model

        self._skills = LeRobotSkills(**kwargs)

        # Retro-attach — _skills 가 lazy 생성이라 transport / phase2 setup
        # 시점에는 None 이었을 수 있다. setup 가 이미 끝난 state 면 새 _skills
        # 에 client + rng + selector hook 를 *다시* 부착해야 plan_batch 가 swap.
        _adapter = getattr(self, "_skill_planner_client", None)
        if _adapter is not None:
            _n = int(getattr(self, "_skill_planner_n_candidates", 4))
            try:
                self._skills.set_skill_planner_client(_adapter, n_candidates=_n)
                print(f"[Skill Perturbation] re-attached planner client to fresh _skills (K={_n})")
            except Exception as e:
                print(f"[Skill Perturbation] re-attach planner client failed: {e}")
        _pending = getattr(self, "_pending_perturbation_seed", None)
        if _pending is not None:
            try:
                self._skills.set_perturbation_rng(_pending)
                print(f"[Skill Perturbation] re-attached perturbation RNG to fresh _skills")
            except Exception as e:
                print(f"[Skill Perturbation] re-attach perturbation RNG failed: {e}")
        if getattr(self, "_phase2_selector", None) is not None:
            try:
                self._attach_phase2_skill_hook()
            except Exception as e:
                print(f"[Method3 phase2] re-attach skill hook failed: {e}")
        # Phase2 subgoal replay re-attach — _setup_phase2_session 이 _skills
        # 생성 전에 돌았으면(lazy-init) 여기서 부착된다.
        if getattr(self, "_phase2_subgoal_replay", None) is not None:
            try:
                self._attach_phase2_subgoal_replay()
            except Exception as e:
                print(f"[Method3 phase2] re-attach subgoal replay failed: {e}")

        # 공유 카메라 주입 (detect_objects에서 사용)
        if self.camera:
            self._skills.camera = self.camera
        elif self.camera_manager and self.camera_manager.is_connected:
            # camera_manager에서 RealSense 가져오기
            for cam_name in ["top", "realsense"]:
                try:
                    self._skills.camera = self.camera_manager.get_camera(cam_name)
                    break
                except KeyError:
                    continue

        # Transit-time wrist-camera pitch preference. Independent of
        # perturbation; only biases multi-IK candidate selection on transits.
        self._setup_transit_pitch_pref_on_skills()
        # Subgoal-level perturbation: read recording_config and attach if enabled.
        # Per-episode RNG is seeded later via _seed_episode_perturbation().
        self._setup_perturbation_on_skills()
        # Skill-level perturbation (curobo backend, in-process). Same gating
        # as subgoal — created only when perturbation.skill.enabled_* is true.
        # Skipped entirely when preselective_filter.transport=grpc (server side
        # owns the planner in that case).
        #
        # 2026-05-21: method3_phase='phase1' 이면 skill perturbation 비활성.
        # Phase1 은 *spec 상 cartesian baseline* — curobo plan_batch 호출은 의도가
        # 아니고, client GPU 부담만 가중 (RTX 3050 OOM → 어차피 cartesian fallback).
        # Phase2 에서만 perturbation 활성 (= acquisition 의 candidate generator).
        if getattr(self, "method3_phase", "phase1") == "phase1":
            print("[skill_perturbation] method3_phase=phase1 → skill perturbation SKIPPED "
                  "(Phase1 은 cartesian baseline; Phase2 에서만 활성)")
        else:
            self._setup_skill_perturbation_on_skills()
        # Method 3 (pre-selective acquisition) is wired later, at run time —
        # see _setup_skill_planner_transport(session_dir). It is deferred
        # so the FAISS buffer can live inside the run's session folder.

        return self._skills

    # ------------------------------------------------------------------
    # Phase-flag helpers (enabled_forward / enabled_reset)
    # ------------------------------------------------------------------
    def _read_phase_flags(
        self, section: dict, default_fwd: bool = True, default_reset: bool = False,
    ) -> tuple[bool, bool, bool]:
        """Parse enabled_forward / enabled_reset with legacy `enabled` fallback.

        Returns (any_phase_enabled, enabled_forward, enabled_reset).

        - If either new key is present, use them (defaults applied to the missing one).
        - Else if legacy `enabled` is present, map to (enabled, default_reset).
          This preserves the historical pattern: `enabled: true` → forward-only.
        - Else both False (section disabled).
        """
        if not section:
            return (False, False, False)
        if "enabled_forward" in section or "enabled_reset" in section:
            fwd = bool(section.get("enabled_forward", default_fwd))
            reset = bool(section.get("enabled_reset", default_reset))
        elif "enabled" in section:
            fwd = bool(section["enabled"])
            reset = default_reset
        else:
            return (False, False, False)
        return (fwd or reset, fwd, reset)

    def _phase_gate(self, phase: str):
        """Return a context manager that disables systems whose flag is false for `phase`.

        Reads cached phase flags set during the 3 _setup_* methods. If a system
        wasn't attached, its disable is a no-op (system is None already).
        Falls back to a null context manager when skills haven't been lazily
        created yet (execute_code creates them on first call) — there's
        nothing to gate in that case.
        """
        from contextlib import nullcontext

        skills = getattr(self, "_skills", None)
        if skills is None:
            return nullcontext()
        sg = getattr(self, "_subgoal_phase", {"forward": False, "reset": False})
        sk = getattr(self, "_skill_planner_phase", {"forward": False, "reset": False})
        pf = getattr(self, "_skill_planner_phase_flags", {"forward": False, "reset": False})
        return skills.systems_disabled(
            subgoal=not sg.get(phase, False),
            skill_planner=not sk.get(phase, False),
            preselective=not pf.get(phase, False),
        )

    def _setup_transit_pitch_pref_on_skills(self) -> None:
        """Read ``transit_pitch_max_deg`` from recording_config and apply to skills.

        Top-level yaml key (no nested section). When present and non-null,
        transits prefer IK solutions whose gripper pitch ≤ this value (deg).
        Negative = tilted down (e.g., -30 → at least 30° below horizontal).
        Silent no-op if absent.
        """
        if not self.recording_config:
            return
        cfg_path = Path(self.recording_config)
        if not cfg_path.exists():
            return
        try:
            import yaml as _yaml
            with open(cfg_path, "r") as f:
                full_cfg = _yaml.safe_load(f) or {}
        except Exception:
            return
        val = full_cfg.get("transit_pitch_max_deg")
        if val is None:
            return
        try:
            self._skills.set_transit_pitch_max_deg(float(val))
            print(f"[Transit IK] pitch ≤ {float(val):.1f}° preferred during transits")
        except Exception as e:
            print(f"[Transit IK] failed to apply transit_pitch_max_deg: {e}")

    def _load_phase1_config_file(self) -> dict | None:
        """Load Phase1 settings from ``pipeline_config/phase1_config.yaml``.

        Method3 Phase1 의 전용 config 파일. 존재하면 그 dict 를 반환하여
        recording_config 의 ``perturbation.subgoal`` 보다 우선 적용한다.
        없으면 None → recording_config 로 폴백.
        """
        path = Path(__file__).resolve().parent / "pipeline_config" / "phase1_config.yaml"
        if not path.exists():
            return None
        try:
            from method3.config import load_phase1_config
            cfg = load_phase1_config(path)
            print(f"[Perturbation] Phase1 config ← {path}")
            return cfg
        except Exception as e:
            print(f"[Perturbation] Failed to read phase1_config.yaml: {e}")
            return None

    def _setup_phase1_readiness_hook(self) -> None:
        """Build the per-episode Phase1 readiness hook.

        Read precedence (mirror of ``_setup_perturbation_on_skills``):

          1. ``pipeline_config/phase1_config.yaml`` → ``readiness:`` section
             (preferred — Phase1 settings live in the Phase1-dedicated file)
          2. ``recording_config_ws*.yaml`` → ``method3_phase1:`` section
             (fallback for ergonomics; emits a deprecation hint)

        Disabled (no-op hook) when neither file has the section or
        ``enabled: false``. Idempotent — re-reads each call so edits between
        resume retries take effect on the next run.
        """
        from method3_integration import Phase1ReadinessHook, Phase1ReadinessHookConfig
        # default-disabled hook so the loop call site can unconditionally check
        # `.enabled` without None-guards.
        self._phase1_readiness_hook = Phase1ReadinessHook(Phase1ReadinessHookConfig())

        # ── 1) phase1_config.yaml.readiness — preferred source.
        section: dict | None = None
        source: str | None = None
        ph1_path = Path(__file__).resolve().parent / "pipeline_config" / "phase1_config.yaml"
        if ph1_path.exists():
            try:
                import yaml as _yaml
                with open(ph1_path, "r") as f:
                    ph1_cfg = _yaml.safe_load(f) or {}
                if ph1_cfg.get("readiness") is not None:
                    section = ph1_cfg["readiness"]
                    source = str(ph1_path.name)
            except Exception as e:
                print(f"[method3:phase1] phase1_config.yaml read failed: {e}")

        # ── 2) recording_config_ws*.yaml.method3_phase1 — fallback (legacy).
        if section is None and self.recording_config:
            rc_path = Path(self.recording_config)
            if rc_path.exists():
                try:
                    import yaml as _yaml
                    with open(rc_path, "r") as f:
                        rc_cfg = _yaml.safe_load(f) or {}
                    if rc_cfg.get("method3_phase1") is not None:
                        section = rc_cfg["method3_phase1"]
                        source = f"{rc_path.name} (deprecated location — move to phase1_config.yaml readiness:)"
                except Exception as e:
                    print(f"[method3:phase1] recording_config read failed: {e}")

        if section is None:
            return

        hook_cfg = Phase1ReadinessHookConfig.from_yaml_section(section)
        self._phase1_readiness_hook = Phase1ReadinessHook(hook_cfg)
        # readiness trajectory jsonl — session_dir 안 (사용자 요청, 분석용).
        # session_dir 가 아직 없을 수 있어 setup 시점엔 path 만 보관 후, 첫 check
        # 직전 _finalize_subgoal_buffer 단계에서 set_readiness_trace 호출.
        if hook_cfg.enabled:
            print(
                f"[method3:phase1] readiness hook ENABLED ← {source} | "
                f"B₁_min={hook_cfg.phase1_min} B₁_max={hook_cfg.phase1_max} "
                f"state_ready={hook_cfg.state_ready_threshold} "
                f"phase2_gain={hook_cfg.phase2_potential_gain_threshold} "
                f"auto_stop={hook_cfg.auto_stop_on_ready}"
            )

    def _setup_perturbation_on_skills(self) -> None:
        """Wire the Phase1 subgoal selector from phase1_config.yaml to skills.

        Phase1 설정은 ``pipeline_config/phase1_config.yaml`` 에서 읽는다 (없으면
        recording_config 의 ``perturbation.subgoal`` 로 폴백). recording_config
        가 없거나 perturbation 이 비활성이면 silent no-op.

        ``method3_phase == "phase2"`` 면 Phase1 subgoal selector 설치를 건너뛴다
        (Phase2 의 candidate generator 와 동시에 돌면 충돌). Phase2 자체 설정은
        session_dir 가 확정된 뒤 ``_setup_phase2_session`` 에서 수행.
        """
        self._subgoal_phase = {"forward": False, "reset": False}
        if getattr(self, "method3_phase", "phase1") == "phase2":
            # Phase1 subgoal 섭동 selector 는 설치하지 않는다. 단 Phase2
            # subgoal replay selector 는 _setup_phase2_session 이
            # set_subgoal_selector 로 이미 붙여 놓았다 — forward episode 의
            # systems_disabled(subgoal = not _subgoal_phase[phase]) 가 그
            # selector 를 None 으로 끄지 않도록 _subgoal_phase["forward"]=True
            # 로 둔다. reset episode 는 Phase1 forward-task buffer 와 무관
            # 하므로 끈 채로 둔다.
            _has_replay = getattr(self, "_phase2_subgoal_replay", None) is not None
            self._subgoal_phase = {"forward": _has_replay, "reset": False}
            print("[Method3] phase=phase2 → Phase1 subgoal selector skipped; "
                  f"subgoal replay forward={'ON' if _has_replay else 'OFF'} "
                  "(_setup_phase2_session 에서 설치)")
            return
        if not self.recording_config:
            return
        cfg_path = Path(self.recording_config)
        if not cfg_path.exists():
            return
        try:
            import yaml as _yaml
            with open(cfg_path, "r") as f:
                full_cfg = _yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Perturbation] Failed to read recording_config: {e}")
            return
        pert_raw = (full_cfg.get("perturbation") or {}).get("subgoal") or {}
        # Phase1 전용 config 파일이 있으면 우선 적용 (recording_config 보다 우선).
        _phase1_file = self._load_phase1_config_file()
        if _phase1_file is not None:
            pert_raw = _phase1_file
        any_en, fwd, reset = self._read_phase_flags(pert_raw)
        if not any_en:
            return
        self._subgoal_phase = {"forward": fwd, "reset": reset}

        # Phase1 buffer-aware subgoal scoring is now the sole path —
        # `mode` yaml key is no longer read (was always `buffer_aware`).
        # Legacy SubgoalPerturbation (gaussian blob) is dead path; if needed
        # it can be restored from git history.
        self._setup_buffer_aware_subgoal(pert_raw, fwd, reset)

    def _setup_buffer_aware_subgoal(self, pert_raw: dict, fwd: bool, reset: bool) -> None:
        """Wire the buffer-aware Phase1 subgoal selector (문서 phase1_subgoal_scoring_core).

        Replaces the legacy SubgoalPerturbation single random offset with
        K-candidate buffer-aware scoring. The per-skill state buffer is
        persisted to a single ``.npz`` file (``buffer_file``); the actual file
        path is bound later in ``_finalize_subgoal_buffer(session_dir)`` because
        the session directory does not exist yet at skills-creation time.
        """
        try:
            from method3.phase1_state_seeding import (
                Phase1SubgoalConfig, Phase1SubgoalSelector, ReachabilityConfig,
                SubgoalBuffer,
            )
            # subgoal 위치 제약 (문서 §1 reachable/safe) — 각 항목 생략/null 이면
            # 해당 검사 비활성. reach annulus 는 move_to_position 이 kinematics
            # 술어를 넘겨 항상 적용된다.
            reach_raw = pert_raw.get("reachability") or {}

            def _opt_float(v):
                return None if v is None else float(v)

            def _opt_bounds(v):
                return tuple(float(x) for x in v) if v else None

            reachability = ReachabilityConfig(
                z_min=_opt_float(reach_raw.get("z_min")),
                z_max=_opt_float(reach_raw.get("z_max")),
                x_bounds=_opt_bounds(reach_raw.get("x_bounds")),
                y_bounds=_opt_bounds(reach_raw.get("y_bounds")),
            )
            # §4.2 hemisphere dist 의 방향 기준점 — robot base 가 좌표 원점이라
            # (0,0,0) 으로 고정. yaml 노출 제거 (이전 robot_origin 키는 무시됨).
            robot_origin = (0.0, 0.0, 0.0)
            # per_skill: skill_id 별 부분 override. 지원 키 sigma/clip_factor/
            # n_candidates. 비어 있거나 항목 누락이면 전역 값 그대로.
            _per_skill_raw = pert_raw.get("per_skill") or {}
            _int_keys = {"n_candidates"}
            per_skill_overrides: dict = {}
            for _sk, _ov in _per_skill_raw.items():
                if not _ov:
                    continue
                per_skill_overrides[str(_sk)] = {
                    str(_k): (int(_v) if _k in _int_keys else float(_v))
                    for _k, _v in _ov.items()
                }
            cfg = Phase1SubgoalConfig(
                sigma=float(pert_raw.get("sigma", 0.05)),
                clip_factor=float(pert_raw.get("clip_factor", 2.0)),
                n_candidates=int(pert_raw.get("n_candidates", 32)),
                k_nn=int(pert_raw.get("k_nn", 5)),
                end_fraction=float(pert_raw.get("end_fraction", 0.2)),
                preview_points=int(pert_raw.get("preview_points", 20)),
                eps=float(pert_raw.get("eps", 1e-6)),
                s_min=float(pert_raw.get("s_min", 0.01)),
                min_buffer_size=int(pert_raw.get("min_buffer_size", 8)),
                candidate_dist=str(pert_raw.get("candidate_dist", "uniform_ball")),
                robot_origin=robot_origin,
                # max_resample: rejection 재샘플 상한. 200 이면 충분
                # (truncated Gaussian 에서 ~200 회 안에 거의 무조건 valid 샘플 확보).
                # yaml 노출 제거 (이전 max_resample 키는 무시됨).
                max_resample=200,
                reachability=reachability,
                per_skill_overrides=per_skill_overrides,
                debug_verbose=bool(pert_raw.get("debug_verbose", False)),
            )
            # buffer_file: 단일 .npz 파일명/경로. 상대경로면 실행 세션
            # 디렉터리 기준으로 _finalize_subgoal_buffer 에서 해석된다.
            # None → 메모리 전용 (이 run 동안만 grow).
            self._subgoal_buffer_file = pert_raw.get("buffer_file")
            buffer = SubgoalBuffer()  # file 은 finalize 에서 바인딩
            selector = Phase1SubgoalSelector(buffer, cfg)
            self._skills.set_subgoal_selector(selector)
            # Kept on self so finalize/teardown can reach the buffer.
            self._subgoal_selector = selector
            _dist_desc = cfg.candidate_dist
            if cfg.candidate_dist == "hemisphere":
                _dist_desc += f"(origin={cfg.robot_origin})"
            print(
                f"[Perturbation] Buffer-aware Phase1 subgoal scoring ENABLED "
                f"(K={cfg.n_candidates}, dist={_dist_desc}, k_nn={cfg.k_nn}, "
                f"min_buffer={cfg.min_buffer_size}, "
                f"buffer_file={self._subgoal_buffer_file or 'memory-only'}, "
                f"debug_verbose={cfg.debug_verbose}, "
                f"phase=forward:{fwd} reset:{reset})"
            )
            pending = getattr(self, "_pending_perturbation_seed", None)
            if pending is not None:
                self._skills.set_perturbation_rng(pending)
        except Exception as e:
            print(f"[Perturbation] Failed to construct buffer-aware selector: {e}")

    def _finalize_subgoal_buffer(self, session_dir: str | None) -> None:
        """Bind the buffer-aware subgoal buffer file to the run's session dir.

        Called once the session directory exists (alongside
        ``_setup_skill_planner_transport``). A relative ``buffer_file``
        resolves to ``<session_dir>/<buffer_file>`` so the buffer lives inside
        the run's session folder; an absolute path is used as-is. Then the
        file is loaded (preload — picks up an existing buffer on session
        resume). No-op for gaussian mode or when no ``buffer_file`` was given.
        """
        selector = getattr(self, "_subgoal_selector", None)
        raw = getattr(self, "_subgoal_buffer_file", None)
        if selector is None or not raw:
            return
        try:
            path = Path(raw)
            if not path.is_absolute():
                base = Path(session_dir) if session_dir else Path(".")
                path = base / path
            selector.buffer.set_file(path)
            selector.buffer.load()
            # subgoal selection trace (jsonl) — 분석용. 사용자 요청.
            # session_dir/subgoal_phase1_trace.jsonl 에 각 select_subgoal 호출의
            # candidate scores + 선정 정보 append.
            if session_dir and hasattr(selector, "set_trace_file"):
                trace_path = Path(session_dir) / "subgoal_phase1_trace.jsonl"
                selector.set_trace_file(trace_path)
                print(f"[Perturbation] subgoal trace → {trace_path}")
            # readiness trajectory trace — phase1 readiness hook 의 매 check
            # 결과를 session_dir/readiness_trajectory.jsonl 에 append.
            _hook = getattr(self, "_phase1_readiness_hook", None)
            if session_dir and _hook is not None and hasattr(_hook, "set_readiness_trace"):
                r_path = Path(session_dir) / "readiness_trajectory.jsonl"
                _hook.set_readiness_trace(r_path)
                print(f"[method3:phase1] readiness trace → {r_path}")
            # Buffer 카운트 로드 — GREEN (사용자 요청: buffer 변화 = 초록색).
            # ANSI 를 inline 으로 직접 — 이 함수 스코프에 색 변수 미정의 → NameError 위험 회피.
            print(
                f"\033[92m[Perturbation] subgoal buffer file → {selector.buffer.file_path()} "
                f"(preloaded {selector.buffer.total_size()} entries over "
                f"{len(selector.buffer.skill_ids())} skills)\033[0m"
            )
            # Layout-consistency check — buffer 의 episode_id 가 현재 폴더
            # layout 과 어긋나면 stale entry 가 reconcile 에서 통째로 drop 된다.
            # 확장 migration 이 폴더만 옮기고 buffer 를 안 옮긴 케이스 (이번 RCA
            # 의 사고 패턴) 를 시작 시점에 경고한다. 워크플로 호환성 유지를 위해
            # 일단 WARN 으로만 노출 — 강제 abort 는 별도 flag 로 옵트인.
            if session_dir:
                # phase1/ phase2/ legacy 모두 — chain reorg 후 episode 위치 인식.
                _folder_ids = {p.name for p in _iter_episode_dirs(session_dir)}
                _buf_ids = {
                    str(e.episode_id)
                    for entries in selector.buffer._skills.values()
                    for e in entries
                    if e.episode_id
                }
                _orphan = _buf_ids - _folder_ids
                if _orphan:
                    # GREEN — buffer 상태 (orphan entry 검출) 도 buffer 카테고리.
                    print(
                        f"\033[92m[Perturbation] WARN — buffer has {len(_orphan)} episode_id(s) "
                        f"with no matching folder (stale layout?): "
                        f"{sorted(_orphan)[:5]}{'...' if len(_orphan)>5 else ''}\033[0m"
                    )
                    print(
                        f"\033[92m[Perturbation] WARN — likely cause: extension migration ran "
                        f"on folders but not on subgoal_buffer.npz. Run "
                        f"`scripts/migrate_session_episodes_per_seed.py --apply` (now "
                        f"handles buffer rewrite) to fix, or expect reconcile to drop "
                        f"these entries on the next step.\033[0m"
                    )

            # resume reconcile — 삭제된 에피소드의 stale entry 를 정리한다.
            # cleanup_dataset_for_resume 가 계산한 생존(judge=TRUE·폴더 존재)
            # 에피소드 집합으로 buffer 를 맞춰 데이터셋과 1:1 정합을 유지한다.
            # fresh run 은 _resume_kept_true_episodes 가 없음 → 스킵.
            _kept = getattr(self, "_resume_kept_true_episodes", None)
            if _kept is not None:
                from method3.episode_lifecycle import EpisodeReconciler, episode_id
                _keep_ids = {episode_id(n) for n in _kept}

                # ── Pre-reconcile breakdown ─────────────────────────────────
                # buffer 안에 어떤 episode 의 entry 가 몇 개씩 있는지 미리
                # 집계해서 drop 대상을 episode-by-episode 로 출력한다.
                # _skills 는 dict[skill_id → list[Entry]]; entry.episode_id 로
                # 그룹화. tagged("") entry 는 retain_episodes 의 안전장치로
                # 보존되므로 drop 대상에서 제외한다.
                _buf = selector.buffer
                _before_total = _buf.total_size()
                _to_drop: dict[str, dict[str, int]] = {}
                _untagged_kept = 0
                for _sid, _entries in _buf._skills.items():
                    for _e in _entries:
                        _eid = str(getattr(_e, "episode_id", ""))
                        if not _eid:
                            _untagged_kept += 1
                            continue
                        if _eid not in _keep_ids:
                            _to_drop.setdefault(_eid, {})
                            _to_drop[_eid][_sid] = _to_drop[_eid].get(_sid, 0) + 1

                # cleanup 로그 그룹과 시각적으로 묶기 위해 GREEN 으로 출력.
                _GREEN = "\033[92m"
                _RESET = "\033[0m"
                if _to_drop:
                    _n_eps = len(_to_drop)
                    _total = sum(sum(d.values()) for d in _to_drop.values())
                    print(
                        f"{_GREEN}[Perturbation] resume reconcile — {_n_eps} episode(s) "
                        f"to drop from subgoal_buffer ({_total} entries total):{_RESET}"
                    )
                    for _eid in sorted(_to_drop):
                        _per_ep = sum(_to_drop[_eid].values())
                        _bd = ", ".join(
                            f"{sk}={n}" for sk, n in sorted(_to_drop[_eid].items())
                        )
                        print(f"{_GREEN}    - {_eid}: {_per_ep} entries ({_bd}){_RESET}")
                else:
                    print(
                        f"{_GREEN}[Perturbation] resume reconcile — no stale subgoal_buffer "
                        f"entries to drop (kept {len(_keep_ids)} TRUE episodes){_RESET}"
                    )

                _result = EpisodeReconciler([_buf]).retain_episodes(_keep_ids)
                _dropped = sum(_result.values())
                print(
                    f"{_GREEN}[Perturbation] resume reconcile DONE — kept {len(_keep_ids)} "
                    f"TRUE episodes, dropped {_dropped} stale buffer entries "
                    f"(buffer {_before_total} → {_buf.total_size()}; "
                    f"{_untagged_kept} untagged preserved){_RESET}"
                )
        except Exception as e:
            print(f"[Perturbation] subgoal buffer finalize skipped: {e}")

    def _reconcile_subgoal_buffer_on_resume(self, session_dir: str | None) -> None:
        """Resume 시 subgoal_buffer.npz 의 stale entry 를 정리한다 (selector-free).

        `_finalize_subgoal_buffer` 의 reconcile 블록과 동일한 로직이지만,
        `_subgoal_selector` 가 lazy init 라 cleanup 시점엔 아직 None 인 문제를
        우회한다. SubgoalBuffer 인스턴스를 직접 만들어 .npz file 을 load 하고
        keep_set 외 entry 를 제거 + save.

        cleanup_dataset_for_resume 가 set 한 `_resume_kept_true_episodes` 를
        keep_set 으로 쓴다. fresh run 또는 buffer file/keep_set 미설정이면 no-op.

        호출 후 첫 episode 시점에 lazy selector init → buffer.load() 가 같은 파일을
        다시 로드 → 이미 깨끗하므로 finalize 안의 두 번째 reconcile 은 0 drop 만
        보고함. 그래서 호출 후 `_resume_kept_true_episodes = None` 으로 비워
        finalize 의 reconcile 블록이 silent skip 되게 한다 (중복 로그 방지).
        """
        # Phase2 는 Phase1 subgoal buffer 를 read-only 로 replay 한다. reconcile
        # (stale episode drop) 은 Phase1 누적 정합용 — Phase2 resume 에서 돌면
        # phase2/ 가 비어 kept_true_episodes=∅ 이 되고, Phase1 도달 subgoal 기록을
        # 전부 stale 로 오판해 retain_episodes 가 날린다 (replay 데이터 소스 파괴).
        if str(getattr(self, "method3_phase", "phase1")).lower() == "phase2":
            print("[Perturbation] resume reconcile SKIPPED — method3_phase=phase2 "
                  "(Phase1 subgoal buffer 는 replay 전용으로 보존)")
            return
        _kept = getattr(self, "_resume_kept_true_episodes", None)
        if _kept is None:
            return
        buffer_file = getattr(self, "_subgoal_buffer_file", None)
        if not buffer_file:
            # _subgoal_selector 가 lazy init (첫 episode robot init 시점) 라
            # cleanup 직후 시점엔 _subgoal_buffer_file 가 아직 None. yaml 을
            # 직접 읽어 buffer_file 경로만 fallback 으로 가져온다 → selector
            # 인스턴스 없이도 transient reconcile 가능 → cleanup 로그와 동기.
            try:
                from method3.config import load_phase1_config
                yaml_path = (
                    Path(__file__).resolve().parent
                    / "pipeline_config" / "phase1_config.yaml"
                )
                if yaml_path.exists():
                    cfg = load_phase1_config(yaml_path)
                    buffer_file = cfg.get("buffer_file")
            except Exception:
                pass
        if not buffer_file:
            return
        try:
            from method3.phase1_state_seeding import SubgoalBuffer
            from method3.episode_lifecycle import episode_id
            bf_path = Path(buffer_file)
            if not bf_path.is_absolute():
                base = Path(session_dir) if session_dir else Path(".")
                bf_path = base / bf_path
            _GREEN = "\033[92m"
            _YELLOW = "\033[93m"
            _RESET = "\033[0m"
            if not bf_path.exists():
                print(
                    f"{_YELLOW}[Perturbation] resume reconcile — "
                    f"no buffer file at {bf_path} (nothing to reconcile){_RESET}"
                )
                self._resume_kept_true_episodes = None
                return

            buf = SubgoalBuffer()
            buf.set_file(bf_path)
            buf.load()
            _before_total = buf.total_size()
            _keep_ids = {episode_id(n) for n in _kept}

            # Pre-reconcile breakdown
            _to_drop: dict[str, dict[str, int]] = {}
            _untagged_kept = 0
            for _sid, _entries in buf._skills.items():
                for _e in _entries:
                    _eid = str(getattr(_e, "episode_id", ""))
                    if not _eid:
                        _untagged_kept += 1
                        continue
                    if _eid not in _keep_ids:
                        _to_drop.setdefault(_eid, {})
                        _to_drop[_eid][_sid] = _to_drop[_eid].get(_sid, 0) + 1

            if _to_drop:
                _n_eps = len(_to_drop)
                _total = sum(sum(d.values()) for d in _to_drop.values())
                print(
                    f"{_GREEN}[Perturbation] resume reconcile — {_n_eps} episode(s) "
                    f"to drop from subgoal_buffer ({_total} entries total):{_RESET}"
                )
                for _eid in sorted(_to_drop):
                    _per_ep = sum(_to_drop[_eid].values())
                    _bd = ", ".join(
                        f"{sk}={n}" for sk, n in sorted(_to_drop[_eid].items())
                    )
                    print(f"{_GREEN}    - {_eid}: {_per_ep} entries ({_bd}){_RESET}")
            else:
                print(
                    f"{_GREEN}[Perturbation] resume reconcile — no stale subgoal_buffer "
                    f"entries to drop ({_before_total} entries; kept {len(_keep_ids)} "
                    f"TRUE episodes){_RESET}"
                )

            _removed = buf.retain_episodes(_keep_ids)  # auto save()
            print(
                f"{_GREEN}[Perturbation] resume reconcile DONE — kept {len(_keep_ids)} "
                f"TRUE episodes, dropped {_removed} stale buffer entries "
                f"(buffer {_before_total} → {buf.total_size()}; "
                f"{_untagged_kept} untagged preserved){_RESET}"
            )

            # 중복 호출 방지 — finalize 의 reconcile 블록이 None 보고 skip.
            self._resume_kept_true_episodes = None
        except Exception as e:
            print(f"[Perturbation] resume reconcile skipped: {e}")
            import traceback; traceback.print_exc()

    def _load_phase2_runtime_paths(self) -> tuple[str | None, str | None]:
        """phase2_config.yaml 에서 ``phase1_trained_vla_path`` / ``phase1_dataset_path``
        를 읽어 (vla_path, dataset_path) 로 반환. CLI 인자가 비어 있을 때 fallback.

        두 값 모두 비어 있는 키는 None 으로 정규화. yaml 부재·읽기 실패 시 (None, None).
        """
        cfg_path = Path(__file__).resolve().parent / "pipeline_config" / "phase2_config.yaml"
        if not cfg_path.exists():
            return None, None
        try:
            import yaml as _yaml
            with open(cfg_path, "r") as f:
                cfg = _yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Method3 phase2] phase2_config.yaml read failed: {e}")
            return None, None
        vla = cfg.get("phase1_trained_vla_path") or None
        ds = cfg.get("phase1_dataset_path") or None
        return (str(vla) if vla else None, str(ds) if ds else None)

    @staticmethod
    def _resolve_phase1_path(path: str | None) -> str | None:
        """local path 면 그대로, HF repo_id ("user/name") 면 그대로 (lerobot 가 다운로드).

        - 절대경로면서 존재: 그대로 반환 (local 사용).
        - 그 외: 문자열 그대로 반환 — LeRobotDataset / make_vla_key_extractor 가
          HF Hub 에서 자동 다운로드한다.
        - None / 빈 문자열: None 반환.
        """
        if not path:
            return None
        p = Path(str(path)).expanduser()
        if p.is_absolute() and p.exists():
            return str(p)
        return str(path)

    def _setup_phase2_session(self, session_dir: str | None) -> None:
        """Phase2 — §6 P_phase1 vector DB load/build + (future) MI selector wiring.

        session_dir 가 확정된 시점에 호출된다 (run_multiple_episodes / resume).
        ``method3_phase != "phase2"`` 면 silent no-op.

        1. ``<session_dir>/skill_wise_vector_db.npz`` 캐시 hit → 그대로 로드.
        2. miss → ``phase1_trained_vla_path`` 로 frozen encoder 로드 +
           ``phase1_dataset_path`` LeRobot dataset 으로 §6 Step 3-4 (re-embedding)
           수행 → SkillVectorDB 생성·저장.
        3. 결과 vector DB 를 ``self._phase2_vector_db`` 에 보관 — Phase2MISelector /
           candidate generator wiring (Task C) 가 이를 reference buffer 로 쓴다.
        """
        _phase = getattr(self, "method3_phase", "phase1")
        if _phase != "phase2":
            print(f"[Method3 phase2] method3_phase={_phase} → SKIPPED "
                  f"(P_phase1 build/load + Phase2MISelector 는 Phase2 전용)")
            return
        if not session_dir:
            print("[Method3 phase2] session_dir is None — vector DB load skipped")
            return
        # 2026-05-21 architecture (옵션 C):
        #   client 가 *local* (RTX 3050) 에서 build → 결과 npz 를 server cache
        #   path 로 *scp 자동 업로드* → server 가 다음 부팅 시 그 cache load.
        # server-side 는 yaml.server_build_phase1: false 로 비활성 (= load only).
        # client 가 P_phase1 ownership 보유, server 는 *kNN reference + accumulate*.
        try:
            from method3.reembedding.build_or_load import (
                build_or_load_phase1_vector_db,
            )
            # CLI 우선, 미지정 시 phase2_config.yaml 에서 fallback.
            yaml_vla, yaml_ds = self._load_phase2_runtime_paths()
            vla_path = self._resolve_phase1_path(self.phase1_trained_vla_path or yaml_vla)
            ds_path = self._resolve_phase1_path(self.phase1_dataset_path or yaml_ds)
            if vla_path:
                print(f"[Method3 phase2] phase1_trained_vla_path ← {vla_path}")
            if ds_path:
                print(f"[Method3 phase2] phase1_dataset_path     ← {ds_path}")
            self._phase2_vector_db = build_or_load_phase1_vector_db(
                session_dir,
                phase1_trained_vla_path=vla_path,
                phase1_dataset_path=ds_path,
            )
            print(
                f"[Method3 phase2] P_phase1 ready — "
                f"{self._phase2_vector_db.total_size()} entries over "
                f"{len(self._phase2_vector_db.skill_ids())} skills "
                f"(skills={self._phase2_vector_db.skill_ids()})"
            )
            # grpc mode + build 성공 시 → 결과 npz 를 *server cache path 로 자동 scp*.
            # server 가 다음 부팅 시 그 cache load. yaml.remote 의 ssh 정보 사용.
            if getattr(self, "_skill_planner_grpc_client", None) is not None:
                self._upload_p_phase1_to_server(session_dir)
        except Exception as e:
            print(f"[Method3 phase2] P_phase1 load/build FAILED: {e}")
            import traceback; traceback.print_exc()
            self._phase2_vector_db = None

        # P_phase1 ready 검증 — local mode. None 또는 empty 면 *silent 진행 금지*.
        # spec §14 의 useful-OOD selection 은 P_phase1 baseline 필수.
        if self._phase2_vector_db is None or self._phase2_vector_db.total_size() == 0:
            _RED = "\033[91m"; _BOLD = "\033[1m"; _RST = "\033[0m"
            _state = "None (build/load FAILED)" if self._phase2_vector_db is None \
                     else f"empty ({self._phase2_vector_db.total_size()} entries)"
            msg = (
                f"\n{_RED}{_BOLD}[Method3 phase2] P_phase1 vector DB is {_state}{_RST}\n"
                f"{_RED}    Phase2 useful-OOD selection 은 P_phase1 baseline 없이 의미가 없습니다.{_RST}\n"
                f"{_RED}    → 다음을 확인:{_RST}\n"
                f"{_RED}      1) phase2_config.yaml.phase1_dataset_path 가 유효한 LeRobot dataset 인지   {_RST}\n"
                f"{_RED}      2) phase2_config.yaml.phase1_trained_vla_path 가 유효한 VLA checkpoint 인지{_RST}\n"
                f"{_RED}      3) session_dir 의 cache 가 부패했는지:                                     {_RST}\n"
                f"{_RED}         rm -f {session_dir}/skill_wise_vector_db.npz   # 후 재시도         {_RST}\n"
                f"{_RED}{_BOLD}    → session 종료 (P_phase1 없이 acquisition 진행 금지).{_RST}\n"
            )
            print(msg)
            raise RuntimeError(
                f"P_phase1 vector DB unavailable ({_state}) — "
                f"check phase1_dataset_path / phase1_trained_vla_path / cache"
            )

        # Phase2MISelector + (옵셔널) U_VLA scorer wiring — useful_ood_updated §11-13.
        # candidate generator 가 후보 batch 를 만들면 `_phase2_select(...)` 가
        # M_MI + Useful-OOD rule 로 ξ* 를 고른다.
        self._phase2_selector = self._build_phase2_selector()
        self._phase2_vla_scorer = self._build_phase2_vla_scorer(vla_path)

        # G_seed anchor buffer (Phase1 누적 subgoal) + skill candidate hook 등록.
        # skills_lerobot 의 plan_batch hook 자리에 Useful-OOD selector 를 꽂아
        # K candidates 중 ξ* 를 골라 그 index 를 반환한다 — accepted 면 즉시
        # vector DB 에 flush 까지 한다 (§14).
        self._phase2_g_seed_buffer = self._load_phase2_g_seed_buffer(session_dir)
        self._attach_phase2_skill_hook()

        # 방법론: Phase2 는 Phase1 이 도달했던 subgoal 을 그대로 경유하고 경로
        # (trajectory) 만 curobo plan_batch 로 다양화한다. session 의
        # subgoal_buffer.npz 를 읽어 episode 별 replay selector 를 설치한다.
        self._setup_phase2_subgoal_replay(session_dir)

    def _setup_phase2_subgoal_replay(self, session_dir: str | None) -> None:
        """Phase2 — Phase1 도달 subgoal 을 episode 별로 replay 하도록 설치.

        Phase1 은 buffer-aware 섭동으로 subgoal 을 *옮겨* 상태 다양성을 키웠고,
        그 도달 위치를 ``<session>/subgoal_buffer.npz`` 에 episode 별로 기록했다.
        Phase2 는 그 기록된 경유점을 그대로 통과해야 하므로(경로만 다양화),
        ``Phase2SubgoalReplay`` 를 ``set_subgoal_selector`` 훅에 꽂는다 — Phase1
        subgoal selector 가 들어갈 자리에 replay 가 대신 들어가, move_to_position
        의 subgoal 분기가 섭동 대신 기록값을 반환한다.

        ``_skills`` 는 lazy 생성이라 본 메서드(``_setup_phase2_session``) 시점엔
        아직 없을 수 있다. 따라서 replay 객체 *생성* 과 ``_skills`` *부착* 을
        분리한다 — 부착은 ``_attach_phase2_subgoal_replay`` 가 담당하고,
        ``_create_skills`` 의 retro-attach 가 _skills 생성 직후 다시 호출한다.

        buffer 파일이 없으면(예: fresh phase2) replay 를 비활성화하고 raw 검출
        nominal 을 그대로 쓴다 — 단, 그 경우 방법론에서 벗어남을 경고한다.
        """
        self._phase2_subgoal_replay = None
        if not session_dir:
            return
        buf_path = Path(session_dir) / "subgoal_buffer.npz"
        if not buf_path.exists():
            _Y = "\033[93m"; _R = "\033[0m"
            print(f"{_Y}[Method3 phase2] subgoal_buffer.npz 없음 ({buf_path}) — "
                  f"subgoal replay 비활성. Phase2 가 raw 검출 nominal 을 쓰게 되어 "
                  f"Phase1 도달 경유점과 어긋날 수 있음.{_R}")
            return
        try:
            from method3.phase2_mi_selection.subgoal_replay import Phase2SubgoalReplay
            replay = Phase2SubgoalReplay(buf_path)
        except Exception as e:
            print(f"[Method3 phase2] subgoal replay 생성 실패: {e}")
            import traceback; traceback.print_exc()
            return
        self._phase2_subgoal_replay = replay
        print(f"[Method3 phase2] subgoal replay READY — "
              f"{replay.n_episodes()} episodes 기록, skills={replay.skill_ids()}")
        # _skills 가 이미 있으면 즉시 부착; 아직 lazy-init 전이면 _create_skills
        # 의 retro-attach 가 부착한다.
        self._attach_phase2_subgoal_replay()

    def _attach_phase2_subgoal_replay(self) -> None:
        """``_phase2_subgoal_replay`` 를 ``_skills`` 의 subgoal selector 훅에 부착.

        ``_skills`` 가 아직 없으면 silent no-op (retro-attach 가 나중에 호출).
        ``set_subgoal_selector`` 는 단순 재할당이라 중복 호출돼도 idempotent.
        replay 분기도 ``move_to_position`` 의 ``_perturbation_rng`` 가 필요하므로
        pending seed 가 있으면 함께 시딩한다.
        """
        replay = getattr(self, "_phase2_subgoal_replay", None)
        skills = getattr(self, "_skills", None)
        if replay is None or skills is None:
            return
        try:
            skills.set_subgoal_selector(replay)
            pending = getattr(self, "_pending_perturbation_seed", None)
            if pending is not None:
                skills.set_perturbation_rng(pending)
            print("[Method3 phase2] subgoal replay → _skills 부착 완료 "
                  "(Phase1 도달 경유점 고정, 경로만 curobo 로 다양화)")
        except Exception as e:
            print(f"[Method3 phase2] subgoal replay 부착 실패: {e}")

    def _phase2_dump_remote(self) -> str | None:
        """phase2_config.yaml 의 remote → 'user@host:<project>/results/phase2_cands'.

        server 의 candidate dump 디렉터리 — episode hook 이 scp 로 가져온다.
        """
        cfg_path = (Path(__file__).resolve().parent
                    / "pipeline_config" / "phase2_config.yaml")
        if not cfg_path.exists():
            return None
        try:
            import yaml as _yaml
            with open(cfg_path, "r") as f:
                cfg = _yaml.safe_load(f) or {}
            r = cfg.get("remote") or {}
            user, host, proj = r.get("user"), r.get("hostname"), r.get("project_path")
            if user and host and proj:
                return f"{user}@{host}:{proj}/results/phase2_cands"
        except Exception:
            pass
        return None

    def _dump_phase2_candidate_overlays(self, forward_dir: str) -> None:
        """이번 episode 의 Phase2 candidate dump 를 server 에서 가져와 top-view
        오버레이 PNG 를 forward_dir 에 ``skill{N}_{skill}.png`` 로 저장한다.

        server 가 plan_and_select 마다 ``phase2_cands/<selection_id>.npz`` 로
        dump 하고, GrpcPlannerClient 가 (selection_id, skill_id) 를 누적해 둔다.
        episode 종료 시 ``pop_dump_refs()`` 로 가져와 selection_id 로 scp →
        ``visualize_phase2_candidates.visualize`` 로 오버레이.
        """
        adapter = getattr(self, "_skill_planner_client", None)
        if adapter is None or not hasattr(adapter, "pop_dump_refs"):
            return
        refs = adapter.pop_dump_refs()
        if not refs:
            return
        remote = self._phase2_dump_remote()
        if not remote:
            print("  [phase2 overlay] phase2_config.remote 미설정 — skip")
            return
        import subprocess
        import sys
        import tempfile
        _scripts = str(Path(__file__).resolve().parent / "scripts")
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        try:
            from visualize_phase2_candidates import visualize
        except Exception as e:
            print(f"  [phase2 overlay] visualize import 실패: {e}")
            return
        # conda env(lerobot_cap)의 LD_LIBRARY_PATH 가 시스템 scp(OpenSSH)의
        # OpenSSL 을 깨뜨린다(version mismatch → rc=255). 제거한 env 로 scp.
        _scp_env = {k: v for k, v in os.environ.items()
                    if k != "LD_LIBRARY_PATH"}
        n_ok = 0
        for idx, (sel_id, skill) in enumerate(refs):
            tmp = tempfile.mktemp(suffix=".npz")
            try:
                r = subprocess.run(
                    ["scp", "-o", "ConnectTimeout=15",
                     f"{remote}/{sel_id}.npz", tmp],
                    capture_output=True, timeout=40, env=_scp_env,
                )
                if r.returncode != 0 or not os.path.exists(tmp):
                    print(f"  [phase2 overlay] scp 실패 {sel_id[:8]}: "
                          f"{r.stderr.decode('utf-8', 'ignore')[:120]}")
                    continue
                out = os.path.join(forward_dir, f"skill{idx}_{skill}.png")
                visualize(tmp, out_path=out)
                n_ok += 1
            except Exception as e:
                print(f"  [phase2 overlay] {sel_id[:8]} 실패: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        if n_ok:
            print(f"  [phase2 overlay] {n_ok}/{len(refs)} 개 → {forward_dir}")

    def _build_phase2_selector(self):
        """phase2_config.yaml.mi_selection 으로 Phase2MISelector 인스턴스화.

        ``_phase2_vector_db`` 가 없으면 None.
        """
        if self._phase2_vector_db is None:
            return None
        try:
            from method3.config import load_phase2_config
            from method3.phase2_mi_selection import Phase2MISelector
            acq = load_phase2_config(
                "pipeline_config/phase2_config.yaml",
                phase1_raw_dir="", phase2_raw_dir="",
            )
            sel = Phase2MISelector(
                vector_db=self._phase2_vector_db,
                config=acq.phase2_mi,
            )
            print(
                f"[Method3 phase2] selector ready — tau_MI={sel.cfg.tau_MI}, "
                f"k_nn_a={sel.cfg.k_nn_a}, q={sel.cfg.q_quantile}"
            )
            return sel
        except Exception as e:
            print(f"[Method3 phase2] selector setup FAILED: {e} — phase2 selection 불가")
            import traceback; traceback.print_exc()
            return None

    def _build_phase2_vla_scorer(self, vla_path: str | None):
        """yaml.vla_informativeness.enabled=true 면 LeRobotVLAInformativenessScorer
        를 만들어 반환. 그 외 (또는 로딩 실패) 는 None — selector 가 fallback 으로
        ``argmax M_MI among eligible`` 사용.

        VLA policy 는 preselective_filter 의 ``make_vla_key_extractor`` 가 wrap
        한 policy 를 그대로 재사용한다 (같은 freeze 된 Phase1-trained VLA).
        """
        if not vla_path:
            return None
        try:
            import yaml
            cfg_path = Path("pipeline_config/phase2_config.yaml")
            raw = yaml.safe_load(cfg_path.read_text()) or {} if cfg_path.exists() else {}
            vla_cfg = (raw.get("vla_informativeness") or {}) if isinstance(raw, dict) else {}
            if not vla_cfg.get("enabled", False):
                print(
                    "[Method3 phase2] vla_informativeness.enabled=false — U_VLA 비활성, "
                    "selection 은 argmax M_MI 로 fallback (backward-compat)"
                )
                return None
            from method3.vectorDB.vla_embedding import (
                make_vla_key_extractor,
            )
            from method3.phase2_mi_selection import (
                LeRobotVLAInformativenessScorer,
            )
            extractor = make_vla_key_extractor(vla_path)
            scorer = LeRobotVLAInformativenessScorer(
                policy=extractor.policy,
                R=int(vla_cfg.get("R", 8)),
                agg=str(vla_cfg.get("agg", "mean")),
            )
            print(
                f"[Method3 phase2] U_VLA scorer ready — "
                f"family={extractor.__class__.__name__} R={scorer.R} agg={scorer.agg}"
            )
            return scorer
        except Exception as e:
            print(
                f"[Method3 phase2] U_VLA scorer setup FAILED: {e} — "
                f"selection 은 argmax M_MI 로 fallback"
            )
            import traceback; traceback.print_exc()
            return None

    def _phase2_select(self, candidates):
        """Phase2MISelector + (옵셔널) VLA scorer 로 §13.2 Useful-OOD selection 실행.

        Phase2 candidate generator 가 후보 batch 를 만들면 호출된다. selector 가
        미준비 (phase1 모드 또는 setup 실패) 면 RuntimeError. candidate 가
        observations/instruction/proprios 를 채워 보내면 U_VLA 도 채점됨.
        """
        if self._phase2_selector is None:
            raise RuntimeError(
                "phase2 selector not initialized — call _setup_phase2_session first "
                "(make sure method3_phase='phase2' and P_phase1 build succeeded)."
            )
        return self._phase2_selector.select(
            candidates, vla_scorer=self._phase2_vla_scorer,
        )

    def _upload_p_phase1_to_server(self, session_dir: str) -> None:
        """client 가 build 한 P_phase1 npz 를 server 의 cache path 로 scp.

        yaml.remote 의 user/hostname/identity_file 정보로 ssh tunnel 없이 직접
        scp. server 가 다음 부팅 시 그 cache load (= 자동 ready). 실패는 warn
        만 — acquisition 자체는 client 측 _phase2_vector_db 가 있어 진행 가능.
        """
        import subprocess, yaml
        from pathlib import Path
        local_npz = Path(session_dir) / "skill_wise_vector_db.npz"
        if not local_npz.exists():
            print(f"[upload] local npz not found: {local_npz} — skip scp")
            return
        # yaml.remote 정보 — phase2_config.yaml 의 remote 섹션
        try:
            with open(Path(__file__).resolve().parent / "pipeline_config" / "phase2_config.yaml") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[upload] yaml read failed ({e}) — skip scp")
            return
        r = cfg.get("remote") or {}
        user = (r.get("user") or "").strip()
        host = (r.get("hostname") or "").strip()
        proj = r.get("project_path") or "~/AutoDataCollector"
        identity = (r.get("identity_file") or "").strip()
        if not (user and host):
            print("[upload] yaml.remote.user/hostname 미설정 — skip scp")
            return
        remote_path = f"{user}@{host}:{proj}/grpc_server/buffer/server_skill_wise_vector_db.npz"
        scp_cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
        if identity:
            scp_cmd += ["-i", str(Path(identity).expanduser())]
        scp_cmd += [str(local_npz), remote_path]
        # OpenSSL ABI 우회 (conda env)
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        try:
            print(f"[upload] scp → {remote_path}")
            res = subprocess.run(scp_cmd, env=env, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                print(f"[upload] ✓ uploaded {local_npz.stat().st_size // 1024} KB → "
                      f"server cache. *server restart 권장*: "
                      f"bash grpc_server/launch_remote_server.sh stop && bash grpc_server/launch_remote_server.sh")
            else:
                print(f"[upload] scp failed (exit {res.returncode}): {res.stderr[:200]}")
        except Exception as e:
            print(f"[upload] scp exception: {e}")

    def _load_phase2_g_seed_buffer(self, session_dir: str | None):
        """Phase1 누적 ``subgoal_buffer.npz`` 를 anchor buffer 로 로드 (§7).

        없으면 빈 buffer 반환 — phase2 hook 이 anchor lookup 실패 시 RNG fallback.
        """
        try:
            from method3.phase2_mi_selection.seed_anchor import load_g_seed
            buf = load_g_seed(session_dir or ".")
            print(
                f"[Method3 phase2] G_seed buffer loaded — "
                f"skills={list(buf._skills.keys()) if hasattr(buf, '_skills') else 'n/a'}, "
                f"total entries={len(buf) if hasattr(buf, '__len__') else 'n/a'}"
            )
            return buf
        except Exception as e:
            print(f"[Method3 phase2] G_seed load FAILED: {e} — anchor lookup 우회")
            return None

    def _attach_phase2_skill_hook(self) -> None:
        """skills_lerobot 의 ``_skill_candidate_selector`` hook 에 Useful-OOD 부착.

        selector / vector_db 미준비면 silent skip — phase2 동작은 안 되지만 phase1
        경로는 영향 없음. hook signature: ``(cands, current_joints, goal_joint_rad,
        is_transit) -> int | None`` — None 반환 시 skills_lerobot 가 RNG fallback.
        """
        if self._phase2_selector is None:
            return
        if not hasattr(self, "_skills") or self._skills is None:
            return
        try:
            self._skills.set_skill_candidate_selector(self._make_phase2_skill_selector())
            print("[Method3 phase2] skill candidate hook attached — Useful-OOD rule active")
        except Exception as e:
            print(f"[Method3 phase2] skill hook attach FAILED: {e}")

    def _make_phase2_skill_selector(self):
        """Returns the actual hook fn closed over self. Defined as a method 라
        Pipeline 인스턴스 상태에 접근 가능 (selector, encoder, buffer, recorder).
        """
        from method3.phase2_mi_selection.curobo_candidate_gen import (
            CurobogenConfig,
            candidates_from_trajectory_list,
        )
        from method3.phase2_mi_selection.seed_anchor import pick_anchor

        cfg = CurobogenConfig()

        def _hook(cands, current_joints, goal_joint_rad, is_transit) -> int | None:
            """ξ* index 반환. None 이면 skills_lerobot 의 RNG fallback."""
            if not cands:
                return None
            # candidate 의 vector DB partition 키 = episode-내 skill ordinal
            # (skill_0, skill_1, ...). _hook 은 move_to_position 실행 중
            # (plan_batch 직후, 해당 move 의 _set_skill_recording 이전) 에
            # 호출되므로 skill_sequence 에 현재 move 가 아직 미반영 →
            # len - _episode_skill_base 가 곧 현재 move 의 ordinal.
            # P_phase1 (skill_{skill_index}) 과 같은 키 공간 → MI 정합.
            _sk = getattr(self, "_skills", None)
            _ord = (len(_sk.skill_sequence) - getattr(_sk, "_episode_skill_base", 0)
                    ) if _sk is not None else 0
            skill_id = f"skill_{_ord}"

            # anchor: 현재 goal_joint_rad 의 EE xyz 추출은 비싸므로 일단 그대로
            # 단순화 — anchor 가 없어도 candidate 의 action 다양성으로 selection
            # rule 은 동작. seed_subgoal 은 candidate metadata 로만 의미.
            anchor = None
            if self._phase2_g_seed_buffer is not None:
                # G_seed 가 있고 lookup 가능한 형태면 그대로 — target_xyz 추출은
                # 호출자가 별도 wiring 할 때까지 None 으로 두고 buffer 의 첫 entry.
                try:
                    anchor = pick_anchor(self._phase2_g_seed_buffer, skill_id)
                except Exception:
                    anchor = None
            seed = anchor if anchor is not None else np.zeros(3, dtype=np.float64)

            # TrajectoryCandidate → Phase2Candidate. encoder 가 None 이면 state_keys
            # 는 proprio 만으로 구성 (degraded 모드).
            try:
                p2_cands = candidates_from_trajectory_list(
                    cands,
                    skill_id=skill_id,
                    seed_subgoal=seed,
                    current_observation=None,
                    instruction="",
                    encoder=None,
                    config=cfg,
                )
            except Exception as e:
                print(f"[Method3 phase2] candidate adapter failed: {e} — RNG fallback")
                return None
            if not p2_cands:
                return None

            try:
                result = self._phase2_select(p2_cands)
            except Exception as e:
                print(f"[Method3 phase2] selector failed: {e} — RNG fallback")
                return None

            # Verbose log — client 측에서도 paradigm 의 실 작동 확인 가능.
            try:
                eligible = list(getattr(result, "eligible_indices", []) or [])
                u_vla = getattr(result, "u_vla_chosen", None)
                rep = result.reports[result.chosen_index] if result.reports else None
                _r = (f"ΔH_A={rep.delta_h_a:.3f}, ΔH_A|S={rep.delta_h_a_given_s:.3f}, "
                      f"M̃_MI={rep.q2_norm:+.2f}, U_VLA={rep.u_vla:.3f}"
                      if rep is not None else "no report")
                print(
                    f"[Phase2-Selection] cands={len(p2_cands)} eligible={len(eligible)} "
                    f"chosen=#{result.chosen_index} accepted={result.accepted} "
                    f"u_vla_chosen={u_vla} | {_r}"
                )
            except Exception as e:
                print(f"[Phase2-Selection] verbose log failed: {e}")

            # §14 flush — accepted 면 즉시 phase2 vector DB 에 window 별 entry append.
            # ref/meta 는 episode/skill 정보 (호출자가 self 에 set 해두면 그대로 사용).
            if result.accepted and self._phase2_selector is not None:
                try:
                    self._phase2_selector.accept_to_buffer(
                        result.chosen_candidate,
                        ref={
                            "episode_id": getattr(self, "current_episode", None),
                            "skill_id": skill_id,
                        },
                        meta={
                            "algo": (result.chosen_candidate.payload or {}).get("algo", ""),
                            "u_vla": result.u_vla_chosen,
                        },
                    )
                except Exception as e:
                    print(f"[Method3 phase2] accept_to_buffer failed: {e}")

            return int(result.chosen_index)

        return _hook

    def _teardown_subgoal_selector(self) -> None:
        """Persist the buffer-aware subgoal buffer to its ``.npz`` file.

        No-op for the legacy gaussian mode or a memory-only buffer (no
        ``buffer_file`` given / file never bound by _finalize_subgoal_buffer).
        """
        selector = getattr(self, "_subgoal_selector", None)
        if selector is None:
            return
        try:
            selector.buffer.save()
            _fp = selector.buffer.file_path()
            if _fp is not None:
                print(
                    f"[Perturbation] subgoal buffer saved → {_fp} "
                    f"({selector.buffer.total_size()} entries over "
                    f"{len(selector.buffer.skill_ids())} skills)"
                )
        except Exception as e:
            print(f"[Perturbation] subgoal buffer persist skipped: {e}")
        self._subgoal_selector = None
        if hasattr(self, "_skills") and self._skills is not None:
            try:
                self._skills.set_subgoal_selector(None)
            except Exception:
                pass

    def _setup_skill_perturbation_on_skills(self) -> None:
        """Read skill_perturbation config and wire a curobo backend.

        우선순위:
          1. ``pipeline_config/phase2_config.yaml`` 의 ``skill_perturbation`` (신규)
          2. ``recording_config_ws*.yaml`` 의 ``perturbation.skill`` (legacy fallback)

        preselective_filter 도 phase2_config.yaml 우선으로 본다 (두 섹션 모두
        Phase2 의 dependency 라 같은 파일에 있는 게 자연스럽다).

        Silent no-op when both yaml absent, skill perturbation disabled,
        or transport=grpc (server side owns the planner in that case).
        """
        self._skill_planner_phase = {"forward": False, "reset": False}

        # 1) phase2_config.yaml 의 skill_perturbation / preselective_filter 우선
        skill_raw: dict = {}
        psf_raw: dict = {}
        ph2_path = Path(__file__).resolve().parent / "pipeline_config" / "phase2_config.yaml"
        if ph2_path.exists():
            try:
                import yaml as _yaml
                with open(ph2_path, "r") as f:
                    ph2_cfg = _yaml.safe_load(f) or {}
                skill_raw = ph2_cfg.get("skill_perturbation") or {}
                psf_raw = (
                    ph2_cfg.get("skill_planner_transport")
                    or ph2_cfg.get("preselective_filter")
                    or {}
                )
                if skill_raw:
                    print(f"[Skill Perturbation] phase2_config.yaml ← {ph2_path}")
            except Exception as e:
                print(f"[Skill Perturbation] phase2_config.yaml read failed ({e}); "
                      f"falling back to recording_config")

        # 2) fallback to recording_config (legacy)
        full_cfg: dict = {}
        if (not skill_raw or not psf_raw) and self.recording_config:
            cfg_path = Path(self.recording_config)
            if cfg_path.exists():
                try:
                    import yaml as _yaml
                    with open(cfg_path, "r") as f:
                        full_cfg = _yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"[Skill Perturbation] Failed to read recording_config: {e}")
                    return
                if not skill_raw:
                    skill_raw = (full_cfg.get("perturbation") or {}).get("skill") or {}
                if not psf_raw:
                    psf_raw = full_cfg.get("preselective_filter") or {}

        any_en, fwd, reset = self._read_phase_flags(skill_raw)
        if not any_en:
            return
        self._skill_planner_phase = {"forward": fwd, "reset": reset}

        # When skill_planner_transport mode is grpc, the remote H100 server owns
        # curobo + selection. Skip local backend creation — GrpcPlannerClient
        # adapter installed by _setup_skill_planner_transport satisfies
        # skills_lerobot's plan_batch contract on this side.
        # New key: mode. Legacy: transport.
        _mode = str(psf_raw.get("mode") or psf_raw.get("transport") or "local").lower()
        if _mode == "grpc":
            print(
                "[Skill Perturbation] mode=grpc — local skill planner skipped; "
                "GrpcPlannerClient is installed by skill_planner_transport."
            )
            return

        # Resolve URDF path from the skills' robot config YAML directly.
        # NOTE: self._skills.config is None until LeRobotSkills.connect() runs;
        # this hook fires earlier (during _create_skills), so we must read the
        # YAML ourselves from self._skills.robot_config_path (set in __init__).
        urdf_path = None
        try:
            import yaml as _yaml
            robot_cfg_path = getattr(self._skills, "robot_config_path", None)
            if robot_cfg_path is None or not Path(robot_cfg_path).exists():
                print(f"[Skill Perturbation] robot_config_path not available; skipping")
                return
            with open(robot_cfg_path, "r") as _f:
                _robot_cfg = _yaml.safe_load(_f) or {}
            urdf_path = (_robot_cfg.get("kinematics") or {}).get("urdf_path")
            # YAML may store relative paths; resolve against the YAML's directory.
            if urdf_path and not Path(urdf_path).is_absolute():
                urdf_path = str((Path(robot_cfg_path).parent / urdf_path).resolve())
        except Exception as _e:
            print(f"[Skill Perturbation] Failed to read robot config: {_e}; skipping")
            return
        if not urdf_path or not Path(urdf_path).exists():
            print(f"[Skill Perturbation] URDF not found ({urdf_path!r}); skipping")
            return

        # Robot id used to locate the curobo robot config under
        # robot_configs/curobo/<robot_id>.yml when no explicit override.
        robot_id = "default"
        try:
            robot_id = Path(self._skills.robot_config_path).stem
        except Exception:
            pass

        try:
            from perturbation.skill_level import get_curobo_backend
            CuroboBackend, CuroboBackendConfig = get_curobo_backend()
            # YAML may specify a custom curobo robot config path; fall back
            # to the auto-generated default under robot_configs/curobo/.
            # Relative paths MUST be resolved to absolute here — curobo's
            # internal robot-loader otherwise tries to resolve them against
            # its own content/configs/robot/ directory, producing nonsense
            # paths like ".../src/nvidia-curobo/.../robot_configs/curobo/X.yml".
            project_root = Path(self.recording_config).resolve().parent.parent
            default_curobo_cfg = str(
                project_root / "robot_configs" / "curobo" / f"{robot_id}.yml"
            )
            curobo_cfg_path = skill_raw.get("curobo_robot_cfg_path") or default_curobo_cfg
            _cp = Path(curobo_cfg_path)
            if not _cp.is_absolute():
                _cp = project_root / _cp
            curobo_cfg_path = str(_cp.resolve())
            if not Path(curobo_cfg_path).exists():
                print(
                    f"[Skill Perturbation] curobo robot config missing: "
                    f"{curobo_cfg_path}\n  Generate via: "
                    f"python -m curobo.examples.getting_started.build_robot_model "
                    f"--urdf {urdf_path} --output {curobo_cfg_path}"
                )
                return
            _n_cand = int(skill_raw.get("n_candidates", 4))
            cb_cfg = CuroboBackendConfig(
                enabled=True,
                robot_cfg_path=curobo_cfg_path,
                num_trajopt_seeds=int(skill_raw.get("curobo_num_trajopt_seeds", 4)),
                num_ik_seeds=int(skill_raw.get("curobo_num_ik_seeds", 16)),
                use_cuda_graph=bool(skill_raw.get("curobo_use_cuda_graph", False)),
                via_offset_mag=float(skill_raw.get("curobo_via_offset_mag", 0.10)),
                junction_smooth_k=int(skill_raw.get("curobo_junction_smooth_k", 5)),
                fixed_joint_indices=tuple(skill_raw.get("fixed_joint_indices") or ()),
                arm_joint_count=int(skill_raw.get("arm_joint_count", 5)),
                max_vias_per_candidate=int(
                    skill_raw.get("curobo_max_vias_per_candidate", 1)
                ),
                # Option A wrist-cam transit bias. Top-level transit_pitch_max_deg
                # (the same knob that gates endpoint IK) also clamps every
                # via-point orientation, so the whole curobo path stays
                # pitch-down. None when the key is absent.
                via_pitch_max_rad=(
                    None if full_cfg.get("transit_pitch_max_deg") is None
                    else float(np.radians(float(full_cfg["transit_pitch_max_deg"])))
                ),
                # CUDA graph batch must cover n_candidates; otherwise plan_batch
                # caps n via `min(n, max_batch_size)` and we silently get fewer.
                max_batch_size=_n_cand,
            )
            client = CuroboBackend(urdf=urdf_path, config=cb_cfg)
            self._skill_planner_client = client
            self._skills.set_skill_planner_client(client, n_candidates=_n_cand)
            print(
                f"[Skill Perturbation] CUROBO backend ENABLED "
                f"(robot_cfg={curobo_cfg_path}, n_candidates={_n_cand})"
            )
            # Apply pending per-episode seed (mirror subgoal setup). The RNG
            # check in skills_lerobot.move_to_position swap requires this.
            pending = getattr(self, "_pending_perturbation_seed", None)
            if pending is not None:
                self._skills.set_perturbation_rng(pending)
        except Exception as e:
            print(f"[Skill Perturbation] Failed to init curobo backend: {e}")
            self._skill_planner_client = None

    def _teardown_skill_perturbation(self) -> None:
        """Shut down the curobo backend (frees CUDA graph + planner state).

        Called between pipeline sessions and on Ctrl+C.
        """
        client = getattr(self, "_skill_planner_client", None)
        if client is None:
            return
        if hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
        self._skill_planner_client = None
        if hasattr(self, "_skills") and self._skills is not None:
            try:
                self._skills.set_skill_planner_client(None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Method3 phase2 — skill planner transport (gRPC to H100 server)
    # ------------------------------------------------------------------
    def _setup_skill_planner_transport(self, session_dir: str | None = None) -> None:
        """Wire the skill planner — local CuroboBackend or remote H100 gRPC.

        Read order:
          1. ``pipeline_config/phase2_config.yaml.skill_planner_transport`` (신규)
          2. ``pipeline_config/phase2_config.yaml.preselective_filter`` (legacy alias)
          3. ``recording_config_ws*.yaml.preselective_filter`` (legacy fallback)

        활성화 조건: yaml 의 ``mode`` 가 ``grpc`` 또는 ``local`` 일 때. ``disabled``
        / 미설정 / ``enabled_forward=false`` 면 silent no-op → robot 은 cartesian-
        line trajectory 그대로 (method3 phase2 useful-OOD selection 비활성).

        gRPC mode: H100 server (grpc_server.server) 가 method3 Phase2MISelector
        를 운영. 이 client 측은 GrpcPlannerClient 를 ``skills.set_skill_planner_client``
        에 install 만 한다. server 가 candidate 1개만 return 하므로 별도 selector
        hook 은 필요 없음 (``skills.set_skill_candidate_selector(None)``).
        """
        self._skill_planner_selector = None
        self._skill_planner_encoder = None
        self._skill_planner_phase_flags = {"forward": False, "reset": False}
        self._skill_planner_debug = False
        self._skill_planner_chunk_size = 50
        self._skill_planner_ingest_thread = None

        # Phase guard — skill_planner_transport 는 *Phase2 전용*. Phase1 은
        # cartesian baseline 이라 grpc/curobo 연결 자체가 의미 없음 (= GPU 부담만).
        _phase = getattr(self, "method3_phase", "phase1")
        if _phase != "phase2":
            print(f"[skill_planner_transport] method3_phase={_phase} → SKIPPED "
                  f"(grpc/curobo 는 Phase2 전용; Phase1 은 cartesian baseline)")
            return

        # mode + server 운영 설정 — phase2_config.yaml 만 읽는다.
        # 2026-05-21: 옛 phase2_server_infer_settings.yaml 의 transport/policy/selector
        # 섹션을 phase2_config.yaml top-level 로 통합. 단일 SoT.
        psf_raw: dict = {}
        full_cfg: dict = {}      # legacy fallback container — 미사용 path 라도 정의 보장
        cfg_src: str = ""
        pkg_root = Path(__file__).resolve().parent / "pipeline_config"
        ph2_path = pkg_root / "phase2_config.yaml"
        try:
            import yaml as _yaml
        except ImportError:
            _yaml = None
        if _yaml and ph2_path.exists():
            try:
                with open(ph2_path, "r") as f:
                    ph2_cfg = _yaml.safe_load(f) or {}
                psf_raw = ph2_cfg.get("skill_planner_transport") or ph2_cfg.get("preselective_filter") or {}
                if psf_raw:
                    key = "skill_planner_transport" if ph2_cfg.get("skill_planner_transport") else "preselective_filter (legacy)"
                    cfg_src = f"phase2_config.yaml [{key}]"
                    print(f"[skill_planner_transport] mode-yaml ← {ph2_path} [{key}]")
                # transport/policy/selector 머지 — 같은 yaml 의 top-level.
                t = ph2_cfg.get("transport") or {}
                psf_raw.setdefault("transport_address", t.get("address"))
                psf_raw.setdefault("transport_timeout_s", t.get("timeout_s"))
                psf_raw.setdefault("debug_verbose", t.get("debug_verbose", False))
                if "policy" not in psf_raw and "policy" in ph2_cfg:
                    psf_raw["policy"] = ph2_cfg["policy"]
                if "selector" not in psf_raw and "selector" in ph2_cfg:
                    psf_raw["selector"] = ph2_cfg["selector"]
            except Exception as e:
                print(f"[skill_planner_transport] phase2_config.yaml read failed ({e}); "
                      f"falling back to recording_config")
        if not psf_raw and self.recording_config:
            cfg_path = Path(self.recording_config)
            if cfg_path.exists():
                try:
                    import yaml as _yaml
                    with open(cfg_path, "r") as f:
                        full_cfg = _yaml.safe_load(f) or {}
                    psf_raw = full_cfg.get("preselective_filter") or {}
                except Exception as e:
                    print(f"[skill_planner_transport] failed to read recording_config: {e}")
                    return
        if not psf_raw:
            return
        self._skill_planner_debug = bool(psf_raw.get("debug_verbose", False))
        # forward-only by design — reset 은 curobo plan 안 함.
        # 신규: mode (disabled | grpc | local). 레거시: transport + enabled_forward.
        mode = str(psf_raw.get("mode", "")).strip().lower()
        if not mode:
            # legacy: transport=grpc + enabled_forward=true 이면 mode=grpc 로 해석.
            transport = str(psf_raw.get("transport", "")).strip().lower()
            legacy_enabled = bool(psf_raw.get("enabled_forward", psf_raw.get("enabled", False)))
            if not legacy_enabled:
                mode = "disabled"
            else:
                mode = transport or "local"
        if mode == "disabled":
            print("[skill_planner_transport] mode=disabled — skill planner 미설치 "
                  "(robot 은 cartesian-line trajectory). gRPC 활성화: yaml 의 "
                  "skill_planner_transport.mode 를 'grpc' 로.")
            return
        self._skill_planner_phase_flags = {"forward": True, "reset": False}
        self._skill_planner_chunk_size = int(
            (psf_raw.get("selector") or {}).get("chunk_size", 50)
        )
        transport = mode  # downstream branch uses `transport` 변수명
        print(
            f"[skill_planner_transport] mode={mode} (source={cfg_src or 'recording_config'})"
        )

        # ──────────────────────────────────────────────────────────────
        # Branch on transport mode.
        #
        # NOTE (refactor 2026-05-20): IG·AC Selector / FAISS / pipeline_setup
        # 의 *local* branch 는 제거됐다. selection 책임은 method3 의 phase2
        # useful-OOD selector (final_method3_spec_useful_ood_updated §11-13)
        # 로 이동. 이 메서드의 유일한 의무는 transport=grpc 일 때 H100 서버의
        # curobo plan_batch 를 호출하는 GrpcPlannerClient 를 install 하는 것.
        #
        # transport != "grpc" 면 silent no-op (local IG·AC 모드 의도된 제거).
        # ──────────────────────────────────────────────────────────────
        if transport == "grpc":
            self._setup_skill_planner_grpc(psf_raw, full_cfg)
            return
        # local IG·AC selector removed — method3 phase2 가 대체. no-op.
        print(
            f"[skill_planner_transport] mode={transport} 는 더 이상 지원되지 "
            "않음 (IG·AC selector 는 method3 phase2 useful-OOD 로 이동). "
            "gRPC mode 만 지원."
        )

    def _setup_skill_planner_grpc(self, psf_raw: dict, full_cfg: dict) -> None:
        """gRPC transport branch — remote server owns curobo + Selector.

        Side effects:
          - self._skill_planner_grpc_client : PreselectiveClient (for commit())
          - self._skill_planner_client     : GrpcPlannerClient adapter
          - skills.set_skill_planner_client(adapter, n_candidates)
          - skills.set_skill_candidate_selector(None)  — server already chose
        """
        addr = str(psf_raw.get("transport_address", "127.0.0.1:50061"))
        timeout_s = float(psf_raw.get("transport_timeout_s", 60.0))
        # ANSI 빨간색 안내 — mode=grpc 일 때 fail 은 *silent fallback 금지*. spec
        # 위반 데이터 수집 (cartesian-line trajectory only) 을 막기 위해 명시 종료.
        _RED = "\033[91m"; _BOLD = "\033[1m"; _RST = "\033[0m"
        try:
            from grpc_server.client import PreselectiveClient
            from method3.phase2_server_inference.grpc_planner_adapter import GrpcPlannerClient
        except Exception as e:
            msg = (
                f"\n{_RED}{_BOLD}[skill_planner_transport] grpc imports failed: {e}{_RST}\n"
                f"{_RED}    grpcio / grpc_server / method3 모듈을 import 못 함. 확인:{_RST}\n"
                f"{_RED}      1) conda env 활성 확인 (현재 conda env 안에서 실행 중?)         {_RST}\n"
                f"{_RED}      2) pip install grpcio grpcio-tools                              {_RST}\n"
                f"{_RED}      3) python -c \"from grpc_server.client import PreselectiveClient\"{_RST}\n"
                f"{_RED}{_BOLD}    → session 종료 (mode=grpc 일 때 silent fallback 금지).{_RST}\n"
            )
            print(msg)
            raise RuntimeError(
                "grpc client imports failed — install grpcio + verify grpc_server module"
            )

        client = PreselectiveClient(server_address=addr, timeout_s=timeout_s)
        try:
            info = client.ready()
        except Exception as e:
            try:
                client.close()
            except Exception:
                pass
            msg = (
                f"\n{_RED}{_BOLD}[skill_planner_transport] grpc Ready() failed at {addr}: {e}{_RST}\n"
                f"{_RED}    → gRPC 서버 연결 실패. 다음 절차로 서버를 띄우세요:{_RST}\n"
                f"{_RED}      1) bash grpc_server/launch_remote_server.sh                     {_RST}\n"
                f"{_RED}         (yaml.remote 섹션의 ssh + GPU + tmux 자동 설정)             {_RST}\n"
                f"{_RED}      2) Ready 응답 확인:                                              {_RST}\n"
                f"{_RED}         python -m grpc_server.tools.check_ready --address {addr}    {_RST}\n"
                f"{_RED}      3) Ready 되면 ws3.sh 다시 실행.                                 {_RST}\n"
                f"{_RED}{_BOLD}    → session 종료 (mode=grpc 일 때 silent fallback 금지).{_RST}\n"
            )
            print(msg)
            raise RuntimeError(
                f"grpc Ready() failed at {addr} — start the server with "
                f"'bash grpc_server/launch_remote_server.sh' first"
            )
        print(
            f"[skill_planner_transport] grpc connected: {addr} | "
            f"device={info.get('device')} buffer={info.get('buffer_total')} "
            f"selector={info.get('selector_summary')}"
        )

        # P_phase1 (= server-side skill-wise vector DB) ready 검증.
        # server 가 자체 build 했어야 useful-OOD selection 의 kNN reference 가
        # P_phase1 ∪ D_phase2 (spec §14). buffer_total=0 이면 build 실패 또는
        # 미수행 — *silent fallback 금지*, 빨간색 + RuntimeError 로 세션 종료.
        server_buf_total = int(info.get("buffer_total", 0))
        if server_buf_total == 0:
            try:
                client.close()
            except Exception:
                pass
            msg = (
                f"\n{_RED}{_BOLD}[skill_planner_transport] server P_phase1 vector DB is EMPTY (buffer_total=0){_RST}\n"
                f"{_RED}    Phase2 useful-OOD selection 은 P_phase1 baseline 없이 의미가 없습니다.{_RST}\n"
                f"{_RED}    → 다음을 확인:{_RST}\n"
                f"{_RED}      1) yaml 의 phase1_dataset_path 가 *유효한 LeRobot dataset* 인지        {_RST}\n"
                f"{_RED}      2) server log 의 [method3_setup] server-side P_phase1 build 메시지     {_RST}\n"
                f"{_RED}         ssh {{remote}} 'tail -200 /tmp/phase2_server.log | grep method3_setup'  {_RST}\n"
                f"{_RED}      3) server cache 삭제 후 재부팅 (= fresh build):                         {_RST}\n"
                f"{_RED}         ssh {{remote}} 'rm -f .../grpc_server/buffer/server_skill_wise_vector_db.npz'{_RST}\n"
                f"{_RED}         bash grpc_server/launch_remote_server.sh stop && bash grpc_server/launch_remote_server.sh{_RST}\n"
                f"{_RED}{_BOLD}    → session 종료 (P_phase1 없이 acquisition 진행 금지).{_RST}\n"
            )
            print(msg)
            raise RuntimeError(
                f"server P_phase1 vector DB is empty (buffer_total=0) at {addr} — "
                f"check yaml.phase1_dataset_path and server build logs"
            )

        adapter = GrpcPlannerClient(client=client, context_provider=self)
        # phase2_config.yaml.skill_perturbation (신규 통합 위치) 우선 —
        # recording_config 의 옛 perturbation.skill path 는 fallback.
        _ph2_skill: dict = {}
        try:
            import yaml as _yaml
            from pathlib import Path as _P
            _ph2_path = _P(__file__).resolve().parent / "pipeline_config" / "phase2_config.yaml"
            if _ph2_path.exists():
                _ph2_skill = (_yaml.safe_load(open(_ph2_path)) or {}).get("skill_perturbation") or {}
        except Exception:
            _ph2_skill = {}
        _legacy_skill = (full_cfg.get("perturbation") or {}).get("skill") or {}
        n_cand = int(_ph2_skill.get("n_candidates", _legacy_skill.get("n_candidates", 4)))
        print(f"[skill_planner_transport] n_candidates = {n_cand} "
              f"(source={'phase2_config.skill_perturbation' if _ph2_skill else 'recording.perturbation.skill' if _legacy_skill else 'default-4'})")

        self._skill_planner_grpc_client = client
        self._skill_planner_client = adapter
        # _create_skills lazy 호출 대비 — n_candidates 저장 후 새 _skills 가
        # 만들어질 때 retro-attach 가능.
        self._skill_planner_n_candidates = int(n_cand)
        if hasattr(self, "_skills") and self._skills is not None:
            self._skills.set_skill_planner_client(adapter, n_candidates=n_cand)
            self._skills.set_skill_candidate_selector(None)  # server picks
            # Re-apply pending RNG so skills_lerobot's plan_batch swap fires.
            pending = getattr(self, "_pending_perturbation_seed", None)
            if pending is not None:
                self._skills.set_perturbation_rng(pending)
            print(
                f"[skill_planner_transport] grpc planner installed on skills "
                f"(K={n_cand}, no local hook)"
            )

    # _build_skill_planner_hook 제거됨 (2026-05-20 refactor):
    # local IG·AC selector hook 은 method3 phase2 useful-OOD 가 대체.
    # gRPC transport 만 살아남았고 그 경우엔 server 가 candidate 선택을 끝내고
    # plan_batch 반환 시점에 이미 1개만 옴 → 별도 selector hook 불필요.

    def _latest_observation_dict(self) -> dict:
        """Return latest camera frames as {camera_name: np.ndarray}.

        Falls back to {} if the async capture surface isn't available — the
        VLA encoder tolerates an empty image dict (language+state only).
        """
        try:
            from record_dataset.context import RecordingContext
            cap = getattr(RecordingContext, "_async_capture", None)
            if cap is None:
                return {}
            imgs = cap.get_latest_images()
            return imgs or {}
        except Exception:
            return {}

    def _latest_robot_state(self) -> "np.ndarray | None":
        """Return current robot proprio (servo positions, normalize=True) — 6-dim.

        Same space as lerobot dataset `observation.state` (range ±100). Used by
        the gRPC planner adapter so the server's candidate state_keys align
        unit-wise with the DB build proprio (Phase2 paradigm consistency).
        Falls back to None if robot is not connected.
        """
        import numpy as _np
        try:
            sk = getattr(self, "_skills", None)
            if sk is None or not getattr(sk, "robot", None):
                return None
            pos = sk.robot.read_positions(normalize=True)
            return _np.asarray(pos, dtype=_np.float32)
        except Exception:
            return None

    def _start_demo_ingest_async(self) -> None:
        """Kick off demo ingestion in the background.

        The per-frame VLA encoding is GPU-heavy (tens of seconds). It is run on
        a daemon thread so it overlaps the NEXT episode's codegen/verify phase
        (no robot motion, GPU idle). ``_await_demo_ingest`` blocks before the
        next rollout so the new demo is in the vector DB before IG·AC runs.
        """
        import threading

        self._await_demo_ingest()  # ensure no prior ingest is still running
        t = threading.Thread(
            target=self._demo_ingest_worker, name="demo-ingest", daemon=True,
        )
        self._skill_planner_ingest_thread = t
        t.start()

    def _await_demo_ingest(self) -> None:
        """Block until the background demo ingestion finishes (if running)."""
        t = getattr(self, "_skill_planner_ingest_thread", None)
        if t is None:
            return
        if t.is_alive():
            import time as _t
            t0 = _t.time()
            t.join()
            if self._skill_planner_debug:
                print(
                    f"[skill_planner_transport] waited "
                    f"{_t.time() - t0:.1f}s for demo ingestion"
                )
        self._skill_planner_ingest_thread = None

    def _demo_ingest_worker(self) -> None:
        """Grow the *remote* vector DB from the just-saved forward demo episode.

        gRPC mode only — stream the episode to the H100 server, which encodes
        keys with its VLA and grows its server-side buffer. Local IG·AC + FAISS
        buffer 는 method3 phase2 useful-OOD 가 대체했으므로 local 모드 X.
        """
        recorder = getattr(self, "dataset_recorder", None)
        if recorder is None:
            return
        ds = getattr(recorder, "_dataset", None)
        if ds is None:
            return
        chunk = int(getattr(self, "_skill_planner_chunk_size", 50))
        grpc_client = getattr(self, "_skill_planner_grpc_client", None)
        if grpc_client is None:
            return  # local 모드는 제거됨 — gRPC 외엔 ingest 안 함
        try:
            res = grpc_client.ingest_episode(
                dataset_root=str(ds.root), chunk_size=chunk,
            )
            if self._skill_planner_debug:
                print(
                    f"[skill_planner_transport] grpc ingested {res['frames']} "
                    f"demo frames (ep {res['episode']}); "
                    f"buffer totals={res['buffer_totals']}"
                )
        except Exception as e:
            print(f"[skill_planner_transport] demo ingest failed: {e}")

    def _teardown_skill_planner_transport(self) -> None:
        """Close the gRPC channel at session end.

        Local IG·AC selector/encoder cleanup 은 더 이상 필요 없다 — local 모드
        제거됨. gRPC client close 만 남음.
        """
        # 진행 중인 비동기 demo ingest 가 끝나야 channel close 가 안전.
        self._await_demo_ingest()
        # Selector/encoder 필드는 local 모드 제거 후 항상 None — 호환 위해 reset 만.
        self._skill_planner_selector = None
        self._skill_planner_encoder = None
        # Close gRPC channel if we were in remote mode.
        grpc_cli = getattr(self, "_skill_planner_grpc_client", None)
        if grpc_cli is not None:
            try:
                grpc_cli.close()
            except Exception:
                pass
            self._skill_planner_grpc_client = None
        if hasattr(self, "_skills") and self._skills is not None:
            try:
                self._skills.set_skill_candidate_selector(None)
            except Exception:
                pass

    def _seed_episode_perturbation(self, batch_index: int, slot_in_batch: int) -> None:
        """Seed per-episode RNG for subgoal perturbation. No-op if disabled.

        Mirrors robotwin's seeding pattern: ``batch_index * 10000 + slot``.
        Same (batch_index, slot) → same offset sequence (reproducible).

        Safe to call before skills are lazily created — the seed is held until
        ``_create_skills`` runs, then applied via ``_setup_perturbation_on_skills``.
        """
        ep_seed = int(batch_index) * 10000 + int(slot_in_batch)
        self._pending_perturbation_seed = ep_seed
        # Seed the RNG when EITHER subgoal-level or skill-level perturbation
        # is attached — both branches in skills_lerobot use _perturbation_rng.
        if hasattr(self, "_skills") and self._skills is not None:
            # buffer_aware 모드는 _subgoal_selector 를, legacy gaussian 은
            # _perturbation 을 attach 한다 (둘은 서로 배타적). 둘 중 하나라도
            # 붙어 있으면 RNG 시딩이 필요한데, 기존 코드가 _subgoal_selector
            # 를 빼먹어서 resume 시 buffer_aware Phase1 이 비활성화됐었다 —
            # _restore_to_seed 가 _create_skills 를 episode loop 전에 트리거
            # 하면 setup 의 pending-seed apply 가 pending 미설정 상태로 통과,
            # 이후 매 episode 의 _seed_episode_perturbation 도 selector 만
            # 붙은 상태를 인식 못 해 RNG 가 영원히 None 으로 남았다.
            has_subgoal = (
                getattr(self._skills, "_perturbation", None) is not None
                or getattr(self._skills, "_subgoal_selector", None) is not None
            )
            has_skill = getattr(self._skills, "_skill_planner_client", None) is not None
            if has_subgoal or has_skill:
                self._skills.set_perturbation_rng(ep_seed)

        # Phase2 — episode 별 Phase1 도달 subgoal replay 컨텍스트 전환.
        # current_episode 는 episode 루프가 이 호출 직전에 설정한다 → episode_NN
        # 의 기록 subgoal 로 replay 커서를 리셋한다.
        _replay = getattr(self, "_phase2_subgoal_replay", None)
        if _replay is not None:
            try:
                from method3.episode_lifecycle import episode_id as _mk_ep_id
                _replay.set_episode(_mk_ep_id(int(self.current_episode)))
            except Exception as _e:
                print(f"[Method3 phase2] subgoal replay set_episode 실패: {_e}")

    def _get_pipeline_camera(self):
        """PipelineCamera lazy init."""
        if not hasattr(self, '_pipeline_camera') or self._pipeline_camera is None:
            from pipeline.camera_session import PipelineCamera
            self._pipeline_camera = PipelineCamera(
                camera_manager=self.camera_manager, verbose=self.verbose)
        return self._pipeline_camera

    def initialize_camera(self) -> bool:
        """카메라 초기화 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera_manager = self.camera_manager  # 동기화
        result = pc.initialize()
        self.camera = pc.camera
        return result

    def shutdown_camera(self) -> None:
        """카메라 종료 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera = self.camera  # 동기화
        pc.shutdown()
        self.camera = pc.camera  # None으로 반영

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
            from record_dataset.config import (
                create_camera_manager_from_config,
                load_cameras_from_yaml,
                build_features_from_yaml,
            )

            print(f"\n[Recording] Initializing multi-camera dataset recorder...")
            print(f"  Repo ID: {self.dataset_repo_id}")
            print(f"  FPS: {self.recording_fps}")

            # 1. 카메라 매니저 초기화 (YAML에서 동적 로드, 싱글암: left_arm만)
            print(f"[Recording] Loading camera configuration...")
            self.camera_manager = create_camera_manager_from_config(yaml_path=self.recording_config, num_robots=1)

            # 2. 카메라 연결
            print(f"[Recording] Connecting cameras...")
            self.camera_manager.connect_all()
            print(f"[Recording] Cameras connected: {self.camera_manager.camera_names}")

            # 3. 카메라 연결 검증
            cameras = load_cameras_from_yaml(yaml_path=self.recording_config, num_robots=1)
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
                config_path = self.recording_config or "pipeline_config/recording_config.yaml"
                raise AssertionError(
                    f"\n"
                    f"========================================\n"
                    f"Camera connection failed!\n"
                    f"========================================\n"
                    f"The following cameras are enabled in {config_path}\n"
                    f"but failed to connect:\n"
                    + "\n".join(missing_details) + "\n"
                    f"\n"
                    f"To fix, either:\n"
                    f"  1. Connect the camera hardware and verify device path\n"
                    f"     (run: v4l2-ctl --list-devices)\n"
                    f"  2. Set 'enabled: false' for unavailable cameras in\n"
                    f"     {config_path}\n"
                    f"========================================"
                )

            # 4. Features 빌드 (연결된 카메라 기준, 멀티암과 동일 패턴)
            features = build_features_from_yaml(yaml_path=self.recording_config, num_robots=1)

            # 5. 기존 dataset 존재 여부 미리 체크 (forward + reset)
            from lerobot.utils.constants import HF_LEROBOT_HOME
            reset_repo_id = self.dataset_repo_id + "_reset"

            # Reset *frame* dataset writer toggle (default ON).
            # ``recording_config.enable_reset_dataset_recording: false`` 로 끄면
            # forward 만 dataset 으로 ingest 되고 reset frame 은 기록 안 함.
            # session 폴더의 reset/ 메타 (judge·crops·code) 는 계속 저장.
            self._enable_reset_dataset_recording = True
            if self.recording_config:
                try:
                    import yaml as _yaml
                    with open(self.recording_config, "r", encoding="utf-8") as _f:
                        _rc = _yaml.safe_load(_f) or {}
                    self._enable_reset_dataset_recording = bool(
                        _rc.get("enable_reset_dataset_recording", True)
                    )
                except Exception as _e:
                    print(f"[Recording] enable_reset_dataset_recording read failed "
                          f"({_e}); defaulting to ON")

            existing_repos = [self.dataset_repo_id]
            if self._enable_reset_dataset_recording:
                existing_repos.append(reset_repo_id)
            existing = []
            for rid in existing_repos:
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

            # 6. 레코더 초기화 (빌드된 features + config_yaml 전달, 멀티암과 동일 패턴)
            self.dataset_recorder = DatasetRecorder(
                repo_id=self.dataset_repo_id,
                fps=self.recording_fps,
                resume=self.resume_recording,
                features=features,
                config_yaml=self.recording_config,
                num_robots=1,
            )
            print(f"[Recording] Recorder initialized successfully")
            print(f"[Recording] Features: {list(self.dataset_recorder.features.keys())}")

            # 7. Reset 레코더 초기화 (별도 dataset, 동일 features)
            # enable_reset_dataset_recording=false 이면 None 유지 — 호출 site
            # (_start_reset_episode_recording / _end_reset_episode_recording) 는
            # ``if self.record_dataset and self.reset_dataset_recorder`` 가드로 보호.
            if self._enable_reset_dataset_recording:
                print(f"\n[Recording] Initializing reset dataset recorder...")
                print(f"  Reset Repo ID: {reset_repo_id}")
                self.reset_dataset_recorder = DatasetRecorder(
                    repo_id=reset_repo_id,
                    fps=self.recording_fps,
                    resume=self.resume_recording,
                    features=features,
                    config_yaml=self.recording_config,
                    num_robots=1,
                )
                print(f"[Recording] Reset recorder initialized")
            else:
                self.reset_dataset_recorder = None
                print(
                    f"\n[Recording] Reset dataset recording DISABLED "
                    f"(recording_config.enable_reset_dataset_recording=false). "
                    f"Reset frame data 는 ingest 되지 않고, judge/positions/code 메타만 "
                    f"session 폴더에 저장됩니다."
                )

            # Signal handler: Ctrl+C 시 finalize() 호출하여 데이터셋 보존
            self._install_recording_signal_handler()

        except ImportError as e:
            # `--record` was explicitly requested but the recording stack
            # cannot even be imported. Fail fast — silently disabling recording
            # here means the whole multi-episode session runs collecting NO
            # data, and the loss is only discovered afterwards.
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"[Recording] FATAL: dataset recording was requested (--record) "
                f"but record_dataset could not be imported: {e}\n"
                f"Fix the import error, or drop --record to run without recording."
            ) from e
        except AssertionError:
            # Dataset already exists - 파이프라인 완전 종료
            raise
        except Exception as e:
            # Same rationale as above: a recorder/camera init failure must abort
            # the run, not silently turn `record_dataset` off. A data-collection
            # pipeline that runs without recording is worse than one that stops.
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"[Recording] FATAL: dataset recording was requested (--record) "
                f"but recorder initialization failed: {e}\n"
                f"Fix the recorder/camera error, or drop --record to run without recording."
            ) from e

    def _start_reset_episode_recording(self, target_positions: Dict) -> None:
        """Reset 에피소드 레코딩 시작 (별도 dataset, recorder 교체)"""
        try:
            from record_dataset.context import RecordingContext

            # Reset recorder로 episode 시작
            self.reset_dataset_recorder.start_episode(task=self.reset_instruction)
            print(f"[Reset Recording] Episode started: {self.reset_instruction}")

            # RecordingContext의 recorder를 reset recorder로 교체
            self._saved_forward_recorder = RecordingContext._recorder
            RecordingContext._recorder = self.reset_dataset_recorder
        except Exception as e:
            print(f"[Reset Recording] Warning: Failed to start: {e}")

    def _end_reset_episode_recording(self, discard: bool = False) -> None:
        """Reset 에피소드 레코딩 종료 (forward recorder로 복원)"""
        try:
            from record_dataset.context import RecordingContext

            # Reset episode 종료
            info = self.reset_dataset_recorder.end_episode(discard=discard)
            if not discard:
                print(f"[Reset Recording] Episode saved: {info.get('num_frames', 0)} frames")
            else:
                print(f"[Reset Recording] Episode discarded")

            # Forward recorder로 복원
            RecordingContext._recorder = self._saved_forward_recorder
            RecordingContext.clear_subtask()
        except Exception as e:
            print(f"[Reset Recording] Warning: Failed to end: {e}")

    def _finalize_recording(self) -> None:
        """데이터셋 레코딩 완료 및 카메라 연결 해제"""
        if self.dataset_recorder and self.record_dataset:
            try:
                self.dataset_recorder.finalize()
                print(f"\n[Recording] Forward dataset finalized at: {self.dataset_recorder._dataset.root}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to finalize forward dataset: {e}")

        if self.reset_dataset_recorder:
            try:
                self.reset_dataset_recorder.finalize()
                print(f"[Recording] Reset dataset finalized at: {self.reset_dataset_recorder._dataset.root}")
            except Exception as e:
                print(f"[Recording] Warning: Failed to finalize reset dataset: {e}")

        # 멀티 카메라 연결 해제
        if self.camera_manager:
            try:
                self.camera_manager.disconnect_all()
                print(f"[Recording] Camera manager disconnected")
            except Exception as e:
                print(f"[Recording] Warning: Failed to disconnect cameras: {e}")
            self.camera_manager = None

        # Skill-level perturbation: shut down curobo backend if it was spawned
        try:
            self._teardown_skill_perturbation()
            self._teardown_skill_planner_transport()
            self._teardown_subgoal_selector()
        except Exception as e:
            print(f"[Recording] Warning: skill perturbation teardown failed: {e}")


    def capture_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 캡처 (PipelineCamera에 위임)."""
        pc = self._get_pipeline_camera()
        pc.camera_manager = self.camera_manager  # 동기화
        pc.camera = self.camera
        return pc.capture_frame()

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
                external_camera = self._get_pipeline_camera().get_realsense()
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
                active_camera = self._get_pipeline_camera().get_realsense()
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
            skip_turn_test=self.skip_turn_test,
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

    def _extract_skill_sequence_from_skills(self, skills):
        """skills 인스턴스에서 스킬 시퀀스를 추출"""
        try:
            if hasattr(skills, 'skill_sequence'):
                self._last_skill_sequence = skills.skill_sequence
        except Exception:
            pass

    @staticmethod
    def _patch_code_block(code: str, block_name: str, new_values: Dict) -> str:
        """코드 내 지정된 dict 블록의 좌표를 치환.

        예: _patch_code_block(code, "target_positions", {...})
            → target_positions = { ... } 블록 내 좌표만 교체
        """
        import re
        block_match = re.search(rf'({block_name}\s*=\s*\{{)(.*?)(\}})', code, re.DOTALL)
        if not block_match:
            return code

        block_body = block_match.group(2)
        for name, info in new_values.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos is None or len(pos) < 3:
                continue
            pattern = rf'("{name}":\s*\[)[^\]]+(\])'
            replacement = rf'\g<1>{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}\2'
            block_body = re.sub(pattern, replacement, block_body)

        return code[:block_match.start()] + block_match.group(1) + block_body + block_match.group(3) + code[block_match.end():]

    @staticmethod
    def _patch_reset_code_targets(code: str, new_targets: Dict) -> str:
        """Reset 코드 내 target_positions 좌표 치환 (하위호환)."""
        return ForwardAndResetPipeline._patch_code_block(code, "target_positions", new_targets)

    def dry_run_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """생성된 코드를 IK 검증만으로 가상 실행 (로봇 연결 없음).

        LeRobotSkills를 DryRunSkills로 교체하여 실행.
        모든 move/pick/place가 IK 체크로 대체됨.
        """
        from skills.dry_run_skills import DryRunSkills

        try:
            dry_skills = DryRunSkills(
                robot_config=f"robot_configs/robot/so101_robot{self.robot_id}.yaml",
                frame="base_link",
            )
            exec_globals = {
                "__name__": "__generated__",
                "skills": dry_skills,
                "positions": positions,
            }
            if extra_globals:
                exec_globals.update(extra_globals)
            exec(code, exec_globals)
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

    def _get_task_runner(self):
        """TaskRunner 인스턴스 반환 (lazy init).

        Note: 캐시된 runner가 있어도 recorder는 매 호출마다 현재 상태로 동기화함.
              이유: _restore_to_seed 등에서 self.record_dataset을 일시적으로 False로
                   바꾸는 케이스가 있어, 그 시점에 처음 캐싱되면 영구적으로
                   recorder=None으로 굳어버리는 stale cache 버그가 발생.
        """
        if not hasattr(self, '_task_runner') or self._task_runner is None:
            from pipeline.task_runner import SingleArmTaskRunner
            skills = self._create_skills()
            # The Phase1 subgoal selector is created lazily inside
            # `_create_skills`, so `self._subgoal_selector` only exists now.
            # `_finalize_subgoal_buffer` already ran once (in
            # run_multiple_episodes, before this lazy init) and no-op'd — the
            # .npz file was never bound, making every flush_episode save() a
            # silent no-op. Re-bind it here so the buffer actually persists.
            self._finalize_subgoal_buffer(getattr(self, "_session_dir", None))
            self._task_runner = SingleArmTaskRunner(
                skills=skills,
                recorder=self.dataset_recorder if self.record_dataset else None,
                camera_manager=self.camera_manager,
                camera=self.camera,
                recording_fps=self.recording_fps,
            )
        else:
            # 매 호출마다 recorder 동기화 (stale cache 방지)
            self._task_runner.recorder = self.dataset_recorder if self.record_dataset else None
            self._task_runner.camera_manager = self.camera_manager
            self._task_runner.camera = self.camera
        return self._task_runner

    def execute_code(self, code: str, positions: Dict, extra_globals: Dict = None) -> bool:
        """생성된 코드 실행 (TaskRunner에 위임)"""
        self._last_skill_sequence = []
        runner = self._get_task_runner()
        # 카메라가 나중에 초기화될 수 있으므로 동기화
        if self.camera and runner.skills:
            runner.skills.camera = self.camera
        success = runner.execute(code, positions, extra_globals)
        # 스킬 시퀀스 추출
        self._extract_skill_sequence_from_skills(runner.skills)
        return success

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
                external_camera = self._get_pipeline_camera().get_realsense()
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
            resetspace=self.resetspace,
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
                active_camera = self._get_pipeline_camera().get_realsense()
            except KeyError:
                pass
        if active_camera is None:
            active_camera = self.camera

        reset_code, current_pos, target_pos, grippable, obstacles, reset_mt_info = lerobot_reset_code_gen_multi_turn(
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
            resetspace=self.resetspace,
            reset_instruction=self.reset_instruction,
        )

        self.reset_multi_turn_info = reset_mt_info

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
        skip_forward: bool = False,
        reset_target_positions: Optional[Dict] = None,
        pre_reset_callback=None,
        post_judge_callback=None,
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

            if skip_forward:
                # forward 외부에서 이미 수행 (forward_ma 의 MA-style forward 등). reset 에 필요한
                # 초기 이미지 + detection 만 수행하고 forward 로깅 종료 후 reset 단계로 진행.
                print(f"\n{YELLOW}[PHASE 1+2] FORWARD + JUDGE - Skipped (skip_forward=True){RESET}")
                result['judge']['prediction'] = 'SKIPPED'
                result['forward']['execution_success'] = False
                if not (self.camera_manager and getattr(self.camera_manager, "is_connected", False)):
                    if self.camera is None:
                        ok = self.initialize_camera()
                        if not ok:
                            print(f"  {YELLOW}[skip_forward] camera busy — waiting 3s and retrying...{RESET}")
                            time.sleep(3.0)
                            self.initialize_camera()
                        time.sleep(0.3)
                self.initial_image = self.capture_frame()
                if self.initial_image is not None:
                    self.initial_image_resolution = (self.initial_image.shape[1], self.initial_image.shape[0])
                    initial_path = Path(forward_dir) / "initial_state.jpg"
                    cv2.imwrite(str(initial_path), self.initial_image)
                    self.forward_initial_image_path = str(initial_path)
                    print(f"  Initial image captured for reset: {initial_path}")
                print(f"\n  [skip_forward] Detecting initial positions for reset...")
                try:
                    self.detected_positions = self.run_detection(
                        queries=objects,
                        timeout=detection_timeout,
                        visualize=visualize_detection,
                    )
                except Exception as _det_e:
                    print(f"  {RED}[skip_forward] detection failed: {_det_e}{RESET}")
                    self.detected_positions = {}
                result['forward']['positions'] = self.detected_positions
                if self.first_episode_positions is None and self.detected_positions:
                    import copy
                    self.first_episode_positions = copy.deepcopy(self.detected_positions)
                    print(f"  {GREEN}[First Episode] Initial positions saved for 'original' reset mode{RESET}")
                    if hasattr(self, 'session_dir') and self.session_dir:
                        fp_path = Path(self.session_dir) / "first_episode_positions.json"
                        with open(fp_path, 'w', encoding='utf-8') as f:
                            json.dump(self.first_episode_positions, f, indent=2, ensure_ascii=False)
                forward_log_path = forward_logger.stop()
                print(f"\n  Forward log saved to: {forward_log_path}")
            else:
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
                    print(f"\n{YELLOW}" + self._log("Capturing image for VLM (no detection)...", step="Step 1/6") + f"{RESET}")

                    # 카메라 초기화
                    if not self.camera and not (self.camera_manager and self.camera_manager.is_connected):
                        if not self.initialize_camera():
                            print(f"{RED}[Error] Camera initialization failed{RESET}")
                            return result
                    # camera_manager 연결됐지만 self.camera가 None이면 꺼내서 설정
                    if not self.camera and self.camera_manager and self.camera_manager.is_connected:
                        pc = self._get_pipeline_camera()
                        pc.camera_manager = self.camera_manager
                        cam = pc.get_realsense()
                        if cam is not None:
                            self.camera = cam

                    self.initial_image = self.capture_frame()

                    # 캡처 실패 시 카메라 재초기화 후 재시도
                    if self.initial_image is None:
                        print(f"  {YELLOW}[Warning] Capture failed, reinitializing camera...{RESET}")
                        pc = self._get_pipeline_camera()
                        pc.camera_manager = self.camera_manager
                        if pc.force_recovery():
                            self.camera = pc.camera
                            self.camera_manager = pc.camera_manager
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
                        print(f"\n{YELLOW}" + self._log(f"Detection only (reusing cached code, T3 skipped)...", step="Step 3/6") + f"{RESET}")
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
                        print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, multi-turn)...", step="Step 3/6") + f"{RESET}")
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
                    print(f"\n{YELLOW}" + self._log(f"Detecting objects: {objects}", step="Step 1/6") + f"{RESET}")
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

                    workspace = BaseWorkspace()
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
                    print(f"\n{YELLOW}" + self._log("Capturing initial state...", step="Step 2/6") + f"{RESET}")
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
                        print(f"\n{YELLOW}" + self._log(f"Reusing cached code (single-turn, T3 skipped)...", step="Step 3/6") + f"{RESET}")
                        print(f"  {GREEN}[CodeReuse] Using cached code (keys matched){RESET}")
                    else:
                        if self.cached_forward_code is not None:
                            missing = set(self.cached_forward_keys) - set(self.detected_positions.keys())
                            print(f"  {YELLOW}[CodeReuse] Key mismatch ({missing}), regenerating{RESET}")
                            self.cached_forward_code = None
                            self.cached_forward_keys = []
                        print(f"\n{YELLOW}" + self._log(f"Generating forward code via LLM ({self.llm_model}, single-turn)...", step="Step 3/6") + f"{RESET}")
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

                    # seed_01_setup 즉시 저장 (forward detection 직후)
                    session_dir_path = Path(save_dir).parent if not use_timestamp_subdir else Path(result_dir)
                    seed1_dir = session_dir_path / "seed_01_setup"
                    seed1_dir.mkdir(parents=True, exist_ok=True)
                    with open(str(seed1_dir / "seed_positions.json"), 'w') as f:
                        json.dump({"positions": self.first_episode_positions}, f, indent=2, default=str)
                    print(f"  [SeedGen] seed_01 (initial) saved: {seed1_dir / 'seed_positions.json'}")

                # [즉시 저장] Generated code
                code_path = Path(forward_dir) / "generated_code.py"
                code_path.write_text(self.generated_code)
                print(f"  Generated code saved: {code_path}")

                # [즉시 저장] Multi-turn info + visualizations (if available)
                if self.multi_turn and self.multi_turn_info:
                    from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
                    save_multi_turn_info(forward_dir, self.multi_turn_info, phase="forward")
                    save_turn_visualizations(forward_dir, self.multi_turn_info, self.initial_image, phase="forward")

                print("\n" + "-" * 40)
                print("Generated Forward Code (preview):")
                print("-" * 40)
                code_preview = self.generated_code[:600]
                if len(self.generated_code) > 600:
                    code_preview += "\n... (truncated)"
                print(code_preview)
                print("-" * 40)

                # Step 4: Code Verification (LLM 기반 코드 검증)
                # 캐시된 코드를 재사용하는 경우 검증 스킵 (이미 이전에 검증됨)
                code_was_cached = (self.cached_forward_code is not None
                                   and self.generated_code == self.cached_forward_code)
                if code_was_cached:
                    print(f"\n{YELLOW}" + self._log("Skipping verification (cached code, already verified)...", step="Step 4/6", tag="Verify") + f"{RESET}")
                else:
                    print(f"\n{YELLOW}" + self._log(f"Verifying generated code via LLM ({self.llm_model})...", step="Step 4/6", tag="Verify") + f"{RESET}")
                    from verification import verify_generated_code

                    max_verification_retries = 2
                    for verify_attempt in range(1, max_verification_retries + 1):
                        passed, reason = verify_generated_code(
                            instruction=instruction,
                            generated_code=self.generated_code,
                            object_positions=self.detected_positions,
                            llm_model=self.llm_model,
                        )

                        if passed:
                            print(f"  {GREEN}[Verify] PASS{RESET}")
                            break
                        else:
                            print(f"  {RED}[Verify] FAIL (attempt {verify_attempt}/{max_verification_retries}): {reason}{RESET}")

                            if verify_attempt < max_verification_retries:
                                # 코드 재생성
                                print(f"  {YELLOW}[Verify] Regenerating code...{RESET}")
                                if self.multi_turn:
                                    self.generated_code = self.generate_forward_code(
                                        instruction, self.detected_positions,
                                        image_path=self.forward_initial_image_path,
                                    )
                                else:
                                    self.generated_code = self.generate_forward_code(
                                        instruction, self.detected_positions,
                                    )
                                result['forward']['code'] = self.generated_code

                                # 재생성된 코드 저장
                                code_path = Path(forward_dir) / "generated_code.py"
                                code_path.write_text(self.generated_code)
                                print(f"  {YELLOW}[Verify] Regenerated code saved: {code_path}{RESET}")
                            else:
                                # 최대 재시도 도달 — 현재 코드로 진행
                                print(f"  {YELLOW}[Verify] Max retries reached, proceeding with current code{RESET}")

                # Step 5: Forward 코드 실행
                print(f"\n{YELLOW}" + self._log(f"Executing forward code on Robot {self.robot_id}...", step="Step 5/6") + f"{RESET}")

                # 레코딩 모드: 에피소드 시작
                if self.record_dataset:
                    self._start_episode_recording(task=instruction)

                # gRPC mode: drop any stale selection_ids the client may still
                # be tracking from a previous (uncommitted) episode.
                _grpc_cli = getattr(self, "_skill_planner_grpc_client", None)
                if _grpc_cli is not None:
                    try:
                        _grpc_cli.reset_pending()
                    except Exception:
                        pass

                import builtins
                builtins._current_execution_dir = forward_dir
                builtins._scene_summary = self.multi_turn_info.get("turn0_response", "") if self.multi_turn_info else ""

                # Method 3: the previous episode's demo ingestion ran in the
                # background during codegen — block here so the new demo is in
                # the vector DB before this rollout's IG·AC selection.
                self._await_demo_ingest()

                # Ensure skills exist before phase-gating so the context manager
                # can actually detach subsystems when enabled_forward=false.
                self._get_task_runner()
                with self._phase_gate("forward"):
                    forward_success = self.execute_code(self.generated_code, self.detected_positions)
                result['forward']['execution_success'] = forward_success

                if forward_success:
                    print(f"  {GREEN}Forward execution SUCCESS{RESET}")
                else:
                    print(f"  {RED}Forward execution FAILED{RESET}")

                # VLM pixel move 시각화 (pixel 좌표로 직접 이동한 경우)
                try:
                    from skills.skills_lerobot import LeRobotSkills
                    if (LeRobotSkills._last_instance
                            and hasattr(LeRobotSkills._last_instance, 'pixel_move_log')
                            and LeRobotSkills._last_instance.pixel_move_log):
                        grasp_img_path = str(Path(forward_dir) / "turn2_grasp_points.jpg")
                        if os.path.isfile(grasp_img_path):
                            self._visualize_pixel_moves(
                                grasp_img_path,
                                LeRobotSkills._last_instance.pixel_move_log,
                                str(Path(forward_dir) / "pixel_moves_overlay.jpg"),
                            )
                except Exception as e:
                    print(f"  Warning: pixel move visualization failed: {e}")

                # Update llm_cost with detect_objects token usage
                try:
                    runner = self._get_task_runner()
                    from pipeline.save_logs import update_llm_cost_with_detect_usage
                    update_llm_cost_with_detect_usage(forward_dir, runner.skills)
                except Exception as e:
                    print(f"  Warning: detect_objects cost merge failed: {e}")

                # Step 5: Context 저장
                print(f"\n{YELLOW}" + self._log("Saving execution context...", step="Step 6/6") + f"{RESET}")
                from pipeline.save_logs import save_execution_context as _save_ec
                _save_ec(forward_dir, instruction, self.detected_positions,
                         self.generated_code, forward_success, robot_id=self.robot_id)

                # Phase2 candidate top-view 오버레이 — forward 종료 직후,
                # dataset 저장 전(execution_context 등 로깅 시점)에 이번 episode
                # 의 plan_and_select dump 를 server 에서 가져와 forward_dir 에
                # skill{N}_{skill}.png 로 저장. phase1 이면 dump_refs 가 비어
                # no-op.
                try:
                    self._dump_phase2_candidate_overlays(forward_dir)
                except Exception as _e:
                    print(f"  Warning: phase2 candidate overlay 실패: {_e}")

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

                # Save batch_info early (before judge) so resume can detect this episode
                batch_idx = getattr(self, '_current_batch_index', 0)
                slot = getattr(self, '_current_slot', 0)
                episode_root = str(Path(forward_dir).parent)
                batch_info = {
                    "batch_seed_index": batch_idx + 1,
                    "slot": slot,
                    "judge": "PENDING",
                }
                bi_path = Path(episode_root) / "batch_info.json"
                bi_path.parent.mkdir(parents=True, exist_ok=True)
                with open(bi_path, 'w') as f:
                    json.dump(batch_info, f, indent=2)

                # Step 2: Judge 실행
                judge_prediction = "UNCERTAIN"
                print(f"\n{YELLOW}" + self._log(f"Running VLM Judge ({self.judge_model})...", step="Step 2/3", tag="Judge") + f"{RESET}")
                if self.initial_image is not None and self.final_image is not None:
                    image_resolution = self.final_image_resolution or self.initial_image_resolution
                    judge_result = self._run_forward_judge(
                        instruction=instruction,
                        initial_image=self.initial_image,
                        final_image=self.final_image,
                        object_positions=self.detected_positions,
                        executed_code=self.generated_code,
                        image_resolution=image_resolution,
                    )
                    result['judge'] = judge_result

                    judge_prediction = judge_result.get('prediction', 'UNCERTAIN')
                    reasoning = judge_result.get('reasoning', '')

                    pred_color = GREEN if judge_prediction == "TRUE" else RED if judge_prediction == "FALSE" else YELLOW
                    print(f"  Prediction: {pred_color}{judge_prediction}{RESET}")
                    print(f"  Reasoning: {reasoning[:100]}...")
                    # Judge 비용을 llm_cost.json에 추가
                    try:
                        from judge.vlm import _call_gemini_vlm
                        judge_usage = getattr(_call_gemini_vlm, '_last_usage', None)
                        if judge_usage:
                            cost_path = Path(forward_dir) / "llm_cost.json"
                            if cost_path.exists():
                                with open(cost_path) as f:
                                    cost_data = json.load(f)
                            else:
                                cost_data = {"phase": "forward"}
                            cost_data["judge"] = {
                                "model": judge_usage.get("model", self.judge_model),
                                "inference_time_s": judge_usage.get("inference_time_s", 0),
                                "input_tokens": judge_usage.get("in", 0),
                                "output_tokens": judge_usage.get("out", 0),
                                "total_tokens": judge_usage.get("total", 0),
                            }
                            # total에 judge 비용 합산
                            if "total" in cost_data:
                                cost_data["total"]["inference_time_s"] = round(
                                    cost_data["total"]["inference_time_s"] + judge_usage.get("inference_time_s", 0), 2)
                                cost_data["total"]["input_tokens"] += judge_usage.get("in", 0)
                                cost_data["total"]["output_tokens"] += judge_usage.get("out", 0)
                                cost_data["total"]["total_tokens"] += judge_usage.get("total", 0)
                            with open(cost_path, 'w') as f:
                                json.dump(cost_data, f, indent=2)
                    except Exception:
                        pass
                else:
                    print(f"  {YELLOW}Skipped (missing images){RESET}")

                # 레코딩 모드: Judge 결과에 따라 에피소드 저장/폐기
                # - TRUE      : 명확히 성공 → 저장
                # - UNCERTAIN : 판단 불가 (VLM 503/타임아웃 등 API 실패 포함) → 폐기
                # - FALSE     : 명확히 실패 → 폐기
                # TRUE 로 명확히 검증된 에피소드만 데이터셋에 남긴다. (UNCERTAIN
                # 은 더 이상 보존하지 않음 — subgoal buffer 와 동일한 TRUE-only 기준.)
                should_discard = judge_prediction != "TRUE"

                # Buffer-aware subgoal: grow + persist the per-skill buffer.
                # Episodes judged FALSE/UNCERTAIN discard their staged subgoals,
                # so the buffer mirrors TRUE episodes only and survives an
                # interrupted multi-episode run.
                # NOTE: Phase1 state seeding is INDEPENDENT of LeRobot dataset
                # recording — this flush must run every episode even when
                # `record_dataset` is disabled. Otherwise `_pending` accumulates
                # across episodes (pending=1,2,3,4 → 5,6,7,8 …), the buffer
                # stays N=0, and buffer-aware selection never activates.
                _subgoal_sel = getattr(self, "_subgoal_selector", None)
                if _subgoal_sel is not None:
                    try:
                        if not should_discard and judge_prediction == "TRUE":
                            # flush 시 episode_id 를 stamp 한다 — 에피소드를 삭제·
                            # 재취득(resume)할 때 buffer 를 episode 단위로 정리·
                            # reconcile 할 수 있도록 (episode lifecycle).
                            from method3.episode_lifecycle import episode_id as _mk_ep_id
                            _ep_n = getattr(self, "current_episode", None)
                            _ep_id = _mk_ep_id(_ep_n) if _ep_n else ""
                            _subgoal_sel.flush_episode(episode_id=_ep_id)
                        else:
                            _subgoal_sel.discard_episode()
                    except Exception as _e:
                        print(f"[Perturbation] subgoal buffer flush skipped: {_e}")

                episode_df = None
                if self.record_dataset:
                    # _end_episode_recording이 save_episode 전에 buffer snapshot을 떠서 반환
                    episode_df = self._end_episode_recording(discard=should_discard)

                    # Method 3: on a strict TRUE judge, grow the vector DB from
                    # the just-saved forward demo per-timestep (key_t, value_t).
                    # Runs in the background so it overlaps the next episode's
                    # codegen; the next rollout awaits it. Local mode encodes
                    # here; gRPC mode streams the episode to the H100 server.
                    #
                    # Phase2 cycle 에서는 IngestEpisode (= 전체 demo frame 의
                    # frame-level descriptor 누적) 를 *호출하지 않는다*. Phase2 의
                    # vector DB 성장은 server 의 useful-OOD accept_to_buffer 가
                    # 정식 경로 (skill-unit DCT, arm-only) — IngestEpisode 의
                    # frame-level/6축 entry 가 섞이면 z-space·차원이 깨진다.
                    _is_phase2 = str(getattr(self, "method3_phase", "")).lower() == "phase2"
                    if (not should_discard and judge_prediction == "TRUE"
                            and not _is_phase2
                            and (
                                getattr(self, "_skill_planner_selector", None) is not None
                                or getattr(self, "_skill_planner_grpc_client", None) is not None
                            )):
                        self._start_demo_ingest_async()

                    if not should_discard and episode_df is not None:
                        # Skill recording 시각화 저장 (성공한 에피소드만)
                        # snapshot이 저장 직전 buffer에서 캡처되므로 parquet footer 미완성 문제 없음
                        try:
                            from record_dataset.visualize_skills import generate_skill_visualizations
                            saved_viz = generate_skill_visualizations(
                                dataframe=episode_df,
                                save_dir=forward_dir,
                                episode_index=None,  # snapshot은 단일 episode만 포함
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
                        # 에피소드별 forward judge 시각화를 세션 공통
                        # judge_results/ 폴더에 즉시 누적 복사 (reset 이전).
                        self._collect_judge_result(forward_dir)
                # 코드 캐시 갱신: 실행 성공 + Judge!=FALSE이면 캐싱
                # 한 번이라도 TRUE가 나온 코드는 유지 (Judge=FALSE로 무효화하지 않음)
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
                elif not forward_success:
                    # 실행 자체가 실패한 경우만 캐시 무효화 (코드 자체의 문제)
                    # Judge=FALSE는 detection/환경 문제일 수 있으므로 이전 성공 코드 유지
                    if self.cached_forward_code is not None:
                        print(f"  {YELLOW}[CodeReuse] Cache invalidated (execution failed){RESET}")
                    self.cached_forward_code = None
                    self.cached_forward_keys = []
                    self._cached_point_labels = None
                elif judge_pred == 'FALSE' and self.cached_forward_code is not None:
                    print(f"  {YELLOW}[CodeReuse] Judge=FALSE but keeping cached code (previously validated){RESET}")

                # Forward 로깅 종료
                forward_log_path = forward_logger.stop()
                print(f"\n  Forward log saved to: {forward_log_path}")

                # Post-judge 콜백 (batch_info 저장 등, reset 전에 실행)
                if post_judge_callback is not None:
                    post_judge_callback(result)

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

                # Reset target 결정 (콜백 전 — 현재 seed 사용, 콜백 후 갱신)
                if reset_target_positions is not None:
                    reset_original_positions = reset_target_positions
                    print(f"  Reset target: seed positions")
                elif self.first_episode_positions is not None:
                    reset_original_positions = self.first_episode_positions
                    print(f"  Reset target: first episode positions")
                else:
                    reset_original_positions = self.detected_positions
                    print(f"  Reset target: current detected positions")

                # 디버그: reset target 상세 출력
                for _name, _info in reset_original_positions.items():
                    _pos = _info.get("position") if isinstance(_info, dict) else _info
                    if _pos and len(_pos) >= 3:
                        print(f"    {_name}: [{_pos[0]:.4f}, {_pos[1]:.4f}, {_pos[2]:.4f}]")

                # Step 1 & 2: Reset 코드 생성
                multi_turn_str = "multi-turn VLM" if self.multi_turn else "single-turn"
                print(f"\n{YELLOW}" + self._log(f"Generating reset code ({multi_turn_str})...", step="Step 1/4") + f"{RESET}")
                try:

                    # Multi-turn 모드: 코드 생성 전에 current_state 이미지 캡처
                    reset_current_state_image_path = None
                    if self.multi_turn:
                        print(f"  [MultiTurn] Capturing current state for VLM...")
                        # Forward 후 카메라가 shutdown된 상태이므로 강제 재초기화
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

                    # Reset 코드 재사용 또는 새로 생성
                    if self.cached_reset_code is not None:
                        # 캐싱된 dict 참조 코드 재사용: 검출(T0~T2)만 수행
                        print(f"  {GREEN}[CodeReuse] Using cached reset code{RESET}")
                        # 검출로 current_positions 획득
                        self.generate_forward_code(
                            instruction, {},
                            image_path=reset_current_state_image_path,
                            skip_codegen=True,
                            canonical_labels=list(reset_original_positions.keys()),
                        )
                        current_positions = self.detected_positions
                        target_positions = reset_original_positions
                        reset_code = self.cached_reset_code
                        # 검출 결과를 reset_multi_turn_info에도 저장 (시각화/로그용)
                        self.reset_multi_turn_info = getattr(self, 'multi_turn_info', None)
                    else:
                        # LLM으로 새로 생성
                        reset_code, _, current_positions, _ = self.generate_reset_code(
                            original_instruction=instruction,
                            original_positions=reset_original_positions,
                            forward_spec=self.generated_spec,
                            forward_code=self.generated_code,
                            detection_timeout=detection_timeout,
                            visualize_detection=visualize_detection,
                            current_state_image_path=reset_current_state_image_path,
                        )
                        # target은 generate_reset_code의 반환값이 아닌
                        # 콜백으로 교체된 reset_original_positions를 직접 사용
                        target_positions = reset_original_positions

                    # 검출 완료 후 콜백: 실제 current_positions로 seed 생성.
                    # round_robin 에선 매 episode 마다 next seed 위치를 swap 함.
                    if pre_reset_callback is not None:
                        new_target = pre_reset_callback(current_positions=current_positions)
                        if new_target is not None:
                            reset_original_positions = new_target
                            target_positions = new_target
                            # NOTE: 이 함수 스코프엔 CYAN/RESET 만 정의돼 있음 (line 2577~).
                            # DIM 등 추가 상수는 정의 안 돼 있어 hardcoded ANSI 사용.
                            _DIM = "\033[2m"
                            print(f"\n  {CYAN}[Reset target UPDATED by pre_reset_callback]{RESET}")
                            for _name, _info in new_target.items():
                                _pos = _info.get("position") if isinstance(_info, dict) else _info
                                if _pos and len(_pos) >= 3:
                                    print(f"    {_name}: [{_pos[0]:.4f}, {_pos[1]:.4f}, {_pos[2]:.4f}]"
                                          f"  {_DIM}(actual reset 목적지){RESET}")

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

                    # LLM 비용 통계 저장 (reset)
                    try:
                        from code_gen_lerobot.reset_execution.code_gen import lerobot_reset_code_gen_multi_turn
                        reset_cost = getattr(lerobot_reset_code_gen_multi_turn, '_last_llm_cost', None)
                        if reset_cost:
                            cost_path = Path(reset_dir) / "llm_cost.json"
                            with open(cost_path, 'w') as f:
                                json.dump({"phase": "reset", **reset_cost}, f, indent=2)
                            print(f"  LLM cost saved: {cost_path}")
                    except Exception:
                        pass

                    # [즉시 저장] Reset multi-turn VLM 데이터
                    reset_mt = getattr(self, 'reset_multi_turn_info', None)
                    if reset_mt:
                        from pipeline.save_logs import save_multi_turn_info, save_turn_visualizations
                        save_multi_turn_info(reset_dir, reset_mt, phase="reset")
                        save_turn_visualizations(reset_dir, reset_mt, self.reset_initial_image, phase="reset")

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
                        # [즉시 저장] Reset initial 이미지 = forward 시작 전 이미지 (되돌려야 할 상태)
                        reset_initial_path = Path(reset_dir) / "initial_state.jpg"
                        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
                            import shutil
                            shutil.copy2(self.forward_initial_image_path, str(reset_initial_path))
                        else:
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

                    # Step 3: Reset 코드 실행 (recording 포함)
                    print(f"\n{YELLOW}" + self._log("Executing reset code...", step="Step 3/4") + f"{RESET}")
                    self.reset_code = reset_code

                    # Reset recording: forward recorder → reset recorder로 교체
                    if self.record_dataset and self.reset_dataset_recorder:
                        self._start_reset_episode_recording(target_positions)

                    import builtins
                    builtins._current_execution_dir = reset_dir
                    reset_mt = getattr(self, 'reset_multi_turn_info', None)
                    builtins._scene_summary = reset_mt.get("turn0_response", "") if reset_mt else ""

                    # Phase-gated: detach systems whose enabled_reset=false.
                    with self._phase_gate("reset"):
                        reset_success = self.execute_code(reset_code, current_positions, extra_globals={
                            "current_positions": current_positions,
                            "target_positions": target_positions,
                        })
                    result['reset']['execution_success'] = reset_success

                    # Reset recording 종료
                    if self.record_dataset and self.reset_dataset_recorder:
                        reset_judge_pred = result.get('reset_judge', {}).get('prediction', 'TRUE')
                        should_discard = not reset_success or reset_judge_pred == "FALSE"
                        self._end_reset_episode_recording(discard=should_discard)

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
                    reset_image_resolution = self.reset_final_resolution or self.reset_initial_resolution
                    reset_judge_result = self._run_reset_judge(
                        reset_mode="original",
                        current_positions=current_positions,
                        target_positions=target_positions,
                        initial_image=self.reset_initial_image,
                        final_image=self.reset_final_image,
                        executed_code=reset_code,
                        original_instruction=instruction,
                        image_resolution=reset_image_resolution,
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

                    # Reset 코드를 캐싱 (dict 참조 방식 + 로컬 재정의 없음)
                    reset_code_text = result['reset'].get('code', '')
                    if result['reset']['execution_success'] and reset_code_text:
                        # 주석(#)이 아닌 실제 코드 라인에서 dict 참조 확인
                        code_lines = [ln.strip() for ln in reset_code_text.splitlines()
                                      if ln.strip() and not ln.strip().startswith('#')]
                        has_cur_ref = any('current_positions[' in ln for ln in code_lines)
                        has_tgt_ref = any('target_positions[' in ln for ln in code_lines)
                        # 로컬 재정의가 있으면 캐시 거부 (exec_globals shadow 방지)
                        has_cur_redef = any(ln.startswith('current_positions') and '=' in ln and '{' in ln
                                           for ln in code_lines)
                        has_tgt_redef = any(ln.startswith('target_positions') and '=' in ln and '{' in ln
                                           for ln in code_lines)
                        if has_cur_ref and has_tgt_ref and not has_cur_redef and not has_tgt_redef:
                            self.cached_reset_code = reset_code_text
                        elif has_cur_redef or has_tgt_redef:
                            print(f"  {YELLOW}[Cache] Reset code has hardcoded position defs — not caching{RESET}")

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

    def _visualize_pixel_moves(
        self,
        base_image_path: str,
        pixel_move_log: list,
        save_path: str,
    ) -> None:
        """VLM이 지정한 pixel 위치들을 grasp_points 이미지 위에 시각화.

        Args:
            base_image_path: turn2_grasp_points.jpg 경로 (이미 bbox+grasp가 그려진 이미지)
            pixel_move_log: LeRobotSkills.pixel_move_log 리스트
                           [{"pixel": [u, v], "target_name": str, "skill_description": str}, ...]
            save_path: 저장 경로
        """
        if not pixel_move_log:
            return

        image = cv2.imread(base_image_path)
        if image is None:
            print(f"  [Visualize] Cannot read base image: {base_image_path}")
            return

        img_h, img_w = image.shape[:2]

        for i, entry in enumerate(pixel_move_log):
            px, py = entry["pixel"]
            name = entry.get("target_name", "")
            desc = entry.get("skill_description", "")

            # 마커: 시안 다이아몬드 (기존 grasp point와 구별)
            color = (255, 255, 0)  # cyan (BGR)
            pts = np.array([
                [px, py - 6], [px + 6, py], [px, py + 6], [px - 6, py]
            ], dtype=np.int32)
            cv2.polylines(image, [pts], True, color, 2)
            cv2.circle(image, (px, py), 2, color, -1)

            # 라벨
            label = name if name else desc
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                text_x = min(px + 10, img_w - tw - 5)
                text_y = max(py - 5, th + 5)
                cv2.rectangle(image, (text_x - 2, text_y - th - 2),
                              (text_x + tw + 2, text_y + 2), color, -1)
                cv2.putText(image, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.imwrite(save_path, image)
        print(f"  Pixel move visualization saved: {save_path}")

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

    def _collect_judge_result(self, forward_dir: str) -> None:
        """에피소드별 forward judge 시각화(``judge_result.jpg``)를 세션 공통
        ``<session_dir>/judge_results/`` 폴더에 ``judge_result_ep{NN}.jpg`` 로
        누적 복사한다.

        judge 결과 이미지가 forward 폴더에 저장되는 **즉시**(= reset 단계
        이전) 복사하므로, run 이 중간에 끊기거나 reset 에서 실패해도 그
        시점까지의 에피소드 결과가 한 폴더에 그대로 쌓여 보존된다. 나중에
        ``judge_results/`` 폴더만 열면 전 에피소드 judge 결과를 볼 수 있다.

        파이프라인을 절대 중단시키지 않도록 모든 예외를 삼킨다. 원본
        ``judge_result.jpg`` 가 없으면(예: record_dataset 미사용으로 judge
        시각화 미저장) 조용히 건너뛴다.
        """
        try:
            import shutil
            src = Path(forward_dir) / "judge_result.jpg"
            if not src.exists():
                return
            session_dir = getattr(self, "_session_dir", None)
            session_dir = (
                Path(session_dir) if session_dir
                else Path(forward_dir).parent.parent
            )
            ep = getattr(self, "current_episode", None)
            if ep is not None:
                ep_tag = f"ep{int(ep):02d}"
            else:
                # 폴백: episode_NN 디렉터리명에서 추출
                ep_tag = Path(forward_dir).parent.name.replace("episode_", "ep")
            out_dir = session_dir / "judge_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"judge_result_{ep_tag}.jpg"
            shutil.copy2(src, dst)
            print(f"  [judge_results] collected → judge_results/{dst.name}")
        except Exception as _e:
            print(f"  [judge_results] collect skipped: {_e}")

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

    def _generate_seed_positions(self, session_dir: str, seed_index: int, current_positions: Dict = None) -> Optional[Dict]:
        """
        새 seed 위치 생성: 랜덤 위치 생성 → IK dry_run 검증.

        first_episode_positions를 기반으로 랜덤 위치를 생성합니다.
        first_episode_positions는 변경하지 않습니다.

        Args:
            session_dir: 세션 디렉토리
            seed_index: 시드 인덱스
            current_positions: 실제 검출된 현재 위치 (dry-run용).
                              None이면 candidate를 current로 사용 (fallback).

        Returns:
            성공 시 새 positions dict, 실패 시 None
        """
        from code_gen_lerobot.reset_execution.workspace import (
            generate_random_positions, classify_objects, ResetWorkspace,
        )

        if self.first_episode_positions is None:
            print("  [SeedGen] No first_episode_positions, cannot generate")
            return None

        save_dir = str(Path(session_dir) / f"seed_{seed_index+1:02d}_setup")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # ── User-provided seed 우선 ──────────────────────────────────────────
        # seed_NN_setup/seed_positions.json 이 이미 존재하면 그것을 그대로 사용
        # (수동 편집 또는 이전 run 에서 복사한 시드). random 자동생성 skip.
        # resume 경로(line 4292-4299)와 동일한 파일 포맷을 공유.
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        existing_path = Path(save_dir) / "seed_positions.json"
        if existing_path.exists():
            try:
                with open(existing_path) as f:
                    payload = json.load(f)
                user_positions = payload.get("positions")
                if user_positions:
                    print(f"  {GREEN}[SeedGen] User-provided seed found: {existing_path}{RESET}")
                    print(f"  [SeedGen] Skipping random generation, using provided positions")
                    for _name, _info in user_positions.items():
                        _pos = _info.get("position") if isinstance(_info, dict) else _info
                        if _pos and len(_pos) >= 3:
                            print(f"    {_name}: [{_pos[0]:.4f}, {_pos[1]:.4f}, {_pos[2]:.4f}]")
                    return user_positions
            except Exception as e:
                print(f"  {YELLOW}[SeedGen] Failed to load {existing_path}: {e}; falling back to random gen{RESET}")

        grippable, obstacles = classify_objects(self.first_episode_positions)

        # pix2robot 로드 — 새 Charuco 우선, 폴백으로 기존 homography
        pix2robot = None
        charuco_path = (
            Path(__file__).parent / "robot_configs" / "charuco_calibration"
            / f"robot{self.robot_id}_cam2robot.npz"
        )
        legacy_path = (
            Path(__file__).parent / "robot_configs" / "pix2robot_matrices"
            / f"robot{self.robot_id}_pix2robot_data.npz"
        )
        if charuco_path.exists():
            try:
                from pix2robot_charuco_calibrator import Pix2RobotCharuco
                pix2robot = Pix2RobotCharuco(robot_id=self.robot_id)
            except Exception:
                pix2robot = None
        if pix2robot is None and legacy_path.exists():
            try:
                from pix2robot_calibrator import Pix2RobotCalibrator
                pix2robot = Pix2RobotCalibrator(robot_id=self.robot_id)
                if not pix2robot.load(str(legacy_path)):
                    pix2robot = None
            except Exception:
                pix2robot = None

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

        # 과거 시드 위치: 같은 객체끼리만 겹침 비교하도록 _pseed 키 사용
        # first_episode_positions도 과거 seed (seed_1)이므로 포함
        all_initial = {}
        for name, info in self.first_episode_positions.items():
            all_initial[f"{name}_pseed_init"] = info
        for i, prev_positions in enumerate(self._all_previous_seed_positions):
            for name, info in prev_positions.items():
                all_initial[f"{name}_pseed{i}"] = info

        # Free state EE 주변 제외 영역 계산 (충돌 방지)
        FREE_STATE_EXCLUSION_RADIUS = 0.08  # 8cm
        exclusion_zones = []
        try:
            from lerobot_cap.kinematics import load_calibration_limits as _load_cl
            free_state_path = Path(__file__).parent / "robot_configs" / "free_state" / f"robot{self.robot_id}_free_state.json"
            if free_state_path.exists():
                with open(free_state_path) as f:
                    free_norm = np.array(json.load(f)["initial_state_normalized"])
                _cl = _load_cl(
                    str(Path(__file__).parent / "robot_configs" / "motor_calibration" / "so101" / f"robot{self.robot_id}_calibration.json"),
                    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                )
                free_rad = _cl.normalized_to_radians(free_norm)
                free_ee = kin_engine.get_ee_position(free_rad)
                exclusion_zones.append({
                    "center": [float(free_ee[0]), float(free_ee[1])],
                    "radius": FREE_STATE_EXCLUSION_RADIUS,
                })
                print(f"  [SeedGen] Free state exclusion: center=[{free_ee[0]:.3f}, {free_ee[1]:.3f}], radius={FREE_STATE_EXCLUSION_RADIUS}m")
        except Exception as e:
            print(f"  [SeedGen] Warning: Could not compute free state exclusion: {e}")

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
                current_positions=current_positions,
                exclusion_zones=exclusion_zones,
                resetspace=self.resetspace,
            )
            if not random_targets:
                print(f"  [SeedGen] Attempt {attempt+1}: position generation failed, retrying...")
                continue

            candidate = self._build_batch_positions(random_targets, obstacles, pix2robot=pix2robot)

            # dry_run 검증: 캐싱된 reset 코드에 실제 current_positions + candidate target 주입
            if reset_code is not None:
                print(f"  [SeedGen] Attempt {attempt+1}: dry-run validating...")
                dry_run_current = current_positions if current_positions else candidate
                if self.dry_run_code(reset_code, candidate, extra_globals={
                    "current_positions": dry_run_current,
                    "target_positions": candidate,
                }):
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

        print(f"  [SeedGen] seed_{seed_index+1} positions:")
        for name, info in new_positions.items():
            pos = info.get("position") if isinstance(info, dict) else info
            if pos and len(pos) >= 3:
                print(f"    {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        # 과거 시드 위치에 추가 (다음 시드 생성 시 겹침 방지) — grippable 만 (is_obstacle 아닌 것)
        self._all_previous_seed_positions.append(
            {name: info for name, info in new_positions.items()
             if isinstance(info, dict) and not info.get("is_obstacle")}
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
        from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image, _get_bbox_px

        # is_obstacle 플래그 기반 분류 (classify_objects 와 동일 정책).
        # raw bbox 휴리스틱 (is_grippable) 은 길쭉한 도구의 AABB 부풀림으로
        # manipulated 객체를 NOT grippable 로 잘못 떨어트리는 회귀가 있어 시각화에서도 폐기.
        def _info_is_grippable(info):
            return not bool(info.get("is_obstacle"))

        # 초기 이미지 로드
        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
            base_img = cv2.imread(self.forward_initial_image_path)
        else:
            base_img = np.zeros((480, 640, 3), dtype=np.uint8) + 60

        # workspace 시각화 베이스
        result = draw_workspace_on_image(base_img, robot_id=self.robot_id, pix2robot_calibrator=pix2robot, resetspace=self.resetspace)

        def _draw_bbox(img, center_px, bbox_px, color, thickness, label=""):
            hw, hh = bbox_px[0] // 2, bbox_px[1] // 2
            cu, cv = int(center_px[0]), int(center_px[1])
            cv2.rectangle(img, (cu - hw, cv - hh), (cu + hw, cv + hh), color, thickness)
            if label:
                cv2.putText(img, label, (cu - hw, cv - hh - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # 장애물 (빨간색) — is_obstacle=True 로 표시된 객체
        if self.first_episode_positions:
            for name, info in self.first_episode_positions.items():
                if not isinstance(info, dict):
                    continue
                if _info_is_grippable(info):
                    continue
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
                        _draw_bbox(result, px, bbox, color, 1, f"s{i+1}:{name}")
                    except Exception:
                        pass

        # 새 시드 (초록색, 굵게) — grippable만 (is_obstacle 아닌 것)
        for name, info in new_positions.items():
            if not isinstance(info, dict):
                continue
            if not _info_is_grippable(info):
                continue
            pos = info.get("position")
            bbox = _get_bbox_px(info)
            if pos and pix2robot:
                try:
                    px = pix2robot.robot_to_pixel(pos[0], pos[1])
                    _draw_bbox(result, px, bbox, (0, 255, 0), 3, f"NEW s{seed_index+1}:{name}")
                except Exception:
                    pass

        # 범례
        img_h = result.shape[0]
        cv2.putText(result, f"Seed {seed_index+1} | Past: {len(self._all_previous_seed_positions)} seeds",
                    (10, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = str(Path(save_dir) / "seed_visualization.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [SeedGen] Visualization saved: {out_path}")

        # ── 객체 종류별 포인트 시각화 ──
        self._visualize_seed_points(save_dir, new_positions, seed_index, pix2robot)

    def _visualize_seed_points(
        self,
        save_dir: str,
        new_positions: Dict,
        seed_index: int,
        pix2robot=None,
    ):
        """
        객체 종류별 색상으로 모든 시드의 위치를 점으로 시각화.

        각 객체에 고유 색상을 할당하고, 과거 시드 + 현재 시드의
        위치를 점(원)으로 표시. 현재 시드는 크게, 과거는 작게.
        """
        import cv2
        from code_gen_lerobot.reset_execution.workspace import draw_workspace_on_image

        # is_obstacle 플래그 기반 분류 (classify_objects 와 동일 정책).
        def _info_is_grippable(info):
            return not bool(info.get("is_obstacle"))

        if self.forward_initial_image_path and Path(self.forward_initial_image_path).exists():
            base_img = cv2.imread(self.forward_initial_image_path)
        else:
            base_img = np.zeros((480, 640, 3), dtype=np.uint8) + 60

        result = draw_workspace_on_image(base_img, robot_id=self.robot_id, pix2robot_calibrator=pix2robot, resetspace=self.resetspace)

        # 객체 이름 수집 (grippable만 — is_obstacle 아닌 것)
        all_obj_names = set()
        if self.first_episode_positions:
            for name, info in self.first_episode_positions.items():
                if isinstance(info, dict) and _info_is_grippable(info):
                    all_obj_names.add(name)
        for name in new_positions:
            if isinstance(new_positions[name], dict) and _info_is_grippable(new_positions[name]):
                all_obj_names.add(name)
        all_obj_names = sorted(all_obj_names)

        # 객체별 고유 색상 (BGR)
        OBJ_COLORS = [
            (0, 0, 255),    # 빨강
            (255, 0, 0),    # 파랑
            (0, 200, 0),    # 초록
            (0, 200, 255),  # 노랑
            (255, 0, 255),  # 마젠타
            (255, 200, 0),  # 시안
            (0, 128, 255),  # 주황
            (200, 0, 128),  # 보라
        ]
        obj_color_map = {name: OBJ_COLORS[i % len(OBJ_COLORS)] for i, name in enumerate(all_obj_names)}

        # 과거 시드 (작은 점 + 시드 번호)
        for seed_i, prev in enumerate(self._all_previous_seed_positions):
            for name, info in prev.items():
                if name not in obj_color_map or not isinstance(info, dict):
                    continue
                pos = info.get("position")
                if pos and pix2robot:
                    try:
                        px, py = pix2robot.robot_to_pixel(pos[0], pos[1])
                        px, py = int(px), int(py)
                        color = obj_color_map[name]
                        cv2.circle(result, (px, py), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(result, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(result, f"s{seed_i+1}", (px + 7, py + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
                    except Exception:
                        pass

        # 현재 시드 (큰 점 + 라벨) — grippable만 (is_obstacle 아닌 것)
        for name, info in new_positions.items():
            if name not in obj_color_map or not isinstance(info, dict):
                continue
            if not _info_is_grippable(info):
                continue
            pos = info.get("position")
            if pos and pix2robot:
                try:
                    px, py = pix2robot.robot_to_pixel(pos[0], pos[1])
                    px, py = int(px), int(py)
                    color = obj_color_map[name]
                    cv2.circle(result, (px, py), 9, color, -1, cv2.LINE_AA)
                    cv2.circle(result, (px, py), 9, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(result, f"s{seed_index+1}", (px + 11, py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                except Exception:
                    pass

        # 범례: 객체별 색상
        img_h = result.shape[0]
        y_offset = img_h - 15 * len(all_obj_names) - 10
        for i, name in enumerate(all_obj_names):
            color = obj_color_map[name]
            y = y_offset + i * 15
            cv2.circle(result, (15, y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(result, name, (25, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        cv2.putText(result, f"Seed {seed_index+1} | {len(self._all_previous_seed_positions)} past seeds",
                    (10, img_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = str(Path(save_dir) / "seed_visualization_point.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [SeedGen] Point visualization saved: {out_path}")

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

    def _load_resume_state(self, session_dir: str) -> Tuple[List[List[bool]], List[Optional[Dict]], List[bool]]:
        """이전 세션에서 상태 복원 (batch_info.json 기반).

        Returns:
            (batch_slots, seed_positions, batch_attempted):
                batch_slots[i] = [True/False, ...] 배치 i의 각 slot 성공 여부
                seed_positions[i] = 배치 i의 seed 위치 (없으면 None)
                batch_attempted[i] = True/False 배치 i가 한 번이라도 시도되었는지
        """
        import copy
        episodes_per_seed = max(1, self.total_episodes // self.num_random_seeds)
        batch_slots = [[False] * episodes_per_seed for _ in range(self.num_random_seeds)]
        batch_attempted = [False] * self.num_random_seeds
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # episode dir 은 chain reorg 후 phase1/ 하위에 있을 수 있다 (Step 0).
        # legacy(session 직속) + phase1/ + phase2/ 를 모두 인식하도록 통합 검색.
        _all_eps = {p.name: p for p in _iter_episode_dirs(session_dir)}

        # 1. first_episode_positions 복원 (ep_01 execution_context)
        _ep01 = _all_eps.get("episode_01")
        ctx_path = (_ep01 / "forward" / "execution_context.json") if _ep01 else None
        if ctx_path is not None and ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            self.first_episode_positions = ctx.get("object_positions", {})
            seed_positions[0] = copy.deepcopy(self.first_episode_positions)
            # forward_initial_image_path 복원 (시각화용)
            initial_img = _ep01 / "forward" / "initial_state.jpg"
            if initial_img.exists():
                self.forward_initial_image_path = str(initial_img)
            print(f"  [Resume] first_episode_positions restored from {ctx_path}")

        # 2. seed positions 복원
        for i in range(1, self.num_random_seeds):
            sp_path = Path(session_dir) / f"seed_{i+1:02d}_setup" / "seed_positions.json"
            if sp_path.exists():
                with open(sp_path) as f:
                    sp = json.load(f)
                seed_positions[i] = sp.get("positions", None)
                print(f"  [Resume] seed_{i+1} restored from {sp_path}")

        # 3. batch_info.json 기반으로 slot별 성공 여부 파악
        #    phase1/ phase2/ legacy 모두 순회 (chain reorg 후 phase1/ 하위).
        for ep_dir in sorted(_iter_episode_dirs(session_dir), key=lambda p: p.name):
            batch_info_path = ep_dir / "batch_info.json"
            if not batch_info_path.exists():
                continue
            with open(batch_info_path) as f:
                bi = json.load(f)
            # batch_seed_index (1-based) 또는 legacy batch_index (0-based) 호환
            if "batch_seed_index" in bi:
                batch_idx = bi["batch_seed_index"] - 1  # 1-based → 0-based
            else:
                batch_idx = bi.get("batch_index", -1)  # legacy 호환
            slot = bi.get("slot", -1)
            judge_pred = bi.get("judge", "")
            if 0 <= batch_idx < self.num_random_seeds and 0 <= slot < episodes_per_seed:
                batch_attempted[batch_idx] = True
                if judge_pred == "TRUE":
                    batch_slots[batch_idx][slot] = True

        # 4. cached_reset_code 복원 (이전 세션의 reset 코드를 찾아서 캐시)
        for ep_dir in sorted(_iter_episode_dirs(session_dir), key=lambda p: p.name, reverse=True):
            reset_code_path = ep_dir / "reset" / "generated_code.py"
            if reset_code_path.exists():
                code = reset_code_path.read_text().strip()
                if code:
                    self.cached_reset_code = code
                    self.cached_reset_keys = self._extract_position_keys(code)
                    print(f"  [Resume] cached_reset_code restored from {reset_code_path}")
                    break

        # 5. cached_forward_code 복원 (이전 세션의 forward 코드를 찾아서 캐시)
        for ep_dir in sorted(_iter_episode_dirs(session_dir), key=lambda p: p.name, reverse=True):
            fwd_code_path = ep_dir / "forward" / "generated_code.py"
            judge_path = ep_dir / "forward" / "judge_result.json"
            if fwd_code_path.exists() and judge_path.exists():
                with open(judge_path) as f:
                    jr = json.load(f)
                pred = jr.get("judge_result", {}).get("prediction", jr.get("prediction", ""))
                if pred == "TRUE":
                    code = fwd_code_path.read_text().strip()
                    if code:
                        self.cached_forward_code = code
                        self.cached_forward_keys = self._extract_position_keys(code)
                        # point labels 캐시 복원
                        ctx_path = ep_dir / "forward" / "execution_context.json"
                        if ctx_path.exists():
                            with open(ctx_path) as f:
                                ctx = json.load(f)
                            self._cached_point_labels = {}
                            for name, info in ctx.get("object_positions", {}).items():
                                if isinstance(info, dict) and "points" in info:
                                    self._cached_point_labels[name] = list(info["points"].keys())
                        print(f"  [Resume] cached_forward_code restored from {fwd_code_path}")
                        break

        # 6. 복원된 seed positions를 _all_previous_seed_positions에 등록
        for i, sp in enumerate(seed_positions):
            if sp is not None:
                self._all_previous_seed_positions.append(sp)

        # 요약 출력
        for i in range(self.num_random_seeds):
            done = sum(batch_slots[i])
            status = "DONE" if done >= episodes_per_seed else f"{done}/{episodes_per_seed}"
            seed_str = "loaded" if seed_positions[i] else "to generate"
            attempted_str = "attempted" if batch_attempted[i] else "new"
            print(f"    Batch {i+1}: {status} ({seed_str}, {attempted_str})")
        print(f"  [Resume] Previous seeds registered: {len(self._all_previous_seed_positions)}")
        return batch_slots, seed_positions, batch_attempted

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

            import tempfile
            tmp_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(tmp_path, current_frame)
            if self.multi_turn and not self.forward_initial_image_path:
                self.forward_initial_image_path = tmp_path

            # Reset 코드 재사용 또는 새로 생성
            if self.cached_reset_code is not None:
                print(f"  {GREEN}[CodeReuse] Using cached reset code{RESET_C}")
                # 검출로 current_positions 획득 (forward 함수 재사용, detection만 수행)
                print(f"\n{CYAN}  [Restore] Running detection to find current positions...{RESET_C}")
                self.generate_forward_code(
                    instruction, {},
                    image_path=tmp_path,
                    skip_codegen=True,
                    canonical_labels=list(target_positions.keys()),
                )
                print(f"{CYAN}  [Restore] Detection complete{RESET_C}")
                current_pos = self.detected_positions
                reset_code = self.cached_reset_code
            else:
                print(f"  Generating reset code via LLM...")
                reset_code, _, current_pos, _ = self.generate_reset_code(
                    original_instruction=instruction,
                    original_positions=target_positions,
                    current_state_image_path=tmp_path,
                )

            # Reset 코드 실행 (레코딩 임시 비활성화 + perturbation 비활성화)
            self.shutdown_camera()
            saved_record = self.record_dataset
            self.record_dataset = False
            try:
                # Phase-gated: detach systems whose enabled_reset=false.
                # Trigger lazy skill creation first so the context manager has
                # a real object to act on.
                self._get_task_runner()
                with self._phase_gate("reset"):
                    success = self.execute_code(reset_code, {}, extra_globals={
                        "current_positions": current_pos,
                        "target_positions": target_positions,
                    })
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

    # ================================================================
    # 공통 헬퍼
    # ================================================================

    def _verify_resume_layout(
        self,
        session_dir: str,
        config_path: Path,
        num_episodes: int,
        episodes_per_seed: int,
    ) -> None:
        """Resume 시 session_config 의 layout 과 현재 (NUM_EPISODES, NUM_RANDOM_SEEDS)
        가 일치하는지 검사.

        ep_idx → (seed, slot) 매핑은 `episode_idx // episodes_per_seed` 로 계산되므로
        ``episodes_per_seed`` 가 바뀌면 episode_NN 폴더와 (seed, slot) 의 1:1 매핑이
        깨진다. 같은 세션을 phase1 → phase2 누적 확장(예: 30/10 → 100/10) 하려면
        폴더 mv 마이그레이션이 선행돼야 한다. 본 함수는 그 조건을 검사하고 불일치 시
        ``scripts/migrate_session_episodes_per_seed.py`` 명령 hint 와 함께 멈춘다.
        """
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            YELLOW = "\033[93m"; RESET = "\033[0m"
            print(f"\n{YELLOW}[Resume] session_config.json 읽기 실패 ({e}); layout 검사 스킵{RESET}")
            return

        stored_episodes = int(cfg.get("num_episodes") or 0)
        stored_seeds = int(cfg.get("num_random_seeds") or 0)
        stored_per_seed = int(cfg.get("episodes_per_seed") or 0)

        # 0 = legacy session_config (해당 필드 없음) — 검사 우회
        seeds_match = stored_seeds in (0, self.num_random_seeds)
        per_seed_match = stored_per_seed in (0, episodes_per_seed)
        if seeds_match and per_seed_match:
            return

        RED = "\033[91m"; YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"
        msg_lines = [
            "",
            f"{RED}{BOLD}[Resume] session layout mismatch — (seed, slot) ↔ episode_NN 매핑이 깨짐{RESET}",
            f"{RED}  session_config.json: num_episodes={stored_episodes}, num_random_seeds={stored_seeds}, episodes_per_seed={stored_per_seed}{RESET}",
            f"{RED}  current settings   : num_episodes={num_episodes}, num_random_seeds={self.num_random_seeds}, episodes_per_seed={episodes_per_seed}{RESET}",
            "",
        ]

        # 수직 확장 케이스 — 같은 seed 개수에서 episodes_per_seed 증가
        is_vertical_extension = (
            stored_seeds == self.num_random_seeds
            and stored_per_seed > 0
            and episodes_per_seed > stored_per_seed
            and num_episodes > stored_episodes
        )
        if is_vertical_extension:
            msg_lines += [
                f"{YELLOW}같은 seed 수에서 episodes_per_seed 가 늘어난 케이스입니다{RESET}",
                f"{YELLOW}(예: phase1 → phase2 누적 확장). 폴더 ↔ (seed, slot) 매핑을{RESET}",
                f"{YELLOW}유지하려면 다음 마이그레이션 명령을 먼저 실행하세요:{RESET}",
                "",
                f"  python scripts/migrate_session_episodes_per_seed.py \\",
                f"      --session {session_dir} \\",
                f"      --old-episodes {stored_episodes} \\",
                f"      --new-episodes {num_episodes} \\",
                f"      --num-seeds {self.num_random_seeds} \\",
                f"      --apply",
                "",
                f"{YELLOW}(먼저 --apply 없이 실행해 dry-run 으로 mv 리스트 확인 권장){RESET}",
                "",
            ]
        else:
            msg_lines += [
                f"{YELLOW}NUM_EPISODES / NUM_RANDOM_SEEDS 를 session_config.json 값과 맞추거나,{RESET}",
                f"{YELLOW}새 세션을 시작하세요. 수직 확장만 마이그레이션 스크립트 사용 가능.{RESET}",
                "",
            ]
        raise RuntimeError("\n".join(msg_lines))

    def _init_session(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        save_dir: Optional[str],
        session_dir: Optional[str] = None,
    ) -> Tuple[str, int, Dict]:
        """세션 초기화: 디렉토리, 배치 계산, 결과 딕셔너리 생성.

        Returns:
            (session_dir, episodes_per_seed, all_results)
        """
        if save_dir is None:
            save_dir = "results"

        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = str(Path(save_dir) / f"session_{timestamp}")
            is_resume = False
        else:
            is_resume = True
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        self.total_episodes = num_episodes
        episodes_per_seed = max(1, num_episodes // self.num_random_seeds)

        # 세션 설정 저장 (최초 생성 시만, resume 시 덮어쓰지 않음)
        config_path = Path(session_dir) / "session_config.json"

        # resume 인데 stored layout 과 현재 (NUM_EPISODES, NUM_RANDOM_SEEDS) 가
        # 불일치하면 (seed, slot) ↔ episode_NN 매핑이 깨지므로 명확히 멈추고
        # 마이그레이션 명령을 hint 로 출력한다. phase1 → phase2 누적 확장
        # (예: 30/10 → 100/10) 도 동일 detector 를 거친다.
        if is_resume and config_path.exists():
            self._verify_resume_layout(session_dir, config_path, num_episodes, episodes_per_seed)

        if not config_path.exists():
            # schedule_mode 도 함께 기록 — cleanup_dataset_for_resume 가 round_robin
            # 의 dataset save 순서를 복원하는 데 필요 (sorted-name ≠ save-order).
            # 기본 round_robin; yaml 에 schedule_mode: seed_major 명시 시만 legacy.
            _sched_mode = "round_robin"
            _ph1_path = Path(__file__).resolve().parent / "pipeline_config" / "phase1_config.yaml"
            if _ph1_path.exists():
                try:
                    import yaml as _yaml
                    with open(_ph1_path, "r") as _f:
                        _ph1 = _yaml.safe_load(_f) or {}
                    _sched_mode = str((_ph1.get("readiness") or {}).get(
                        "schedule_mode", "round_robin"))
                except Exception:
                    pass
            session_config = {
                "num_episodes": num_episodes,
                "num_random_seeds": self.num_random_seeds,
                "episodes_per_seed": episodes_per_seed,
                "instruction": instruction,
                "objects": objects,
                "schedule_mode": _sched_mode,
            }
            with open(config_path, 'w') as f:
                json.dump(session_config, f, indent=2)
            print(f"  [Session] Config saved: {config_path} (schedule_mode={_sched_mode})")

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

        return session_dir, episodes_per_seed, all_results

    def _finalize_session(
        self,
        all_results: Dict,
        session_dir: str,
        num_episodes: int,
        instruction: str,
        objects: list,
        skip_reset: bool,
    ) -> None:
        """세션 마무리: 요약 출력, JSON 저장, 레코딩 finalize."""
        self._print_final_summary(all_results, skip_reset)

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

        # 최종 시드 시각화 (new_seed 없이 — 모든 시드 분포 확인용)
        if self._all_previous_seed_positions and self.num_random_seeds > 1:
            try:
                pix2robot = None
                charuco_path = (
                    Path(__file__).parent / "robot_configs" / "charuco_calibration"
                    / f"robot{self.robot_id}_cam2robot.npz"
                )
                legacy_path = (
                    Path(__file__).parent / "robot_configs" / "pix2robot_matrices"
                    / f"robot{self.robot_id}_pix2robot_data.npz"
                )
                if charuco_path.exists():
                    try:
                        from pix2robot_charuco_calibrator import Pix2RobotCharuco
                        pix2robot = Pix2RobotCharuco(robot_id=self.robot_id)
                    except Exception:
                        pix2robot = None
                if pix2robot is None and legacy_path.exists():
                    from pix2robot_calibrator import Pix2RobotCalibrator
                    pix2robot = Pix2RobotCalibrator(robot_id=self.robot_id)
                    if not pix2robot.load(str(legacy_path)):
                        pix2robot = None

                last_seed_idx = len(self._all_previous_seed_positions) - 1
                final_save_dir = str(Path(session_dir) / f"seed_{last_seed_idx+1:02d}_setup")
                Path(final_save_dir).mkdir(parents=True, exist_ok=True)
                # new_positions를 빈 dict로 → 초록 bbox 없음
                self._visualize_seed_positions(
                    save_dir=final_save_dir,
                    new_positions={},
                    seed_index=last_seed_idx,
                    pix2robot=pix2robot,
                )
                print(f"  [Session] Final seed visualization saved")
            except Exception as e:
                print(f"  [Session] Warning: Final seed visualization failed: {e}")

        if self.record_dataset:
            self._finalize_recording()


    # ================================================================
    # 새 세션 모드
    # ================================================================

    def run_multiple_episodes(
        self,
        num_episodes: int,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        save_dir: Optional[str] = None,
        skip_reset: bool = False,
        skip_forward: bool = False,
    ) -> Dict:
        """새 세션: 모든 에피소드를 순차 실행. 실패해도 멈추지 않고 계속 진행.

        배치 마지막 에피소드의 reset에서 다음 seed를 생성하고 그 위치로 reset하여
        다음 배치의 시작 상태를 준비한다.

        흐름 (6ep, 3seed, eps_per_seed=2):
            EP01: forward(seed0) → reset(seed0)
            EP02: forward(seed0) → seed_gen(1) → reset(seed1)   ← 배치 마지막
            EP03: forward(seed1) → reset(seed1)
            EP04: forward(seed1) → seed_gen(2) → reset(seed2)   ← 배치 마지막
            EP05: forward(seed2) → reset(seed2)
            EP06: forward(seed2) → reset(seed2)                  ← 전체 마지막
        """
        import copy

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        session_dir, episodes_per_seed, all_results = self._init_session(
            num_episodes, instruction, objects, save_dir,
        )
        # Method 3: wire the IG·AC selector now that the session folder exists,
        # so the FAISS buffer lands inside <session_dir>/preselective_buffer.
        self._setup_skill_planner_transport(session_dir)
        # Buffer-aware subgoal: remember the session dir so `_get_task_runner`
        # can bind the .npz buffer once the selector is lazily created. The
        # call here still no-ops (selector not built yet) — kept harmless.
        self._session_dir = session_dir
        self._finalize_subgoal_buffer(session_dir)
        self._setup_phase2_session(session_dir)
        # Method3 Stage 1 — per-episode Phase1 readiness hook (opt-in via
        # method3_phase1 section of recording config). When enabled, the loop
        # can early-stop on R_ready > τ or B₁_max.
        self._setup_phase1_readiness_hook()
        seed_positions: List[Optional[Dict]] = [None] * self.num_random_seeds

        # 헤더 출력
        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  MULTI-EPISODE SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        # 스케줄 결정 — readiness hook 의 schedule_mode 가 좌우 (default seed_major).
        # round_robin: seed0·1, seed1·1, ..., seedN·1, seed0·2, ... 형태로 cycle.
        from method3_integration.scheduling import schedule_iter
        _hook_cfg = getattr(getattr(self, "_phase1_readiness_hook", None),
                            "cfg", None)
        schedule_mode = getattr(_hook_cfg, "schedule_mode", "seed_major")
        is_round_robin = (schedule_mode == "round_robin")
        print(f"  Schedule: {schedule_mode}")

        # 에피소드 루프 — schedule_iter 가 (exec_idx, batch, slot, ep_num, round_last) 산출.
        for execution_idx, batch_index, slot, episode_num, is_round_last in schedule_iter(
                num_episodes, episodes_per_seed, self.num_random_seeds, schedule_mode):

            self.current_episode = episode_num
            self._current_batch_index = batch_index
            self._current_slot = slot

            # Per-episode perturbation RNG seed (no-op if perturbation disabled).
            self._seed_episode_perturbation(batch_index, slot)

            print("\n" + CYAN + "=" * 70 + RESET)
            print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Episode "
                  f"(Batch {batch_index+1}, Slot {slot})  ".center(70) + RESET)
            print(CYAN + "=" * 70 + RESET)

            episode_dir = str(_episode_dir(session_dir, episode_num, getattr(self, "method3_phase", None)))
            Path(episode_dir).parent.mkdir(parents=True, exist_ok=True)

            # Reset target & next-seed handling.
            # round_robin: 매 episode 마다 next seed 가 cycle 로 바뀜 → pre_reset_cb 항상 set.
            # seed_major:  batch 의 마지막 slot 일 때만 pre_reset_cb 설정.
            reset_target = seed_positions[batch_index]
            if is_round_robin:
                next_batch_index = (batch_index + 1) % self.num_random_seeds
                _need_cb = True
            else:
                next_batch_index = batch_index + 1
                _need_cb = is_round_last and next_batch_index < self.num_random_seeds

            pre_reset_cb = None
            if _need_cb:
                _next_idx = next_batch_index
                _sp = seed_positions
                _sd = session_dir
                def _make_next_seed(next_idx=_next_idx, sp=_sp, sd=_sd, current_positions=None):
                    if sp[0] is None and self.first_episode_positions is not None:
                        sp[0] = copy.deepcopy(self.first_episode_positions)
                        self._all_previous_seed_positions.append(sp[0])
                    if sp[next_idx] is None:
                        print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{next_idx+1}...{RESET}")
                        sp[next_idx] = self._generate_seed_positions(sd, next_idx, current_positions=current_positions)
                    return sp[next_idx]
                pre_reset_cb = _make_next_seed

            try:
                _ep_dir = episode_dir
                _bi = batch_index
                _slot = slot
                def _post_judge(res):
                    jp = res['judge'].get('prediction', 'UNCERTAIN')
                    from pipeline.save_logs import save_batch_info as _sbi
                    _sbi(_ep_dir, _bi, _slot, jp)

                result = self.run(
                    instruction=instruction, objects=objects,
                    detection_timeout=detection_timeout,
                    visualize_detection=visualize_detection,
                    save_dir=episode_dir, use_timestamp_subdir=False,
                    skip_reset=skip_reset, skip_forward=skip_forward,
                    reset_target_positions=reset_target,
                    pre_reset_callback=pre_reset_cb,
                    post_judge_callback=_post_judge,
                )

                if seed_positions[0] is None and self.first_episode_positions is not None:
                    seed_positions[0] = copy.deepcopy(self.first_episode_positions)
                    self._all_previous_seed_positions.append(seed_positions[0])

                self._update_results(all_results, result, episode_num, skip_reset)

            except Exception as e:
                print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                import traceback; traceback.print_exc()
                all_results['episodes'].append({'episode': episode_num, 'result': None, 'success': False, 'error': str(e)})

            # Hook check — round_robin: end-of-round 만. seed_major: 매 episode.
            _hook = getattr(self, "_phase1_readiness_hook", None)
            if _hook is not None and _hook.enabled:
                _do_check = is_round_last if is_round_robin else True
                if _do_check:
                    _sel = getattr(self, "_subgoal_selector", None)
                    _buf = getattr(_sel, "buffer", None) if _sel is not None else None
                    _episodes_done = execution_idx + 1
                    if _hook.should_stop(_episodes_done, _buf,
                                         episode_label=episode_num):
                        print(MAGENTA + BOLD +
                              f"  [method3:phase1] loop terminated early — "
                              f"done={_episodes_done} folder=ep{episode_num} "
                              f"(reason={_hook.stop_reason})  ".center(70) + RESET)
                        break

            time.sleep(2)

        self._finalize_session(all_results, session_dir, num_episodes, instruction, objects, skip_reset)
        return all_results

    # ================================================================
    # Resume 모드
    # ================================================================

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
        """Resume: 이전 세션의 미완료 배치만 골라서 재시도.

        1. 완료된 배치 → 스킵
        2. 시도했지만 실패한 배치 → _restore_to_seed 후 미완료 slot 재시도 (최대 3회)
        3. 미시도 배치 → _restore_to_seed 후 순차 실행 (새 세션과 동일 방식)

        흐름 예시 (이전 세션 6ep, 3seed):
            Batch 0: slot0=TRUE, slot1=TRUE  → 스킵
            Batch 1: slot0=TRUE, slot1=FALSE → restore(seed1) → slot1 재시도
            Batch 2: 미시도                   → restore(seed2) → EP05, EP06 순차 실행
        """
        import copy

        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        session_dir = str(resume_session_dir)
        session_dir, episodes_per_seed, all_results = self._init_session(
            num_episodes, instruction, objects, save_dir, session_dir=session_dir,
        )
        # Method 3: wire the IG·AC selector against the resumed session folder.
        self._setup_skill_planner_transport(session_dir)
        # Buffer-aware subgoal: remember the session dir so `_get_task_runner`
        # can bind the .npz buffer once the selector is lazily created (picks
        # up the buffer persisted by the earlier run on resume).
        self._session_dir = session_dir
        self._setup_phase2_session(session_dir)
        # Method3 Stage 1 — readiness hook on resume too (same opt-in via yaml).
        # NOTE: hook must be created BEFORE _finalize_subgoal_buffer so the
        # finalize can bind readiness_trajectory.jsonl to it.
        self._setup_phase1_readiness_hook()
        # _finalize_subgoal_buffer 호출은 cleanup_dataset_for_resume 직후로
        # 옮겼다 — finalize 안의 resume reconcile 이 `_resume_kept_true_episodes`
        # 를 보고 동작하는데, 그 값은 cleanup 이 set 한다. 순서를 반대로 하면
        # reconcile 이 항상 None 을 보고 skip 되어 stale buffer entry 가 남는다.
        batch_slots, seed_positions, batch_attempted = self._load_resume_state(session_dir)

        # 헤더 출력
        print("\n" + MAGENTA + "=" * 70 + RESET)
        print(MAGENTA + BOLD + f"  RESUME SESSION: {num_episodes} Episodes  ".center(70) + RESET)
        print(MAGENTA + "=" * 70 + RESET)
        print(f"  Instruction: {instruction}")
        print(f"  Objects: {objects}")
        if self.num_random_seeds > 1:
            print(f"  Random Seeds: {self.num_random_seeds} batches × {episodes_per_seed} episodes")
        print(f"  Resume from: {session_dir}")
        print(f"  Save Dir: {session_dir}")
        print(MAGENTA + "=" * 70 + RESET)

        # --------------------------------------------------------
        # 데이터셋 정리: 재취득 대상 에피소드를 데이터셋에서 제거
        # --------------------------------------------------------
        if self.record_dataset:
            if self.dataset_repo_id:
                try:
                    from record_dataset.cleanup import cleanup_dataset_for_resume
                    cleanup_stats = cleanup_dataset_for_resume(
                        session_dir=session_dir,
                        repo_id=self.dataset_repo_id,
                    )
                    # Dataset 카운트 변화 — BLUE (사용자 요청).
                    # ANSI inline → NameError 위험 0.
                    print(f"\n\033[94m[Cleanup] Result: {cleanup_stats['dataset_episodes_before']} → {cleanup_stats['dataset_episodes_after']} episodes\033[0m")
                    if cleanup_stats['deleted_indices']:
                        print(f"\033[94m[Cleanup] Deleted dataset indices: {cleanup_stats['deleted_indices']}\033[0m")
                    # episode lifecycle — 생존 에피소드 집합을 보관해 두면
                    # _finalize_subgoal_buffer 가 subgoal buffer 를 그 집합으로
                    # reconcile 한다 (삭제된 에피소드의 stale entry 제거).
                    self._resume_kept_true_episodes = cleanup_stats.get("kept_true_episodes")

                    # forward·reset dataset 인덱스 정합 유지: reset repo 도
                    # 같은 로직으로 trim 한다. forward 만 청소하면 reset 이
                    # stale 누적해 forward[N] ↔ reset[N] 페어 매칭이 깨진다.
                    try:
                        reset_stats = cleanup_dataset_for_resume(
                            session_dir=session_dir,
                            repo_id=self.dataset_repo_id + "_reset",
                        )
                        # Dataset 카운트 변화 — BLUE (사용자 요청).
                        print(
                            f"\033[94m[Cleanup-Reset] Result: "
                            f"{reset_stats['dataset_episodes_before']} → "
                            f"{reset_stats['dataset_episodes_after']} episodes\033[0m")
                        if reset_stats['deleted_indices']:
                            print(f"\033[94m[Cleanup-Reset] Deleted dataset indices: "
                                  f"{reset_stats['deleted_indices']}\033[0m")
                    except Exception as e:
                        print(f"\n{YELLOW}[Cleanup-Reset] Warning: "
                              f"Reset dataset cleanup failed: {e}{RESET}")
                        import traceback; traceback.print_exc()
                except Exception as e:
                    print(f"\n{YELLOW}[Cleanup] Warning: Dataset cleanup failed: {e}{RESET}")
                    import traceback; traceback.print_exc()

            # cleanup 후 레코딩 초기화 (resume=True로 append 모드)
            self._init_recording()

        # _subgoal_selector 는 _create_skills (lazy in _get_task_runner) 시점에
        # 만들어지므로 cleanup 직후 _finalize_subgoal_buffer 를 호출해도 selector
        # 가 아직 None → silent return → reconcile 로그 0개 였다. 사용자가
        # cleanup 로그 옆에서 buffer 정리를 보고 싶어하므로, selector 없이도
        # 동작하는 transient reconcile 을 직접 수행한다. 이후 첫 episode 시점에
        # lazy selector init 가 같은 .npz 를 다시 load 하므로 정합성 OK.
        self._reconcile_subgoal_buffer_on_resume(session_dir)

        # --------------------------------------------------------
        # 통합 에피소드 루프: 성공 slot 스킵, 실패/미시도 slot 실행
        # --------------------------------------------------------

        # 스케줄 결정 — resume 도 동일 schedule_mode 따른다. first_incomplete
        # (초기 restore) 와 reset_target 모두 schedule *순서* 를 따라야
        # round_robin 에서 seed 가 맞는다 — round_robin 은 episode 마다 batch 가
        # 바뀌므로 seed_major 식 "batch 0 부터 스캔" 은 첫 실행 episode 의 batch
        # 와 어긋난다 (예: 첫 미완료 episode 가 batch 11 인데 batch 0 을 고름).
        from method3_integration.scheduling import schedule_iter
        _hook_cfg = getattr(getattr(self, "_phase1_readiness_hook", None),
                            "cfg", None)
        schedule_mode = getattr(_hook_cfg, "schedule_mode", "seed_major")
        is_round_robin = (schedule_mode == "round_robin")
        print(f"  Schedule: {schedule_mode}")
        _schedule = list(schedule_iter(
            num_episodes, episodes_per_seed, self.num_random_seeds, schedule_mode))

        # 첫 미완료 episode 의 batch 찾기 → _restore_to_seed 1회.
        first_incomplete = None
        for _e_i, _b_i, _s_i, _ep_i, _rl_i in _schedule:
            if not batch_slots[_b_i][_s_i]:
                first_incomplete = _b_i
                break

        if first_incomplete is None:
            # 설정 불일치 감지: session_config.json 또는 실제 데이터에서 원래 설정 확인
            config_path = Path(session_dir) / "session_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    orig = json.load(f)
                orig_eps = orig.get("num_episodes", num_episodes)
                orig_seeds = orig.get("num_random_seeds", self.num_random_seeds)
                if orig_eps != num_episodes or orig_seeds != self.num_random_seeds:
                    print(f"\n{RED}{BOLD}  [Config Mismatch] 원래 세션 설정: NUM_EPISODES={orig_eps}, NUM_RANDOM_SEEDS={orig_seeds}{RESET}")
                    print(f"{RED}  현재 설정: NUM_EPISODES={num_episodes}, NUM_RANDOM_SEEDS={self.num_random_seeds}{RESET}")
                    print(f"{RED}  → run_forward_and_reset.sh에서 NUM_EPISODES={orig_eps}, NUM_RANDOM_SEEDS={orig_seeds}로 맞춰주세요.{RESET}")
            else:
                # session_config.json 없는 이전 세션: 실제 데이터에서 추정
                # phase1/, phase2/ subdir 인식 (chain 이후 reorg 된 session 포함)
                actual_episodes = len(_iter_episode_dirs(session_dir))
                if actual_episodes > num_episodes:
                    print(f"\n{RED}{BOLD}  [Config Mismatch] Session has {actual_episodes} episodes but NUM_EPISODES={num_episodes}, NUM_RANDOM_SEEDS={self.num_random_seeds}{RESET}")
                    print(f"{RED}  → run_forward_and_reset.sh의 NUM_EPISODES와 NUM_RANDOM_SEEDS를 원래 세션 설정으로 맞춰주세요.{RESET}")
            print(f"\n{GREEN}  All batches complete, nothing to resume{RESET}")
        else:
            # Resume 모드에서는 워크스페이스 상태가 보장되지 않으므로,
            # 첫 미완료 배치(batch 0 포함)의 seed 위치로 항상 물리적 restore 수행.
            # - Batch 0의 seed_positions[0]은 _load_resume_state에서 first_episode_positions로 채워짐
            # - Batch 1+ 의 seed가 누락되면 _generate_seed_positions로 새로 생성
            if seed_positions[first_incomplete] is None and first_incomplete > 0:
                print(f"\n{MAGENTA}{BOLD}  Generating seed_{first_incomplete+1}...{RESET}")
                seed_positions[first_incomplete] = self._generate_seed_positions(session_dir, first_incomplete)
            if seed_positions[first_incomplete] is not None:
                print(f"\n{CYAN}{BOLD}  Restoring to seed_{first_incomplete+1}...{RESET}")
                self._restore_to_seed(seed_positions[first_incomplete], instruction, detection_timeout)
            else:
                print(f"\n{YELLOW}  [Resume] Warning: seed_{first_incomplete+1} positions unavailable, "
                      f"skipping physical restore. Workspace must already be in correct state.{RESET}")

            # 에피소드 루프 — 위에서 산출한 _schedule 을 그대로 순회.
            for execution_idx, batch_index, slot, episode_num, is_round_last in _schedule:

                # 이미 성공한 slot → 스킵
                if batch_slots[batch_index][slot]:
                    continue

                self.current_episode = episode_num
                self._current_batch_index = batch_index
                self._current_slot = slot

                # Per-episode perturbation RNG seed (no-op if perturbation disabled).
                self._seed_episode_perturbation(batch_index, slot)

                print("\n" + CYAN + "=" * 70 + RESET)
                print(CYAN + BOLD + f"  [{episode_num:02d}/{num_episodes:02d}] Episode (Batch {batch_index+1}, Slot {slot})  ".center(70) + RESET)
                print(CYAN + "=" * 70 + RESET)

                episode_dir = str(_episode_dir(session_dir, episode_num, getattr(self, "method3_phase", None)))
                Path(episode_dir).parent.mkdir(parents=True, exist_ok=True)

                # Reset target — schedule 순서상 "다음에 실행할 미완료
                # episode" 의 batch seed. round_robin 은 episode 마다 batch 가
                # 바뀌므로 "같은 batch 유지" 가정이 깨진다 (seed_major 에서는
                # 다음 slot/batch 가 동일 batch 라 기존과 같은 결과).
                reset_target = None
                for _e2, _b2, _s2, _ep2, _rl2 in _schedule[execution_idx + 1:]:
                    if not batch_slots[_b2][_s2]:
                        if seed_positions[_b2] is None:
                            print(f"\n{MAGENTA}  [Seed Transition] Generating seed_{_b2+1}...{RESET}")
                            seed_positions[_b2] = self._generate_seed_positions(session_dir, _b2)
                        reset_target = seed_positions[_b2]
                        break
                if reset_target is None:
                    # 마지막 실행 episode — 더 실행할 게 없으면 현재 batch 로 정리.
                    reset_target = seed_positions[batch_index]

                try:
                    _slot_r = slot
                    _ep_dir_r = episode_dir
                    _bi_r = batch_index
                    def _post_judge_resume(res):
                        jp = res['judge'].get('prediction', 'UNCERTAIN')
                        from pipeline.save_logs import save_batch_info as _sbi_r
                        _sbi_r(_ep_dir_r, _bi_r, _slot_r, jp)

                    result = self.run(
                        instruction=instruction, objects=objects,
                        detection_timeout=detection_timeout,
                        visualize_detection=visualize_detection,
                        save_dir=episode_dir, use_timestamp_subdir=False,
                        skip_reset=skip_reset, reset_target_positions=reset_target,
                        post_judge_callback=_post_judge_resume,
                    )

                    judge_pred = result['judge'].get('prediction', 'UNCERTAIN')
                    if judge_pred == 'TRUE':
                        batch_slots[batch_index][slot] = True
                    self._update_results(all_results, result, episode_num, skip_reset)

                except Exception as e:
                    print(f"\n{RED}[{episode_num:02d}/{num_episodes:02d}] Error: {e}{RESET}")
                    import traceback; traceback.print_exc()
                    all_results['episodes'].append({'episode': episode_num, 'result': None, 'success': False, 'error': str(e)})

                # Hook check — round_robin: end-of-round 만. seed_major: 매 episode.
                _hook = getattr(self, "_phase1_readiness_hook", None)
                if _hook is not None and _hook.enabled:
                    _do_check = is_round_last if is_round_robin else True
                    if _do_check:
                        _sel = getattr(self, "_subgoal_selector", None)
                        _buf = getattr(_sel, "buffer", None) if _sel is not None else None
                        # 누적 episodes_done = batch_slots 의 True 개수.
                        # 이전 run + 이번 run 의 success 합. round_robin 에서
                        # phase1_min/max 가 *누적* 카운트와 비교돼야 정합.
                        _episodes_done = sum(int(s) for row in batch_slots for s in row)
                        if _hook.should_stop(_episodes_done, _buf,
                                             episode_label=episode_num):
                            print(MAGENTA + BOLD +
                                  f"  [method3:phase1] resume loop terminated early — "
                                  f"done={_episodes_done} folder=ep{episode_num} "
                                  f"(reason={_hook.stop_reason})  ".center(70) + RESET)
                            break
                time.sleep(2)

        self._finalize_session(all_results, session_dir, num_episodes, instruction, objects, skip_reset)
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
        default=[],
        help="Objects to detect (only used in single-turn mode)"
    )

    # 선택 인자
    parser.add_argument(
        "--robot", "-r",
        type=int,
        nargs="+",
        default=[3],
        help="Robot ID(s). Single: --robot 2, Dual: --robot 2 3"
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
        help="Show real-time detection visualization (single-turn mode only)"
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
        "--skip-forward",
        action="store_true",
        help="Skip forward+judge phase (only run reset+reset_judge). Used by forward_ma which performs forward externally and delegates reset to AC."
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

    parser.add_argument(
        "--recording-config",
        type=str,
        default=None,
        help="Path to recording config YAML (e.g., pipeline_config/recording_config_ws1.yaml). "
             "If not specified, uses pipeline_config/recording_config.yaml"
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
        "--detect-model",
        type=str,
        default=None,
        help="VLM model for detect_objects skill (default: uses gemini-3.1-flash-lite)"
    )

    parser.add_argument(
        "--side-view-image",
        type=str,
        default=None,
        help="Side-view image path for Turn Test waypoint trajectory prediction"
    )

    parser.add_argument(
        "--reset-instruction",
        type=str,
        default=None,
        help="Reset task instruction for recording. Default: 'move objects to certain position'"
    )

    parser.add_argument(
        "--skip-turn-test",
        action="store_true",
        help="Skip Turn Test (Waypoint Trajectory Prediction) in multi-turn code generation"
    )

    parser.add_argument(
        "--resetspace-per-robot",
        type=str,
        nargs="+",
        default=None,
        help="Per-robot reset quadrant (all, top-left, top-right, bottom-left, bottom-right). "
             "Order matches --robot order. Default: 'all' for each robot."
    )

    # ── Method3 phase toggle (final_method3_spec §2) ──
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase1", "phase2"],
        default="phase1",
        help="Method3 phase. phase1 = subgoal seeding (buffer-aware, default). "
             "phase2 = MI-based selection (Q2). phase2 requires --phase1-trained-vla-path "
             "and either --phase1-dataset-path or a cached skill_wise_vector_db.npz in session_dir."
    )
    parser.add_argument(
        "--phase1-trained-vla-path",
        type=str,
        default=None,
        help="(--phase phase2) §6 frozen VLA encoder checkpoint. Phase1-trained VLA "
             "가 준비되기 전엔 pretrained VLA 체크포인트 경로를 임시로 넣어 동작 확인 가능."
    )
    parser.add_argument(
        "--phase1-dataset-path",
        type=str,
        default=None,
        help="(--phase phase2) §6 re-embedding 의 Phase1 raw dataset (LeRobot repo_id "
             "또는 local path). 캐시된 skill_wise_vector_db.npz 가 있으면 무시."
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

    # 파이프라인 실행: 로봇 수에 따라 분기
    robot_ids = args.robot  # list of int (nargs="+")

    if len(robot_ids) == 1:
        # ── Single-arm: 기존 ForwardAndResetPipeline ──
        # resetspace: single-arm이면 첫 번째 값 사용
        _rs = args.resetspace_per_robot
        single_resetspace = _rs[0] if _rs else None

        pipeline = ForwardAndResetPipeline(
            robot_id=robot_ids[0],
            llm_model=args.llm,
            judge_model=args.judge_model,
            judge_timeout_ms=int(args.judge_timeout * 1000),
            num_random_seeds=args.num_random_seeds,
            verbose=True,
            record_dataset=args.record,
            dataset_repo_id=args.dataset_repo_id,
            resume_recording=bool(args.resume),
            multi_turn=args.multi_turn,
            cad_image_dirs=args.cad_image_dirs,
            side_view_image=args.side_view_image,
            recording_fps=args.recording_fps,
            codegen_model=args.codegen_session2_model,
            reset_instruction=args.reset_instruction,
            skip_turn_test=args.skip_turn_test,
            detect_model=args.detect_model,
            resetspace=single_resetspace,
            recording_config=args.recording_config,
            method3_phase=args.phase,
            phase1_trained_vla_path=args.phase1_trained_vla_path,
            phase1_dataset_path=args.phase1_dataset_path,
        )
    else:
        # ── Multi-arm: UnifiedMultiArmPipeline ──
        from unified_multi_arm import UnifiedMultiArmPipeline
        # resetspace: multi-arm이면 로봇 순서대로 매핑
        _rs = args.resetspace_per_robot or ["all"] * len(robot_ids)
        # 부족하면 마지막 값으로 채움
        while len(_rs) < len(robot_ids):
            _rs.append(_rs[-1] if _rs else "all")
        resetspace_per_robot = dict(zip(robot_ids, _rs))

        pipeline = UnifiedMultiArmPipeline(
            robot_ids=robot_ids,
            llm_model=args.llm,
            judge_model=args.judge_model,
            judge_timeout_ms=int(args.judge_timeout * 1000),
            num_random_seeds=args.num_random_seeds,
            verbose=True,
            record_dataset=args.record,
            dataset_repo_id=args.dataset_repo_id,
            resume_recording=bool(args.resume),
            multi_turn=args.multi_turn,
            cad_image_dirs=args.cad_image_dirs,
            side_view_image=args.side_view_image,
            recording_fps=args.recording_fps,
            codegen_model=args.codegen_session2_model,
            reset_instruction=args.reset_instruction,
            skip_turn_test=args.skip_turn_test,
            detect_model=args.detect_model,
            resetspace_per_robot=resetspace_per_robot,
            recording_config=args.recording_config,
        )

    # 에피소드 실행: resume 모드와 새 세션 모드 분기
    if args.resume:
        all_results = pipeline.resume_multiple_episodes(
            num_episodes=args.num_episodes,
            instruction=args.instruction,
            objects=args.objects,
            resume_session_dir=args.resume,
            detection_timeout=args.timeout,
            visualize_detection=args.visualize_detection,
            save_dir=args.save,
            skip_reset=args.skip_reset,
        )
    else:
        all_results = pipeline.run_multiple_episodes(
            num_episodes=args.num_episodes,
            instruction=args.instruction,
            objects=args.objects,
            detection_timeout=args.timeout,
            visualize_detection=args.visualize_detection,
            save_dir=args.save,
            skip_reset=args.skip_reset,
            skip_forward=args.skip_forward,
        )

    # ========================================================
    # Phase1 early termination → 3-option prompt loop (chain / resume / more)
    # ========================================================
    # phase1 boundary check 가 ready → method3_integration.phase1_end_prompt 가
    # 사용자 입력 받아 chain / resume / more 분기. (2)/(3) 은 self 재호출로 loop
    # 진입; (1) chain 만 break out.
    _hook = getattr(pipeline, "_phase1_readiness_hook", None)
    _stop_reason = getattr(_hook, "stop_reason", None) if _hook is not None else None
    _phase_str = str(getattr(args, "phase", "")).lower()
    _session_dir_str = getattr(pipeline, "_session_dir", None)
    if (_phase_str == "phase1"
            and _stop_reason
            and str(_stop_reason).startswith("phase1_ready")
            and _session_dir_str
            and getattr(args, "dataset_repo_id", None)):
        try:
            from method3_integration.phase1_end_prompt import run_prompt_loop
            run_prompt_loop(
                session_dir=Path(_session_dir_str),
                dataset_repo_id=str(args.dataset_repo_id),
                stop_reason=str(_stop_reason),
                project_root=Path(__file__).resolve().parent,
                self_argv=sys.argv,
            )
        except Exception as _e:
            print(f"  [phase1_end_prompt] failed: {_e}", flush=True)

    # 종료 코드 결정 (성공률 기반). resume 이 "all done" 으로 빈 결과를
    # 돌릴 수 있으므로 .get() 로 안전 접근.
    n = all_results.get('num_episodes', 0)
    s = all_results.get('summary', {}) or {}
    if n > 0:
        success_rate = s.get('forward_judge_true', 0) / n
    else:
        success_rate = 1.0  # nothing to do = success

    if success_rate >= 0.5:  # 50% 이상 성공
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
