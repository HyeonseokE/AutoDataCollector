"""Method3 acquisition orchestrator — two-phase online acquisition (문서 §2).

method3 의 다섯 라이브러리를 하나의 acquisition 루프로 통합한다:

    Phase1 (state seeding)  ──readiness(§15.1)──▶ transition(§15.3)
        │                                              │
        │ raw → D_phase1_raw (storage §3)               │ re-embed (§6)
        ▼                                              ▼
    PhaseController (§15.3)                      P_phase1 vector DB
        │                                              │
        └──────────────▶ Phase2 (MI selection §7-14) ◀──┘
                              │ accept → D_phase2_raw + vector DB
                              ▼
                      saturation/budget stop (§15.2/§15.3)

오케스트레이터는 phase 전환·budget·re-embedding 트리거·MI scoring 등 모든
method3 로직을 직접 소유하고, 로봇 실행·VLA 모델·후보 생성은 ``Acquisition
Environment`` 포트에 위임한다 (ports.py).

``verbose=True`` 면 run 전 과정을 ``[method3:acq]`` 로그로 narration 한다 —
하드웨어/offline 디버깅 시 phase 전환·readiness·re-embedding·accept 흐름을
한 스트림으로 추적하기 위한 것. per-candidate MI 상세는 ``phase2_mi.debug_verbose``
로 따로 켠다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from method3.acquisition.ports import AcquisitionEnvironment
from method3.phase2_mi_selection.mi_selector import Phase2MIConfig, Phase2MISelector
from method3.phase_control.phase1_readiness import (
    Phase1ReadinessConfig,
    evaluate_phase1_readiness,
)
from method3.phase_control.phase_controller import PhaseController, PhaseControllerConfig
from method3.reembedding.seed_builder import ReembeddingConfig, build_phase1_vector_db
from method3.storage.raw_dataset import RawTrajectoryDataset


@dataclass
class Method3AcquisitionConfig:
    """Method3 acquisition 전체 설정 (문서 §2-§16)."""

    phase1_raw_dir: str | Path                       # D_phase1_raw 디렉터리
    phase2_raw_dir: str | Path                       # D_phase2_raw 디렉터리
    phase_controller: PhaseControllerConfig = field(default_factory=PhaseControllerConfig)
    readiness: Phase1ReadinessConfig = field(default_factory=Phase1ReadinessConfig)
    phase2_mi: Phase2MIConfig = field(default_factory=Phase2MIConfig)
    reembedding: ReembeddingConfig = field(default_factory=ReembeddingConfig)
    verbose: bool = False                            # True → run narration 로그 출력

    def __post_init__(self) -> None:
        # §6 — re-embedding 과 Phase2 candidate 의 action descriptor 는 같은 K 를
        # 써야 같은 metric space 에 놓인다.
        if self.phase2_mi.dct_coeffs != self.reembedding.dct_coeffs:
            raise ValueError(
                "phase2_mi.dct_coeffs must equal reembedding.dct_coeffs "
                f"({self.phase2_mi.dct_coeffs} != {self.reembedding.dct_coeffs})"
            )


@dataclass(frozen=True)
class AcquisitionReport:
    """acquisition run 종료 요약."""

    total_episodes: int                  # t — 소비한 전체 episode
    phase1_episodes: int | None          # B_1 — Phase1 episode 수 (전환 안 했으면 None)
    phase2_episodes: int                 # Phase2 episode 수
    phase2_accepted: int                 # accept 된 Phase2 후보 수
    final_phase: str                     # 종료 시점 phase
    stop_reason: str                     # "budget" | "saturation"
    last_readiness: float                # 마지막으로 측정한 R_ready
    phase1_seed_size: int                # P_phase1 entry 수 (re-embedding 결과)
    vector_db_size: int                  # 종료 시점 B_t^{(m)} 전체 entry 수


class Method3Acquisition:
    """Phase1↔Phase2 two-phase acquisition 오케스트레이터 (문서 §2).

    Usage::

        acq = Method3Acquisition(env, config)
        report = acq.run()
    """

    def __init__(
        self,
        env: AcquisitionEnvironment,
        config: Method3AcquisitionConfig,
    ) -> None:
        self.env = env
        self.cfg = config
        self.d_phase1_raw = RawTrajectoryDataset(config.phase1_raw_dir)
        self.d_phase2_raw = RawTrajectoryDataset(config.phase2_raw_dir)
        self.controller = PhaseController(config.phase_controller)
        # transition(§6) 후에 채워진다.
        self._encoder = None
        self._seed_db = None
        self.mi_selector: Phase2MISelector | None = None
        self._seed_subgoals: dict = {}       # G_seed^{(m)} — 전환 시 채워진다
        # 통계.
        self._phase2_episodes = 0
        self._phase2_accepted = 0
        self._last_readiness = 0.0
        self._phase1_seed_size = 0           # 전환 시점 P_phase1 크기 (고정값)

    def _log(self, msg: str) -> None:
        """verbose 가 켜져 있으면 run narration 로그를 출력한다."""
        if self.cfg.verbose:
            print(f"[method3:acq] {msg}")

    def run(self) -> AcquisitionReport:
        """budget 소진 또는 saturation 까지 two-phase acquisition 을 돌린다 (§15.3)."""
        pc = self.cfg.phase_controller
        self._log(
            f"run start — budget={pc.budget} "
            f"phase1 guard=[{pc.phase1_min},{pc.phase1_max}] τ_ready={pc.tau_ready}"
        )
        while not self.controller.should_stop():
            if self.controller.phase == "phase1":
                self._run_phase1_step()
            else:
                self._run_phase2_step()
        report = self._report()
        self._log(
            f"run end — t={report.total_episodes} stop={report.stop_reason} "
            f"phase1={report.phase1_episodes} phase2={report.phase2_episodes} "
            f"accepted={report.phase2_accepted} "
            f"vectorDB={report.vector_db_size} (seed={report.phase1_seed_size})"
        )
        return report

    # ── Phase1 ──────────────────────────────────────────────
    def _run_phase1_step(self) -> None:
        """Phase1 episode 한 번 — raw 적재 + readiness 측정 + 전환 판단 (§4, §15)."""
        result = self.env.run_phase1_episode()
        for entry in result.raw_entries:
            self.d_phase1_raw.append(entry)               # → D_phase1_raw (§3)
        self.controller.record_episode()
        self._log(
            f"Phase1 ep t={self.controller.t} — "
            f"judged={'TRUE' if result.judged_true else 'FALSE'} "
            f"+{len(result.raw_entries)} raw (D_phase1_raw={len(self.d_phase1_raw)})"
        )

        # §15.3 — B_{1,min} 이후부터 readiness 를 측정하고 전환을 판단한다.
        if self.controller.t >= self.cfg.phase_controller.phase1_min:
            probes = self.env.probe_readiness_candidates()
            report = evaluate_phase1_readiness(probes, self.cfg.readiness)
            self._last_readiness = report.r_ready
            per = {k: round(v, 2) for k, v in report.per_skill.items()}
            self._log(
                f"  readiness probe — R_ready={report.r_ready:.3f} "
                f"ready={report.is_ready} per-skill={per}"
            )
            if self.controller.maybe_transition(report.r_ready):
                self._transition_to_phase2()

    def _transition_to_phase2(self) -> None:
        """§6 — Phase1 raw 를 re-embedding 하여 Phase2 vector DB seed 를 만든다."""
        self._log(f"── Phase1→Phase2 transition at t={self.controller.t} ──")
        self._encoder = self.env.build_phase2_encoder(self.d_phase1_raw)   # §6 Step1-2
        self._seed_db = build_phase1_vector_db(                            # §6 Step3-4
            self.d_phase1_raw,
            self._encoder,
            self.env.resolve_observation,
            self.cfg.reembedding,
        )
        # P_phase1 크기를 전환 시점에 고정한다 — 이후 vector DB 는 Phase2
        # accept 로 커지므로 (_seed_db 와 mi_selector.db 는 동일 객체).
        self._phase1_seed_size = self._seed_db.total_size()
        self.mi_selector = Phase2MISelector(self._seed_db, self.cfg.phase2_mi)
        # phase1_seed_anchor_logic — Phase1 이 모은 seed subgoal 집합 G_seed^{(m)}.
        # Phase2 는 새 subgoal 을 탐색하지 않고 이 anchor 주변만 큐레이션한다.
        self._seed_subgoals = self.env.collect_seed_subgoals()
        _anchors = {m: len(g) for m, g in self._seed_subgoals.items()}
        self._log(
            f"  §6 re-embed — D_phase1_raw({len(self.d_phase1_raw)}) → "
            f"P_phase1 vector DB({self._phase1_seed_size} entries, "
            f"{len(self._seed_db.skill_ids())} skills) | G_seed anchors={_anchors}"
        )

    # ── Phase2 ──────────────────────────────────────────────
    def _run_phase2_step(self) -> None:
        """Phase2 episode 한 번 — 후보 생성·MI scoring·accept (§7-14)."""
        assert self.mi_selector is not None, "phase2 reached before transition"
        self._phase2_episodes += 1
        ep = self.controller.t + 1                        # 진행 중인 episode 번호
        candidates = self.env.generate_phase2_candidates(
            self._encoder, self._seed_subgoals)
        if not candidates:
            self._log(f"Phase2 ep t={ep} — no candidates generated, skip")
            self.controller.record_episode()
            return
        selection = self.mi_selector.select(candidates)               # §11-12
        rep = selection.reports[selection.chosen_index]
        self._log(
            f"Phase2 ep t={ep} — {len(candidates)} candidates → chose "
            f"#{selection.chosen_index} ΔH_A={rep.delta_h_a:.3f} "
            f"ΔH_A|S={rep.delta_h_a_given_s:.3f} Q2={rep.q2:.3f} "
            f"Q̃2={rep.q2_norm:.3f} accepted={selection.accepted}"
        )
        if selection.accepted:
            self._accept_phase2(selection)
        self.controller.record_episode()

    def _accept_phase2(self, selection) -> None:
        """accept 된 후보를 실행·D_phase2_raw 적재·vector DB 누적한다 (§14)."""
        result = self.env.execute_phase2_candidate(selection.chosen_candidate)
        start = len(self.d_phase2_raw)
        for entry in result.raw_entries:
            self.d_phase2_raw.append(entry)                                # → D_phase2_raw
        report = selection.reports[selection.chosen_index]
        # §3.1 dataset_ref — accept 된 trajectory 의 raw entry 구간 포인터.
        ref = {
            "dataset_dir": str(self.d_phase2_raw.dataset_dir),
            "entry_start": start,
            "entry_end": len(self.d_phase2_raw),
        }
        # §14 — seed anchor 도 metadata 에 남긴다 (phase1_seed_anchor_logic).
        _anchor = selection.chosen_candidate.seed_subgoal
        self.mi_selector.accept_to_buffer(                                 # §14
            selection.chosen_candidate,
            ref=ref,
            meta={
                "q2": report.q2, "q2_norm": report.q2_norm,
                "seed_subgoal": None if _anchor is None else [float(x) for x in _anchor],
            },
        )
        self.controller.record_phase2_accept(report.q2_norm)               # §15.2
        self._phase2_accepted += 1
        window_mean = self.controller.status().saturation_window_mean
        sat = f" Q̄̃2^(W)={window_mean:.3f}" if window_mean is not None else ""
        self._log(
            f"  accepted → +{len(result.raw_entries)} raw "
            f"(D_phase2_raw={len(self.d_phase2_raw)}) "
            f"vector DB={self.mi_selector.db.total_size()}{sat}"
        )

    # ── report ──────────────────────────────────────────────
    def _report(self) -> AcquisitionReport:
        budget = self.cfg.phase_controller.budget
        stop_reason = "budget" if self.controller.t >= budget else "saturation"
        return AcquisitionReport(
            total_episodes=self.controller.t,
            phase1_episodes=self.controller.status().phase1_episodes,
            phase2_episodes=self._phase2_episodes,
            phase2_accepted=self._phase2_accepted,
            final_phase=self.controller.phase,
            stop_reason=stop_reason,
            last_readiness=self._last_readiness,
            phase1_seed_size=self._phase1_seed_size,
            vector_db_size=self.mi_selector.db.total_size() if self.mi_selector else 0,
        )
