"""Drop-in skill-planner client backed by the preselective gRPC server.

The local CuroboBackend exposes
``plan_batch(start, goal, n, seed) → list[TrajectoryCandidate]`` and lets
skills_lerobot sample ONE candidate locally. In gRPC mode the server runs
both planning AND IG·AC selection; the adapter here returns a single-element
list containing the server's chosen trajectory, so skills_lerobot's existing
selection code naturally picks it.

Per-step inputs flow:
    skills_lerobot.move_to_position
        ↓ plan_batch(start, goal, n, seed)
    GrpcPlannerClient.plan_batch
        ↓ pulls latest images + instruction from orchestrator
        ↓ PreselectiveClient.plan_and_select(server_addr)
    server.PlanAndSelect
        ↓ curobo.plan_batch + selector.select
        ↓ caches (ctx, selection) by selection_id
    GrpcPlannerClient
        ↓ wraps server response as TrajectoryCandidate (single-element list)
    skills_lerobot
        → executes trajectory
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grpc_server.client import PreselectiveClient


@dataclass(frozen=True)
class _RemoteTrajectoryCandidate:
    """TrajectoryCandidate-shaped duck-type built from server response.

    Mirrors perturbation.skill_level.planner.TrajectoryCandidate fields
    that skills_lerobot.move_to_position consumes for logging + execution.
    """
    waypoints: np.ndarray  # (N, dof) joint-space
    algo: str
    seed: int
    plan_time_s: float
    cost: float | None
    # Helper: server-provided times (T,). Optional metadata for downstream use.
    times: np.ndarray | None = None


class GrpcPlannerClient:
    """Wraps PreselectiveClient to look like a skill-planner backend.

    Construct with a ContextProvider that knows how to fetch the current
    camera frames + episode instruction at plan time (typically the orchestrator
    object via duck-typed methods).
    """

    def __init__(
        self,
        client: PreselectiveClient,
        context_provider,
        skill_id: str = "move_to",
    ) -> None:
        self._client = client
        self._provider = context_provider
        self._skill_id = str(skill_id)
        # candidate dump 참조 — plan_batch 마다 (selection_id, skill_id) 누적.
        # episode 종료 시 client 가 pop_dump_refs() 로 가져가 server 의 dump
        # (phase2_cands/<selection_id>.npz) 를 scp + 오버레이한다.
        self._dump_refs: list[tuple[str, str]] = []
        # episode-내 transit move ordinal counter — P_phase1 DB 의 partition
        # key namespace (skill_0..skill_N) 와 server-side lookup 을 정합시키기
        # 위한 가장 안정적인 source. plan_batch 마다 increment, pop_dump_refs()
        # (episode 종료) 가 0 으로 reset 한다. caller 가 plan_batch(skill_id=)
        # 로 명시 override 하지 않는 한 server 로 보내는 ordinal 키.
        self._skill_call_index = 0

    def pop_dump_refs(self) -> list[tuple[str, str]]:
        """누적된 (selection_id, skill_id) 목록을 반환하고 비운다.

        client 가 episode 종료 시 호출 — 이번 episode 의 plan_and_select 들이
        남긴 dump 참조를 모두 가져가고 다음 episode 를 위해 clear.
        """
        refs = list(self._dump_refs)
        self._dump_refs.clear()
        # episode 경계 — 다음 episode 의 첫 transit move 가 skill_0 부터 시작
        # 하도록 ordinal counter 도 함께 reset (P_phase1 DB key 와 정합).
        self._skill_call_index = 0
        return refs

    # ------------------------------------------------------------------
    # plan_batch — drop-in replacement for CuroboBackend
    # ------------------------------------------------------------------
    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        seed: int | None = None,
        rng: Any | None = None,  # accepted for signature parity; ignored
        skill_id: str | None = None,  # ← caller 가 명시 전달 (episode-내 ordinal skill_k)
        skill_type: str | None = None,  # ← caller 가 명시 전달 (VLA instruction prefix 용)
    ) -> list[_RemoteTrajectoryCandidate]:
        """Send context to server, receive ONE chosen trajectory.

        Images are shipped so the server's frozen VLA encoder can compute the
        FAISS key embedding; the encoder tolerates an empty dict.
        """
        _episode_task = ""
        try:
            _episode_task = str(getattr(self._provider, "instruction", "") or "")
        except Exception:
            pass
        try:
            images = self._provider._latest_observation_dict()
        except Exception:
            images = {}

        # skill_id = episode-내 ordinal partition 키 (skill_0..) — P_phase1 의
        # skill_{skill_index} 와 같은 key namespace 라야 server-side DB lookup
        # 이 hit 한다. caller 가 명시 ordinal 을 전달하면 그대로 사용 (test/
        # override), 아니면 내부 counter (pop_dump_refs 에서 episode 마다 reset)
        # 가 매기는 skill_{N}. NL skill_type 은 별도 skill_type= 인자로 받아
        # VLA instruction format 에만 쓴다 (DB partition key 와 분리).
        if skill_id is not None:
            effective_skill_id = str(skill_id)
        else:
            effective_skill_id = f"skill_{self._skill_call_index}"
        self._skill_call_index += 1
        # VLA instruction = {skill_type}: {episode_task} — 학습(SkillDCTDataset)·
        # DB build 와 동일 format. server 는 skill_id 가 ordinal 이라 재포맷 못 하므로
        # client 가 여기서 format 해 보낸다. skill_type 은 caller 전달, fallback 은
        # RecordingContext._current_skill_type.
        _runtime_skill = ""
        try:
            from record_dataset.context import RecordingContext as _RC
            _runtime_skill = _RC._current_skill_type or ""
        except Exception:
            pass
        from method3.dct.instruction_format import format_skill_instruction
        instruction = format_skill_instruction(
            skill_type or _runtime_skill, _episode_task
        )

        # *current robot state* (servo position 6-dim, range ±100) — DB build
        # proprio (observation.state) 와 같은 unit. provider 가 노출하면 사용,
        # 아니면 start_qpos (joint angles radians) 로 fallback (legacy).
        _rs = None
        try:
            _rs = self._provider._latest_robot_state()
        except Exception:
            _rs = None
        _state_arg = np.asarray(
            _rs if _rs is not None else start_qpos, dtype=np.float32,
        )

        try:
            resp = self._client.plan_and_select(
                skill_id=effective_skill_id,
                start_qpos=np.asarray(start_qpos, dtype=np.float32),
                goal_qpos=np.asarray(goal_qpos, dtype=np.float32),
                state=_state_arg,
                images=images,
                instruction=instruction,
                n_candidates=int(n),
                seed=int(seed) if seed is not None else 0,
                is_transit=True,
            )
        except Exception as e:
            print(f"  [Skill Perturbation] grpc plan_and_select failed: {e}")
            return []

        # selection 결과를 client 터미널에 즉시 출력 — TRUE(green) / FALLBACK(yellow)
        # 라벨 + fallback 시 reason 한 줄. PNG 시각화의 라벨과 짝.
        _summary = _summarize_score_report(
            resp.get("score_report_json", ""),
            chosen_index=resp.get("chosen_index", -1),
        )
        _is_fallback = bool(resp.get("used_fallback", False))
        _G = "\033[1;92m"   # bold green
        _Y = "\033[1;93m"   # bold yellow
        _DIM_Y = "\033[93m"  # plain yellow (reason)
        _R = "\033[0m"
        if _is_fallback:
            _label = f"{_Y}FALLBACK{_R}"
            _reason = _fallback_reason(
                resp.get("score_report_json", ""),
                chosen_index=resp.get("chosen_index", -1),
            )
            _reason_line = f"\n  {_DIM_Y}└─ reason: {_reason}{_R}" if _reason else ""
        else:
            _label = f"{_G}TRUE{_R}"
            _reason_line = ""
        print(
            f"[Phase2-Selection] skill={effective_skill_id} {_label} | "
            f"{_summary}{_reason_line}",
            flush=True,
        )

        # candidate dump 참조 누적 — server 가 dump 한 npz 파일명 = selection_id.
        # episode 종료 시 client 가 pop_dump_refs() 로 가져가 scp + 오버레이.
        _sid = str(resp.get("selection_id", "") or "")
        if _sid:
            self._dump_refs.append((_sid, effective_skill_id))

        if resp.get("used_fallback", False):
            return []

        traj = resp["trajectory"]
        wp = np.asarray(traj["waypoints"], dtype=np.float32)
        times = np.asarray(traj.get("times", []), dtype=np.float32)
        return [_RemoteTrajectoryCandidate(
            waypoints=wp,
            algo=str(traj.get("algo", "grpc:remote")),
            seed=int(traj.get("seed", 0)),
            plan_time_s=0.0,
            cost=float(traj.get("cost", 0.0)),
            times=times if times.size else None,
        )]


def _fallback_reason(json_str: str, chosen_index: int = -1) -> str:
    """fallback 시 1-line reason — under_covered 비율 + chosen 의 under 으로 추론.

    server 의 Phase2MISelector 가 fallback (eligible=∅) 으로 가는 3 가지 케이스:
      - 모든 candidate under_covered (buffer 가 그 영역 미관측)
      - under 이 일부지만 M̃_MI ≥ τ_MI 만족 0
      - server-side curobo 가 0 valid candidates 반환 (used_fallback=True 자체)
    """
    if not json_str:
        return "server returned used_fallback (no score report — likely 0 valid curobo candidates)"
    try:
        import json as _json
        reports = _json.loads(json_str)
    except Exception:
        return "score report parse failed"
    if not reports:
        return "empty score report (server-side curobo returned 0 candidates)"
    K = len(reports)
    under = sum(1 for r in reports if r.get("under_covered"))
    chosen = next((r for r in reports if int(r.get("i", -1)) == int(chosen_index)),
                  None)
    chosen_under = bool(chosen.get("under_covered", False)) if chosen else False
    if under == K:
        return (f"all {K} candidates under-covered "
                f"(buffer never observed this region) → eligible=0, "
                f"argmax M_MI fallback, NOT accepted to buffer")
    if chosen_under:
        return (f"chosen #{chosen_index} under-covered "
                f"(under={under}/{K}); eligible was empty → fallback argmax M_MI")
    return (f"eligible=0 (under={under}/{K}, no candidate satisfies "
            f"M̃_MI ≥ τ_MI) → fallback argmax M_MI")


def _summarize_score_report(json_str: str, chosen_index: int = -1) -> str:
    """server 의 score_report_json (Phase2ScoreReport list) → 한 줄 summary."""
    if not json_str:
        return "no report"
    try:
        import json as _json
        reports = _json.loads(json_str)
    except Exception as e:
        return f"parse failed: {e}"
    if not reports:
        return "empty report"
    n = len(reports)
    under = sum(1 for r in reports if r.get("under_covered"))
    def _mm(key):
        xs = [float(r.get(key, 0.0)) for r in reports]
        if not xs:
            return (0.0, 0.0, 0.0)
        return (min(xs), max(xs), sum(xs) / n)
    dha_min, dha_max, dha_mean = _mm("delta_h_a")
    dhas_min, dhas_max, dhas_mean = _mm("delta_h_a_given_s")
    mn_min, mn_max, mn_mean = _mm("m_mi_norm")
    m_min, m_max, m_mean = _mm("m_mi")
    u_min, u_max, u_mean = _mm("u_vla")
    chosen_str = ""
    if chosen_index >= 0:
        cr = next((r for r in reports if int(r.get("i", -1)) == int(chosen_index)), None)
        if cr is not None:
            chosen_str = (
                f" | chosen #{chosen_index}: ΔH_A={cr.get('delta_h_a', 0):.3f} "
                f"ΔH_A|S={cr.get('delta_h_a_given_s', 0):.3f} "
                f"M_MI={cr.get('m_mi', 0):.3f} M̃_MI={cr.get('m_mi_norm', 0):+.3f} "
                f"U_VLA={cr.get('u_vla', 0):.3f} under={cr.get('under_covered', False)}"
            )
    return (
        f"K={n} under_covered={under}/{n} "
        f"ΔH_A=[{dha_min:.3f},{dha_max:.3f},μ={dha_mean:.3f}] "
        f"ΔH_A|S=[{dhas_min:.3f},{dhas_max:.3f},μ={dhas_mean:.3f}] "
        f"M̃_MI=[{mn_min:+.3f},{mn_max:+.3f},μ={mn_mean:+.3f}] "
        f"M_MI=[{m_min:.3f},{m_max:.3f},μ={m_mean:.3f}] "
        f"U_VLA=[{u_min:.3f},{u_max:.3f},μ={u_mean:.3f}]"
        f"{chosen_str}"
    )

    # ------------------------------------------------------------------
    # update_scene — server-side curobo can also accept obstacle updates,
    # but the prototype skips this (TODO: extend proto with UpdateScene RPC).
    # ------------------------------------------------------------------
    def update_scene(self, obstacles) -> None:  # noqa: ARG002
        # Silent no-op for now — server's curobo uses its yaml's static workspace.
        return None

    # ------------------------------------------------------------------
    # Lifecycle helpers used by orchestrator
    # ------------------------------------------------------------------
    def commit(self, judge_true: bool, episode_id: str = "") -> dict:
        return self._client.commit(judge_true=judge_true, episode_id=episode_id)

    def reset_pending(self) -> None:
        self._client.reset_pending()

    def ready(self) -> dict:
        return self._client.ready()

    def close(self) -> None:
        self._client.close()
