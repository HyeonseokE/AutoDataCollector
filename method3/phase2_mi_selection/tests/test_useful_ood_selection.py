"""Useful-OOD selection rule unit tests — final_method3_spec_useful_ood_updated §13.2.

테스트 전략 — selector 의 score 함수 자체는 다른 모듈이 검증하므로 여기는
**rule 자체** 만 본다. ``Phase2MISelector.score_one`` 을 monkeypatch 로 대체해
M_MI 값을 임의로 주입하고, vla_scorer 의 score 도 임의 값으로 주입한다.
이렇게 하면 다음 케이스가 깔끔하게 분리된다:

  1. vla_scorer 가 주어졌을 때 — argmax U_VLA among eligible
  2. M̃_MI < 0 인 후보는 eligible 에서 제외 (Harmful OOD reject)
  3. under_covered 후보는 eligible 에서 제외
  4. eligible 이 비면 accept=False, chosen=argmax M_MI
  5. vla_scorer=None backward-compat — argmax M_MI among eligible
  6. tau_MI 조정 시 stricter filter
  7. accept_threshold (deprecated alias) 가 tau_MI 로 매핑
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from method3.phase2_mi_selection import (
    ActionMagnitudeScorer,
    ConstantScorer,
    Phase2Candidate,
    Phase2MIConfig,
    Phase2MISelector,
    Phase2ScoreReport,
    SkillVectorDB,
)


# ---------------------------------------------------------------------------
# Helpers — selector 의 score_one 을 monkeypatch 하기 위한 stub 들
# ---------------------------------------------------------------------------


def _make_candidate(
    skill_id: str = "pick",
    T: int = 4,
    H: int = 5,
    D_a: int = 3,
    payload: object = None,
) -> Phase2Candidate:
    rng = np.random.default_rng(0)
    return Phase2Candidate(
        skill_id=skill_id,
        state_keys=rng.normal(size=(T, 8)).astype(np.float64),
        action_chunks=rng.normal(size=(T, H, D_a)).astype(np.float64),
        payload=payload,
    )


def _make_selector(tau_MI: float = 0.0) -> Phase2MISelector:
    db = SkillVectorDB()
    cfg = Phase2MIConfig(tau_MI=tau_MI, debug_verbose=False)
    return Phase2MISelector(vector_db=db, config=cfg)


def _stub_score_one_factory(m_mi_values: list[float], under_covered: list[bool]):
    """selector.score_one 을 대체할 stub — index 별 임의 M_MI / under_covered 주입."""

    def _stub(index: int, candidate):
        return Phase2ScoreReport(
            candidate_index=index,
            delta_h_a=0.0,
            delta_h_a_given_s=0.0,
            q2=float(m_mi_values[index]),
            q2_norm=0.0,
            covered_ratio=0.0 if under_covered[index] else 1.0,
            under_covered=bool(under_covered[index]),
        )

    return _stub


@dataclass(frozen=True)
class _FixedScorer:
    """index 별 임의 U_VLA 를 반환하는 stub scorer.

    selector 가 eligible 후보에만 score() 를 호출하므로 호출 순서가 아닌
    ``candidate.payload`` (테스트 helper 가 index 를 박아둠) 로 lookup 한다.
    """

    values: list[float]

    def score(self, candidate):
        i = int(candidate.payload)
        return float(self.values[i])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUsefulOODRule:
    def test_argmax_u_vla_among_eligible(self):
        """M̃_MI ≥ 0 인 후보 4개 중 U_VLA 가 가장 큰 것이 선택된다."""
        sel = _make_selector(tau_MI=-10.0)  # 모두 eligible 로 만들기 위해 매우 낮음
        cands = [_make_candidate(payload=i) for i in range(4)]
        sel.score_one = _stub_score_one_factory([1.0, 1.0, 1.0, 1.0], [False] * 4)  # type: ignore
        scorer = _FixedScorer(values=[0.1, 0.9, 0.5, 0.7])

        result = sel.select(cands, vla_scorer=scorer)

        assert result.accepted
        assert result.chosen_index == 1
        assert result.u_vla_chosen == pytest.approx(0.9)
        assert sorted(result.eligible_indices) == [0, 1, 2, 3]

    def test_low_mi_norm_filtered_out(self):
        """M̃_MI < τ_MI 인 후보는 U_VLA 가 높더라도 제외 (Harmful OOD reject)."""
        sel = _make_selector(tau_MI=0.0)
        cands = [_make_candidate(payload=i) for i in range(4)]
        # M_MI: [10, 1, 1, 1] — index 0 은 평균보다 매우 위, 1-3 은 평균보다 아래.
        sel.score_one = _stub_score_one_factory([10.0, 1.0, 1.0, 1.0], [False] * 4)  # type: ignore
        # U_VLA: index 2 가 가장 높지만 그 후보는 M̃_MI<0 이라 제외돼야 함.
        scorer = _FixedScorer(values=[0.5, 0.1, 9.9, 0.1])

        result = sel.select(cands, vla_scorer=scorer)

        assert result.accepted
        assert result.chosen_index == 0  # 유일하게 eligible
        assert result.eligible_indices == [0]

    def test_under_covered_filtered_out(self):
        """under_covered 후보는 M̃_MI 가 충분해도 eligible 에서 빠진다 (§9.1)."""
        sel = _make_selector(tau_MI=-10.0)
        cands = [_make_candidate(payload=i) for i in range(3)]
        sel.score_one = _stub_score_one_factory(  # type: ignore
            [1.0, 1.0, 1.0], [False, True, False],
        )
        scorer = _FixedScorer(values=[0.2, 9.9, 0.5])

        result = sel.select(cands, vla_scorer=scorer)

        assert result.accepted
        assert 1 not in result.eligible_indices
        assert result.chosen_index == 2  # 0,2 중 max U_VLA
        assert result.u_vla_chosen == pytest.approx(0.5)

    def test_no_eligible_fallback_not_accepted(self):
        """모든 후보가 under_covered 이거나 M̃_MI<τ_MI 면 accept=False, fallback chosen."""
        sel = _make_selector(tau_MI=100.0)  # 통과 불가능한 임계
        cands = [_make_candidate(payload=i) for i in range(3)]
        sel.score_one = _stub_score_one_factory([0.5, 0.7, 0.3], [False] * 3)  # type: ignore
        scorer = _FixedScorer(values=[0.0, 0.0, 0.0])

        result = sel.select(cands, vla_scorer=scorer)

        assert not result.accepted
        assert result.chosen_index == 1  # argmax M_MI
        assert result.eligible_indices == []

    def test_backward_compat_without_vla_scorer(self):
        """vla_scorer=None — stage 3 가 argmax M_MI 로 fallback (기존 동작)."""
        sel = _make_selector(tau_MI=-10.0)
        cands = [_make_candidate() for _ in range(3)]
        sel.score_one = _stub_score_one_factory([0.2, 0.9, 0.4], [False] * 3)  # type: ignore

        result = sel.select(cands, vla_scorer=None)

        assert result.accepted
        assert result.chosen_index == 1  # argmax M_MI
        assert result.u_vla_chosen is None

    def test_tau_mi_stricter_filter(self):
        """tau_MI 를 올리면 eligible 이 줄어든다."""
        cands = [_make_candidate() for _ in range(4)]
        m_mi = [3.0, 2.0, 1.5, 1.0]  # 평균 1.875 → 정규화 후 index 0 만 강하게 양수
        scorer = ConstantScorer(value=1.0)

        sel = _make_selector(tau_MI=0.0)
        sel.score_one = _stub_score_one_factory(m_mi, [False] * 4)  # type: ignore
        r1 = sel.select(cands, vla_scorer=scorer)
        eligible_at_zero = set(r1.eligible_indices)

        sel = _make_selector(tau_MI=1.0)  # 정규화 후 1σ 이상만 통과
        sel.score_one = _stub_score_one_factory(m_mi, [False] * 4)  # type: ignore
        r2 = sel.select(cands, vla_scorer=scorer)

        assert set(r2.eligible_indices).issubset(eligible_at_zero)
        assert len(r2.eligible_indices) < len(eligible_at_zero)

    def test_accept_threshold_deprecated_alias(self):
        """yaml 호환: accept_threshold 만 지정해도 tau_MI 로 매핑된다."""
        cfg = Phase2MIConfig(accept_threshold=1.0)
        assert cfg.tau_MI == 1.0

    def test_action_magnitude_scorer_breaks_ties(self):
        """ActionMagnitudeScorer 가 eligible 내에서 더 큰 action 의 후보를 선호한다."""
        sel = _make_selector(tau_MI=-10.0)
        cands = [
            Phase2Candidate(
                skill_id="pick",
                state_keys=np.zeros((2, 4)),
                action_chunks=np.full((2, 3, 2), fill_value=val),
            )
            for val in (0.1, 0.5, 0.9)
        ]
        sel.score_one = _stub_score_one_factory([1.0, 1.0, 1.0], [False] * 3)  # type: ignore

        result = sel.select(cands, vla_scorer=ActionMagnitudeScorer())

        assert result.accepted
        assert result.chosen_index == 2  # action magnitude 가장 큰 것
        assert result.u_vla_chosen == pytest.approx(0.9)


class TestPhase2MIConfigPostInit:
    def test_default_tau_mi_zero(self):
        assert Phase2MIConfig().tau_MI == 0.0
        assert Phase2MIConfig().accept_threshold is None

    def test_accept_threshold_overrides_tau_mi(self):
        cfg = Phase2MIConfig(tau_MI=0.5, accept_threshold=2.0)
        # accept_threshold (alias) 가 우선 — yaml 가 그것만 지정하던 경로 보존.
        assert cfg.tau_MI == 2.0


class TestVlaInformativenessProtocol:
    def test_constant_scorer_returns_same_value(self):
        s = ConstantScorer(value=3.14)
        c = _make_candidate()
        assert s.score(c) == 3.14

    def test_action_magnitude_scorer_deterministic(self):
        s = ActionMagnitudeScorer()
        c = Phase2Candidate(
            skill_id="x",
            state_keys=np.zeros((1, 2)),
            action_chunks=np.array([[[1.0, -2.0, 3.0]]]),  # mean abs = 2.0
        )
        assert s.score(c) == pytest.approx(2.0)
