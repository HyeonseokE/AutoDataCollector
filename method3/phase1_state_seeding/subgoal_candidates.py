"""Subgoal candidate generation for Phase1 buffer-aware subgoal scoring.

문서 final_method3_spec §4.2 / §5.5 — nominal subgoal ``g`` 주변에 K개 후보
``G = {g'_1, ..., g'_K}`` 를 만들고, reachable/safe 하지 않은 후보는 제거한다.

분포 (``dist``):
- ``uniform_ball`` (기본) — 반지름 ``R = clip_factor·sigma`` 인 3D 공 안에서
  **부피 균일** 샘플. Phase1 은 state coverage 증폭이 목적이므로, nominal
  근처에 쏠리지 않고 공 전체에 골고루 퍼진 후보가 선택기에 유리하다.
- ``gaussian`` — truncated isotropic Gaussian (legacy 3D blob; ablation 용).
- ``hemisphere`` — 반지름 ``R`` 인 3D 공의 절반만: nominal goal 에서 로봇 좌표
  원점으로 향하는 단위벡터 ``d`` 쪽 반구에서 부피 균일 샘플. 후보가 로봇
  baseside(=reachable 영역 쪽)로만 생성된다. 기준점은 ``robot_origin`` 인자.

reachable/safe (문서 §4.2 ``G_valid``):
- ``valid_fn`` 이 주어지면 **생성 단계에서 rejection resampling** — invalid 후보는
  버리고 다시 뽑아 K개 valid 를 채운다. generate-then-filter 와 달리 선택기
  메뉴가 줄지 않는다. ``max_resample`` 회 안에 못 찾으면 마지막 샘플을 둔다
  (상위 선택기의 valid 필터가 안전망).

후보 간 non-overlap 제약은 두지 않는다 — "겹치지 않게" 는 상위의 buffer-aware
argmax(이력 대비 novelty)가 담당한다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from method3.phase1_state_seeding.legacy_blob import _sample_truncated_gaussian_3d

CANDIDATE_DISTS = ("uniform_ball", "gaussian", "hemisphere")

# hemisphere dist 에서 방향 d = robot_origin - nominal_goal 의 노름이 이 값
# 이하면 방향이 정의되지 않은 것으로 보고 등방 uniform_ball 로 폴백한다.
_MIN_DIRECTION_NORM = 1e-9


def _sample_uniform_ball_3d(radius: float, rng: np.random.Generator) -> np.ndarray:
    """반지름 ``radius`` 인 3D 공 안에서 부피 균일하게 offset 하나를 샘플.

    방향은 S² 균일(정규분포 정규화), 반지름은 ``r = R·u^{1/3}`` (부피 균일).
    """
    direction = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(direction))
    direction = direction / (norm if norm > 0.0 else 1.0)
    r = radius * (rng.random() ** (1.0 / 3.0))
    return r * direction


def _sample_hemisphere_3d(
    radius: float, direction: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """반지름 ``radius`` 인 3D 반-공(half-ball) 안에서 부피 균일하게 offset 샘플.

    ``direction`` 쪽 반구만 채운다 — S² 균일 방향 ``v`` 를 뽑아 ``direction`` 과
    반대 반구이면 반사한다 (``v → -v``). 반사는 등거리사상이라 measure-
    preserving 이므로 결과는 반구 위 정확히 균일하다. 반경은 uniform_ball 과
    동일하게 부피 균일 (``r = R·u^{1/3}``).
    """
    v = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(v))
    v = v / (norm if norm > 0.0 else 1.0)
    if float(v @ direction) < 0.0:                 # 반대 반구면 접어 올린다
        v = -v
    r = radius * (rng.random() ** (1.0 / 3.0))
    return r * v


def _draw_offset(
    dist: str,
    sigma: float,
    radius: float,
    rng: np.random.Generator,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """``dist`` 에 따라 offset 하나를 샘플한다.

    ``direction`` 은 ``hemisphere`` dist 에서만 쓰인다 (반구 방향 기준 단위벡터).
    """
    if dist == "uniform_ball":
        return _sample_uniform_ball_3d(radius, rng)
    if dist == "hemisphere":
        return _sample_hemisphere_3d(radius, direction, rng)
    return _sample_truncated_gaussian_3d(sigma, radius, rng)


def sample_subgoal_candidates(
    nominal_goal: np.ndarray,
    n_candidates: int,
    sigma: float,
    clip_factor: float,
    rng: np.random.Generator,
    include_nominal: bool = True,
    dist: str = "uniform_ball",
    valid_fn: Callable[[np.ndarray], bool] | None = None,
    max_resample: int = 200,
    robot_origin: np.ndarray | None = None,
) -> np.ndarray:
    """nominal_goal 주변에 K개 candidate subgoal 을 생성한다.

    Args:
        nominal_goal: 현재 nominal subgoal ``g`` — (3,) xyz.
        n_candidates: 후보 개수 K (>= 1).
        sigma: 분포 폭 파라미터 (metres).
        clip_factor: 후보 최대 반경 = ``clip_factor · sigma``.
        rng: 재현 가능한 numpy Generator.
        include_nominal: True 면 candidate 0 = 변형 없는 nominal goal.
            (nominal 은 계획된 타깃이므로 ``valid_fn`` rejection 대상에서 제외.)
        dist: ``"uniform_ball"``, ``"gaussian"``, ``"hemisphere"`` 중 하나.
        valid_fn: ``callable(xyz) -> bool`` — reachable/safe 술어 (문서 §4.2).
            주어지면 생성 단계에서 invalid 후보를 reject & 재샘플한다.
        max_resample: 후보 하나당 재샘플 시도 상한. 초과하면 마지막 샘플을 둔다.
        robot_origin: ``hemisphere`` dist 의 방향 기준점 (로봇 좌표 원점) — (3,).
            후보가 ``nominal_goal`` 에서 이 점으로 향하는 반구 안에만 생성된다.
            None 이면 ``(0,0,0)``. 다른 dist 면 무시. ``nominal_goal`` 과 거의
            일치하면 방향 불능 → ``uniform_ball`` 로 폴백.

    Returns:
        (n_candidates, 3) candidate goal 배열.
    """
    nominal_goal = np.asarray(nominal_goal, dtype=np.float64).reshape(3)
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if dist not in CANDIDATE_DISTS:
        raise ValueError(f"dist must be one of {CANDIDATE_DISTS}, got {dist!r}")

    radius = clip_factor * sigma

    # §4.2 hemisphere — nominal goal → 로봇 원점 방향 d 를 한 번 계산해 모든
    # offset 샘플이 공유한다. g 가 robot_origin 과 거의 일치해 방향이 정의되지
    # 않으면 등방 uniform_ball 로 폴백한다.
    direction: np.ndarray | None = None
    if dist == "hemisphere":
        origin = (np.zeros(3) if robot_origin is None
                  else np.asarray(robot_origin, dtype=np.float64).reshape(3))
        d = origin - nominal_goal
        d_norm = float(np.linalg.norm(d))
        if d_norm <= _MIN_DIRECTION_NORM:
            dist = "uniform_ball"
        else:
            direction = d / d_norm

    out = np.empty((n_candidates, 3), dtype=np.float64)

    start = 0
    if include_nominal:
        out[0] = nominal_goal
        start = 1

    for j in range(start, n_candidates):
        cand = nominal_goal + _draw_offset(dist, sigma, radius, rng, direction)
        if valid_fn is not None:
            # §4.2 — reachable/safe 하지 않은 후보는 reject & 재샘플.
            attempts = 1
            while attempts < max_resample and not bool(valid_fn(cand)):
                cand = nominal_goal + _draw_offset(dist, sigma, radius, rng, direction)
                attempts += 1
        out[j] = cand

    return out
