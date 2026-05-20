"""Phase2 seed anchor accessor — phase1_seed_anchor_logic / useful_ood_updated §7.

Phase1 이 누적한 skill-wise subgoal buffer ``B_{g,t}^{(m)}`` 에서 seed subgoal
집합 ``G_seed^{(m)} = {g_1^{(m)}, …, g_N^{(m)}}`` 만 얇게 노출한다. Phase2 는
새 subgoal 을 탐색하지 않고 이 집합을 anchor 로 쓴다 (§7 box).

본 모듈의 책임:
  * ``load_g_seed(session_dir) -> SubgoalBuffer`` — npz 파일을 로드.
  * ``pick_anchor(buffer, skill_id, target_xyz=None) -> g | None`` — Phase2
    candidate generator 가 쓸 anchor 한 점을 결정.

Anchor 결정 정책 (가장 단순):
  * ``target_xyz`` 주어지면 → ``G_seed^{(m)}`` 내 nearest neighbor 반환.
    skill 의 현재 goal 이 이미 Phase1 seed 와 가까우면 그 seed 가 곧 anchor.
  * ``target_xyz=None`` → buffer 의 첫 entry 반환 (deterministic).

더 정교한 정책 (random sampling, coverage-aware) 는 caller 가 ``buffer``
직접 만져 결정. 본 모듈은 default 만 책임.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from method3.phase1_state_seeding.subgoal_buffer import SubgoalBuffer


def load_g_seed(session_dir: str | Path, filename: str = "subgoal_buffer.npz") -> SubgoalBuffer:
    """Phase1 의 ``subgoal_buffer.npz`` 를 로드해 SubgoalBuffer 로 반환.

    파일이 없거나 비어 있으면 빈 buffer 를 반환 (caller 가 ``len(buffer)==0``
    으로 분기). exception 은 던지지 않는다 — phase2 setup 이 보호적이어야 함.
    """
    path = Path(session_dir) / filename
    if not path.exists():
        return SubgoalBuffer()
    try:
        return SubgoalBuffer.load(str(path))
    except Exception:
        return SubgoalBuffer()


def pick_anchor(
    buffer: SubgoalBuffer,
    skill_id: str,
    target_xyz: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """skill ``m`` 에서 Phase2 candidate 생성에 쓸 anchor ``g`` 한 점.

    Args:
        buffer: Phase1 누적 SubgoalBuffer.
        skill_id: 현재 skill m.
        target_xyz: skill 의 의도된 goal xyz. 주어지면 G_seed^{(m)} 내 nearest
            neighbor 를 anchor 로 반환 — Phase1 이 *이미 본 적 있는* target
            중 가장 가까운 것을 골라 그 주위 action variation 만 한다.

    Returns:
        ``g ∈ R^3`` (xyz). G_seed^{(m)} 이 비어 있으면 None.
    """
    seeds = buffer.seed_subgoals(skill_id)
    if seeds.size == 0 or seeds.shape[0] == 0:
        return None
    if target_xyz is None:
        return np.asarray(seeds[0], dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    d2 = np.sum((seeds - target[None, :]) ** 2, axis=1)
    return np.asarray(seeds[int(np.argmin(d2))], dtype=np.float64).reshape(3)
