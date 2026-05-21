"""method3/dct — skill 단위 trajectory ↔ DCT feature 변환.

사용자 명시 paradigm:
  raw skill trajectory (T_skill, dof) ── cubic resample ──▶ (L0, dof)
                                       ── DCT-II orthonormal ──▶ (L0, dof)

  candidate trajectory     (N, dof) ── 동일 변환 ──▶ (L0, dof)
  VLA target / U_VLA score / VectorDB key 가 모두 같은 DCT space 위에서 정의.
"""
from method3.dct.transform import (
    traj_to_dct,
    dct_to_traj,
)

__all__ = ["traj_to_dct", "dct_to_traj"]
