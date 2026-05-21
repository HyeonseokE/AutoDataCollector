"""Skill-conditioned instruction formatter — method3 DCT paradigm 일관성.

학습 / DB build / Phase2 inference 모두 *같은 language string* 으로 VLA
encoder 에 들어가야 한다. 본 모듈은 그 single SoT.

학습 시: ``method3/dct/skill_dct_dataset.py`` 가 dataset frame 의 ``task``
column 에 skill_type prefix 를 inject 한 후 model 에 넘긴다. paradigm spec
의 *Inputs: (observation, language, skill_type)* 의 *language slot* 이
실제로는 *prefixed string* 으로 구현된다.

DB build 시: ``method3.reembedding.seed_builder`` / lerobot adapters 가
encoder.encode(obs, instruction, state) 호출 시 본 helper 로 instruction
포맷팅 — 학습 분포와 동일.

Phase2 inference 시: ``grpc_server.server.PlanAndSelect`` 가 *skill_id*
와 *instruction* 받아서 helper 로 합친 후 candidate_gen / VLA scorer 에
전달.

format string 변경 시 *세 곳 모두 자동 sync*.
"""
from __future__ import annotations

DEFAULT_FORMAT = "{skill_type}: {instruction}"


def format_skill_instruction(
    skill_type: str | None,
    instruction: str | None,
    fmt: str = DEFAULT_FORMAT,
) -> str:
    """Return ``{skill_type}: {instruction}`` (or configured ``fmt``).

    - ``skill_type`` 가 비어있으면 prefix 생략하고 instruction 그대로 (legacy
      또는 cold-start safe path).
    - ``instruction`` 가 비어있어도 prefix 만으로 반환 (degenerate but safe).
    """
    s = str(skill_type or "").strip()
    i = str(instruction or "").strip()
    if not s:
        return i
    try:
        return fmt.format(skill_type=s, instruction=i)
    except (KeyError, IndexError):
        # malformed fmt fallback
        return f"{s}: {i}" if i else s
