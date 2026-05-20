"""Canonical episode 식별자 — 문서 episode lifecycle.

``results/<session>/episode_NN`` 폴더명, subgoal buffer entry 의 ``episode_id``,
``record_dataset.cleanup`` 의 ``ep_num`` 이 모두 같은 식별자 규약을 쓰도록 하는
단일 출처. 에피소드를 삭제하고 resume 으로 재취득할 때, 어느 store 의 어느 entry
가 어느 episode 에 속하는지 일관되게 매칭하려면 식별자 규약이 한 곳에 있어야 한다.
"""
from __future__ import annotations


def episode_id(episode_num: int) -> str:
    """episode 번호(1-base) → canonical episode_id 문자열.

    ``results/<session>/episode_NN`` 폴더명과 동일한 규약 (2자리 zero-pad).
    예) 39 → ``"episode_39"``, 5 → ``"episode_05"``.
    """
    return f"episode_{int(episode_num):02d}"


def episode_num(episode_id_str: str) -> int:
    """canonical episode_id → episode 번호(1-base). ``episode_id`` 의 역함수."""
    return int(str(episode_id_str).rsplit("_", 1)[1])
