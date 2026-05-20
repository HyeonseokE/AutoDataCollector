"""Episode-scoped store 포트 — 문서 episode lifecycle.

episode 단위 정리(삭제·재취득)를 지원하는 store 의 공통 인터페이스. subgoal
buffer / raw dataset / vector DB 가 모두 이 Protocol 을 구조적으로 구현하면,
``EpisodeReconciler`` 가 store 종류에 무관하게 균일하게 정리할 수 있다.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class EpisodeScopedStore(Protocol):
    """episode_id 로 entry 를 식별·제거할 수 있는 store (duck-typed)."""

    def remove_episode(self, episode_id: str) -> int:
        """``episode_id`` 에 속한 entry 를 모두 제거하고 제거 수를 반환한다."""
        ...

    def retain_episodes(self, episode_ids: Iterable[str]) -> int:
        """주어진 episode_id 집합 밖의 entry 를 모두 제거하고 제거 수를 반환한다."""
        ...
