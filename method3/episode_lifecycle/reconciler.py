"""Episode reconciler — 문서 episode lifecycle.

여러 ``EpisodeScopedStore`` 를 한 번에 episode 단위로 정리한다.

  - ``retain_episodes`` : resume 시 생존(judge=TRUE·폴더 존재) 에피소드 집합으로
    모든 store 를 맞춘다. 삭제된 에피소드의 stale entry 가 함께 제거된다.
  - ``remove_episode``  : 특정 에피소드를 모든 store 에서 제거한다.

store 가 ``subgoal buffer`` 든 ``raw dataset`` 이든 ``vector DB`` 든, 모두
``EpisodeScopedStore`` 만 만족하면 동일하게 다룬다 — store 가 늘어도 등록만 하면
된다.
"""
from __future__ import annotations

from typing import Iterable

from method3.episode_lifecycle.store_ports import EpisodeScopedStore


class EpisodeReconciler:
    """등록된 store 들을 episode 단위로 일괄 정리한다."""

    def __init__(self, stores: Iterable[EpisodeScopedStore | None]) -> None:
        """
        Args:
            stores: 정리 대상 store 목록. ``None`` 은 무시한다 (아직 wiring 안 된
                store 를 자리만 잡아둘 때 편하도록).
        """
        self._stores = [s for s in stores if s is not None]

    def remove_episode(self, episode_id: str) -> dict[str, int]:
        """모든 store 에서 ``episode_id`` 를 제거한다. ``store명 → 제거 수``."""
        return {
            f"{type(s).__name__}#{i}": s.remove_episode(episode_id)
            for i, s in enumerate(self._stores)
        }

    def retain_episodes(self, episode_ids: Iterable[str]) -> dict[str, int]:
        """모든 store 를 생존 episode 집합에 맞춰 정리한다. ``store명 → 제거 수``."""
        keep = list(episode_ids)
        return {
            f"{type(s).__name__}#{i}": s.retain_episodes(keep)
            for i, s in enumerate(self._stores)
        }
