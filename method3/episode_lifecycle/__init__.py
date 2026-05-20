"""method3 episode lifecycle — episode 단위 삭제·재취득 정합성 (문서 episode lifecycle).

resume 으로 에피소드를 재수집할 때, 삭제된 에피소드의 stale entry 를 subgoal
buffer / raw dataset / vector DB 에서 함께 정리하여 store 들이 데이터셋과 1:1
정합을 유지하게 한다.

| 파일 | 역할 |
|---|---|
| ``episode_ref.py``  | canonical episode_id 규약 (폴더명·buffer·cleanup 단일 출처) |
| ``store_ports.py``  | ``EpisodeScopedStore`` Protocol — episode 단위 제거 인터페이스 |
| ``reconciler.py``   | ``EpisodeReconciler`` — 여러 store 를 일괄 retain/remove |
"""
from method3.episode_lifecycle.episode_ref import episode_id, episode_num
from method3.episode_lifecycle.reconciler import EpisodeReconciler
from method3.episode_lifecycle.store_ports import EpisodeScopedStore

__all__ = ["episode_id", "episode_num", "EpisodeReconciler", "EpisodeScopedStore"]
