"""[DEPRECATED] subgoal_buffer.npz 를 dataset 기반으로 재생성 — 사용 금지.

============================================================
DEPRECATED — 2026-05-28
============================================================

이 script 는 episode_id 를 `dataset 의 episode_index + 1` 으로 stamp 한다:

    ep_id = f"episode_{ei + 1:02d}"   # ei = dataset 의 episode_index

이는 buffer 의 ep_id 가 phase1 폴더 이름 (= episode lifecycle 의 SoT) 와
*같다는 가정* 에 의존하지만, 그 가정이 **항상 보장되지 않는다**:
  - dataset record 시 첫 episode_index 가 0 으로 시작 안 할 수도 있음 (resume 등)
  - cleanup_dataset_for_resume 가 episode_index 를 재정렬할 수도 있음
  - dataset 에 phase1 외 다른 episode 가 섞일 수도 있음 (warm-up 등)

결과: buffer 의 ep_id 가 phase1 폴더 와 +1 이상 shift → phase2 replay 시
잘못된 episode 의 subgoal 사용 → trajectory 정합 깨짐.

→ **phase1 폴더 이름 (SoT) 기반 rebuild 를 사용**:

    python -m scripts.rebuild_subgoal_buffer_from_forward_log \
        --session results/completed_logs/<task> \
        --out    results/completed_logs/<task>/subgoal_buffer.npz

이 script 는 호출 시 즉시 실패한다.
============================================================
"""
from __future__ import annotations

import sys


_DEPRECATION_MESSAGE = """
============================================================
DEPRECATED: rebuild_subgoal_buffer_unified.py 는 사용 금지.
============================================================

이 script 는 dataset 의 episode_index 기반으로 buffer ep_id 를 stamp 하지만,
dataset index 와 phase1 폴더 이름 사이 shift 가 발생 가능 → phase2 replay 의
episode mismatch 원인.

대신 phase1 폴더 이름 (SoT) 기반 rebuild 사용:

    python -m scripts.rebuild_subgoal_buffer_from_forward_log \\
        --session results/completed_logs/<task> \\
        --out    results/completed_logs/<task>/subgoal_buffer.npz

참고: 2026-05-28 RCA — buffer ep_01 entry 가 phase1 ep_02 의 데이터 로 stamp
되는 episode shift bug 확인 후 deprecated.
============================================================
"""


def main() -> int:
    print(_DEPRECATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
