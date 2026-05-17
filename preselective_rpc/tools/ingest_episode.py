#!/usr/bin/env python
"""preselective_rpc 서버에 에피소드 하나를 ingest (vector-DB 성장).

에피소드를 로컬에서 읽어(CPU only) 프레임별로 서버에 스트리밍하면, 서버가
frozen VLA 로 key_t 를 인코딩해 버퍼에 (key_t, value_t) 항목을 추가한다.
반환값: {frames, buffer_totals, episode}.

실행 (AutoDataCollector 프로젝트 루트에서):
    python -m preselective_rpc.tools.ingest_episode --dataset-root /path/to/dataset
"""

from __future__ import annotations

import argparse
import pprint
import sys

from preselective_rpc.client import PreselectiveClient


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--address", default="127.0.0.1:50061", help="서버 gRPC 주소")
    p.add_argument("--dataset-root", required=True, help="LeRobot 데이터셋 경로")
    p.add_argument("--chunk-size", type=int, default=50, help="action chunk 크기")
    p.add_argument("--skill-id", default="move_to", help="스킬 ID")
    p.add_argument("--episode-index", type=int, default=None, help="ingest 할 에피소드 인덱스")
    p.add_argument("--max-frames", type=int, default=None, help="스트리밍할 최대 프레임 수 (빠른 점검용)")
    p.add_argument("--timeout", type=float, default=600.0, help="RPC 타임아웃(초)")
    args = p.parse_args()

    client = PreselectiveClient(args.address, timeout_s=args.timeout)
    try:
        result = client.ingest_episode(
            dataset_root=args.dataset_root,
            chunk_size=args.chunk_size,
            skill_id=args.skill_id,
            episode_index=args.episode_index,
            timeout_s=args.timeout,
            max_frames=args.max_frames,
        )
        pprint.pprint(result)
    except Exception as e:  # noqa: BLE001 — 점검 도구: 모든 실패를 한 줄로 보고
        print(f"XX ingest 실패 ({args.address}): {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
