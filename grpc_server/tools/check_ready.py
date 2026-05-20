#!/usr/bin/env python
"""grpc_server 서버 readiness 점검.

서버에 Ready RPC 를 보내고 응답(체크포인트/디바이스/버퍼 상태 등)을 출력한다.

실행 (AutoDataCollector 프로젝트 루트에서):
    python -m grpc_server.tools.check_ready
    python -m grpc_server.tools.check_ready --address 127.0.0.1:50061 --timeout 10
"""

from __future__ import annotations

import argparse
import pprint
import sys

from grpc_server.client import PreselectiveClient


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--address", default="127.0.0.1:50061", help="서버 gRPC 주소")
    p.add_argument("--timeout", type=float, default=10.0, help="응답 타임아웃(초)")
    args = p.parse_args()

    client = PreselectiveClient(args.address, timeout_s=args.timeout)
    try:
        pprint.pprint(client.ready())
    except Exception as e:  # noqa: BLE001 — 점검 도구: 모든 실패를 한 줄로 보고
        print(f"XX 서버 응답 실패 ({args.address}): {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
