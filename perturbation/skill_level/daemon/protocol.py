"""Wire protocol between PlanServiceClient (lerobot_cap) and PlanDaemon (mplib_env).

Frame: 4-byte big-endian uint32 length + pickle-serialized payload.

Why pickle: payloads include numpy arrays + dataclasses; pickle is the standard
single-package solution. (msgpack-numpy could be plugged in later if profiling
shows benefit; payloads are typically <200KB so pickle overhead is ~1-2ms.)

Why uint32 BE: trivially compatible. Max 4GB, ~10⁵× more than any plan reply.

Request schema (Python dict, pickled):
    {"id": int, "op": str, "args": dict}
        op ∈ {"init", "ping", "plan_batch", "plan_single", "shutdown"}
Response schema (pickled):
    {"id": int, "ok": bool, "result": Any | None, "error": str | None}
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Optional

LEN_HEADER_FMT = ">I"
LEN_HEADER_SIZE = 4


def encode(obj: Any) -> bytes:
    """Serialize an object to a length-prefixed pickle frame."""
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack(LEN_HEADER_FMT, len(payload)) + payload


def decode_from(sock: socket.socket) -> Optional[Any]:
    """Read one framed message from a socket. Returns the decoded object,
    or ``None`` on clean EOF (peer closed)."""
    header = _recv_exact(sock, LEN_HEADER_SIZE)
    if not header:
        return None
    (n,) = struct.unpack(LEN_HEADER_FMT, header)
    if n == 0:
        return None
    payload = _recv_exact(sock, n)
    if not payload:
        return None
    return pickle.loads(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes. Returns ``b""`` on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)
