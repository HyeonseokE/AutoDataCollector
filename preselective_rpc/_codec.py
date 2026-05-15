"""Wire-format helpers shared by server + client.

Conventions:
- Numpy arrays   → np.save bytes (preserves dtype + shape)
- Image dicts    → pickle of {key: ndarray HWC uint8 OR jpeg bytes}
- Trajectories   → pickle of dict {waypoints, times, algo, cost, seed}

Pickle is fine for trusted server↔client; switch to a typed schema
(e.g., msgpack with explicit fields) if exposing publicly.
"""
from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np


def encode_ndarray(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def decode_ndarray(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def encode_pickle(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode_pickle(data: bytes) -> Any:
    return pickle.loads(data)
