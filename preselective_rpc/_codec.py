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


def encode_jpeg(arr: np.ndarray, quality: int = 95) -> bytes:
    """JPEG-compress one camera frame for transport.

    Accepts HWC uint8 or CHW/HWC float in [0,1]; normalizes to HWC uint8 then
    encodes. Lossy — fine for frozen-encoder feature keys; the model only sees
    the decoded pixels. encode/decode are inverse regardless of RGB/BGR order.
    """
    import cv2

    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] > 4:
        a = np.transpose(a, (1, 2, 0))               # CHW -> HWC
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if float(a.max()) <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes back to an HWC uint8 array."""
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cv2.imdecode failed")
    return arr
