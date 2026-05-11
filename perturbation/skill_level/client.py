"""PlanServiceClient — talks to the OMPL plan daemon over a Unix socket.

Designed to import cleanly from ``lerobot_cap`` (numpy 2.x) without pulling
mplib. Auto-spawns the daemon (in ``mplib_env``) on first use; reuses an
existing daemon if its socket is alive.

Usage::

    from perturbation.skill_level.client import PlanServiceClient

    client = PlanServiceClient(
        urdf="assets/urdf/so101_robot0.urdf",
        config=dict(
            enabled=True,
            algorithms=("RRTConnect", "PRMstar", "BITstar", "KPIECE1"),
            planning_time=0.5,
            waypoint_density=0.02,
            workspace=dict(
                table_enabled=True,
                table_z=0.0,
                table_thickness=0.10,
                table_size=(2.0, 2.0),
            ),
        ),
    )
    cands = client.plan_batch(start_qpos, goal_qpos, n=8, seed=42)
    client.close()

Each candidate is a :class:`TrajectoryCandidate` (defined in
``perturbation.skill_level.planner``) — pure numpy + dataclass, safe to
deserialize without mplib.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from perturbation.skill_level.daemon.protocol import decode_from, encode

# Default daemon python — points at the isolated mplib_env.
# Override via env var PLAN_DAEMON_PYTHON or ctor argument.
DEFAULT_DAEMON_PYTHON = os.environ.get(
    "PLAN_DAEMON_PYTHON",
    "/home/lerobot3/miniconda3/envs/mplib_env/bin/python",
)

# User-isolated socket: avoids collisions when multiple users share the host.
DEFAULT_SOCKET_PATH = f"/tmp/lerobot_planner_{os.getuid()}.sock"

# Spawn / connect timeouts
_SPAWN_TIMEOUT_S = 30.0   # generous: includes mplib first-time SRDF generation
# Per-request recv timeout — bounds how long the client waits on one daemon
# response. Daemon's plan_batch has its OWN wall-budget (planning_time × 2 + 2s)
# and returns empty on internal timeout, so this just needs to cover daemon
# round-trip plus a small slack. 8s is plenty for default 0.5s planning_time.
_RECV_TIMEOUT_S = 8.0


class PlanServiceError(RuntimeError):
    """Raised when the daemon returns an error response."""


class PlanServiceClient:
    def __init__(
        self,
        urdf: str | Path,
        config: dict,
        socket_path: str = DEFAULT_SOCKET_PATH,
        daemon_python: str = DEFAULT_DAEMON_PYTHON,
        n_workers: int = 4,
        autospawn: bool = True,
    ) -> None:
        self.urdf = str(urdf)
        self.config = dict(config)
        self.socket_path = socket_path
        self.daemon_python = daemon_python
        self.n_workers = n_workers
        self._sock: Optional[socket.socket] = None
        self._req_id = 0
        self._daemon_proc: Optional[subprocess.Popen] = None
        self._owns_daemon = False  # True if this client spawned the daemon

        if autospawn and not self._is_daemon_alive():
            self._spawn_daemon()
        self._connect()
        self._call("init", urdf=self.urdf, config_dict=self.config,
                   n_workers=self.n_workers)

    # ── Public API ─────────────────────────────────────────────────────────

    def ping(self) -> str:
        return self._call("ping")

    def plan_batch(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        n: int,
        seed: Optional[int] = None,
    ) -> list:
        """Plan ``n`` parallel candidates. Returns list[TrajectoryCandidate]."""
        return self._call(
            "plan_batch",
            start=np.asarray(start_qpos, dtype=float).tolist(),
            goal=np.asarray(goal_qpos, dtype=float).tolist(),
            n=int(n),
            seed=seed,
        )

    def plan_single(
        self,
        start_qpos: np.ndarray,
        goal_qpos: np.ndarray,
        seed: Optional[int] = None,
    ):
        """Plan a single candidate. Returns TrajectoryCandidate or None."""
        return self._call(
            "plan_single",
            start=np.asarray(start_qpos, dtype=float).tolist(),
            goal=np.asarray(goal_qpos, dtype=float).tolist(),
            seed=seed,
        )

    def update_scene(self, detected: dict) -> dict:
        """Phase 6 placeholder; daemon currently no-ops."""
        return self._call("update_scene", detected=detected)

    def close(self, shutdown_daemon: Optional[bool] = None) -> None:
        """Close socket. If we own the daemon, send shutdown unless overridden."""
        send_shutdown = shutdown_daemon if shutdown_daemon is not None else self._owns_daemon
        if self._sock is not None:
            if send_shutdown:
                try:
                    self._call("shutdown")
                except Exception:
                    pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._daemon_proc is not None and send_shutdown:
            try:
                self._daemon_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._daemon_proc.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Internals ──────────────────────────────────────────────────────────

    def _is_daemon_alive(self) -> bool:
        if not os.path.exists(self.socket_path):
            return False
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(self.socket_path)
            s.close()
            return True
        except (OSError, socket.timeout):
            # Stale socket file (daemon crashed) — clean it up so spawn works.
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
            return False

    def _spawn_daemon(self) -> None:
        if not Path(self.daemon_python).exists():
            raise PlanServiceError(
                f"daemon python not found: {self.daemon_python} "
                f"(set PLAN_DAEMON_PYTHON env var)"
            )
        cmd = [
            self.daemon_python, "-m", "perturbation.skill_level.daemon.server",
            "--socket", self.socket_path,
        ]
        # Run from project root so ``-m perturbation.skill_level...`` resolves.
        project_root = Path(__file__).resolve().parents[2]
        self._daemon_proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._owns_daemon = True

        # Wait for socket to appear and accept connections.
        deadline = time.time() + _SPAWN_TIMEOUT_S
        while time.time() < deadline:
            if self._daemon_proc.poll() is not None:
                # Daemon exited — read stderr to surface error.
                out = self._daemon_proc.stdout.read().decode("utf-8", errors="replace")
                raise PlanServiceError(
                    f"daemon exited (code={self._daemon_proc.returncode}) before "
                    f"accepting connections.\n--- daemon output ---\n{out}"
                )
            if self._is_daemon_alive():
                return
            time.sleep(0.1)
        raise PlanServiceError(
            f"daemon did not bind socket within {_SPAWN_TIMEOUT_S}s: {self.socket_path}"
        )

    def _connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(_RECV_TIMEOUT_S)
        s.connect(self.socket_path)
        self._sock = s

    def _call(self, op: str, **args) -> Any:
        if self._sock is None:
            raise PlanServiceError("client not connected")
        self._req_id += 1
        req = {"id": self._req_id, "op": op, "args": args}
        try:
            self._sock.sendall(encode(req))
            resp = decode_from(self._sock)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise PlanServiceError(f"socket error during {op!r}: {e}") from e
        if resp is None:
            raise PlanServiceError(f"daemon closed connection during {op!r}")
        if not resp.get("ok"):
            raise PlanServiceError(f"daemon error on {op!r}: {resp.get('error')}")
        return resp.get("result")
