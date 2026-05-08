"""Plan daemon. Runs inside ``mplib_env``. Listens on a Unix domain socket,
serves plan requests using ParallelEnsemble.

Lifecycle:
  - Spawned by PlanServiceClient via ``python -m perturbation.skill_level.daemon.server``
  - Listens forever; exits on SIGTERM/SIGINT, on the ``shutdown`` request, or
    when its parent (lerobot_cap python) dies (PR_SET_PDEATHSIG on Linux).
  - One daemon per Unix socket path. Multi-robot setups use separate sockets.

Operations (selected via request ``op`` field):
    init         → set up the underlying ParallelEnsemble (URDF + workspace).
    ping         → returns "pong"; used by client to test liveness.
    plan_batch   → run N parallel OMPL plans, return list[TrajectoryCandidate].
    plan_single  → single plan, returns one TrajectoryCandidate or None.
    update_scene → (Phase 6 placeholder) push detected obstacles into world.
    shutdown     → stop accepting requests and exit.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make the project importable when running ``python -m`` from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from perturbation.skill_level.daemon.protocol import encode, decode_from
from perturbation.skill_level.planner import PlannerEnsembleConfig
from perturbation.skill_level.parallel import ParallelEnsemble


def _arm_pdeathsig() -> None:
    """Linux: arrange for the daemon to die when its parent exits.
    No-op on non-Linux. Uses prctl(PR_SET_PDEATHSIG, SIGTERM)."""
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


class PlanDaemon:
    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self.planner: Optional[ParallelEnsemble] = None
        self.running = True
        self._server_sock: Optional[socket.socket] = None
        # ParallelEnsemble worker pool isn't safe to call from concurrent
        # threads (the multiprocessing.Pool is shared mutable state). Serialize
        # plan handlers across connections with this lock. ping/init/shutdown
        # are cheap so the lock barely matters.
        self._planner_lock = threading.Lock()
        self._client_threads: list[threading.Thread] = []
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        _arm_pdeathsig()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def serve(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server_sock.listen(2)
        self._log(f"listening on {self.socket_path} (pid={os.getpid()})")

        while self.running:
            try:
                self._server_sock.settimeout(1.0)
                try:
                    conn, _ = self._server_sock.accept()
                except socket.timeout:
                    continue
            except OSError:
                if not self.running:
                    break
                raise
            # Spawn a thread per connection so a slow / crashed client doesn't
            # block other clients from connecting. Plan handlers are serialized
            # by self._planner_lock since the underlying Pool isn't thread-safe.
            t = threading.Thread(
                target=self._client_thread_main, args=(conn,), daemon=True,
            )
            t.start()
            self._client_threads.append(t)
            # Reap finished threads to keep the list bounded.
            self._client_threads = [tt for tt in self._client_threads if tt.is_alive()]
        self._cleanup()

    def _client_thread_main(self, conn: socket.socket) -> None:
        try:
            self._handle_client(conn)
        except Exception:
            self._log(f"client thread crash:\n{traceback.format_exc()}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_client(self, conn: socket.socket) -> None:
        while self.running:
            try:
                req = decode_from(conn)
            except Exception:
                self._log(f"decode error:\n{traceback.format_exc()}")
                return
            if req is None:
                return
            resp = self._dispatch(req)
            try:
                conn.sendall(encode(resp))
            except (BrokenPipeError, ConnectionResetError):
                return

    def _dispatch(self, req: dict) -> dict:
        op = req.get("op")
        rid = req.get("id", 0)
        args = req.get("args", {}) or {}
        try:
            # Plan-touching ops use the planner lock; lightweight ops don't.
            if op in ("init", "plan_batch", "plan_single", "update_scene"):
                with self._planner_lock:
                    if op == "init":
                        return {"id": rid, "ok": True, "result": self._init(**args)}
                    if op == "plan_batch":
                        return {"id": rid, "ok": True, "result": self._plan_batch(**args)}
                    if op == "plan_single":
                        return {"id": rid, "ok": True, "result": self._plan_single(**args)}
                    if op == "update_scene":
                        return {"id": rid, "ok": True, "result": self._update_scene(**args)}
            if op == "ping":
                return {"id": rid, "ok": True, "result": "pong"}
            if op == "shutdown":
                self.running = False
                return {"id": rid, "ok": True, "result": "bye"}
            return {"id": rid, "ok": False, "error": f"unknown op: {op!r}"}
        except Exception as e:
            return {
                "id": rid, "ok": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }

    # ── Request handlers ──────────────────────────────────────────────────

    def _init(self, urdf: str, config_dict: dict, n_workers: int = 4) -> dict:
        if self.planner is not None:
            self.planner.close()
        cfg = PlannerEnsembleConfig(**config_dict)
        self.planner = ParallelEnsemble(urdf=urdf, config=cfg, n_workers=n_workers)
        self._log(f"initialized: urdf={urdf} workers={n_workers}")
        return {"n_workers": n_workers, "algorithms": list(cfg.algorithms)}

    def _plan_batch(
        self, start, goal, n: int, seed: Optional[int] = None
    ) -> list:
        self._require_planner()
        rng = np.random.default_rng(seed) if seed is not None else None
        cands = self.planner.plan_batch(
            np.asarray(start, dtype=float),
            np.asarray(goal, dtype=float),
            n=n,
            rng=rng,
        )
        return list(cands)

    def _plan_single(self, start, goal, seed: Optional[int] = None) -> Optional[Any]:
        cands = self._plan_batch(start=start, goal=goal, n=1, seed=seed)
        return cands[0] if cands else None

    def _update_scene(self, detected: dict) -> dict:
        """Refresh the dynamic part of the collision world.

        Strategy: each detected object is registered as a Box centered at its
        position. Static workspace objects (workspace_table, ceiling, other
        arms) are preserved untouched — we only add/remove objects whose name
        starts with the dynamic prefix ``scene__``.

        Args:
            detected: dict[str, dict]. Each value MUST contain at least
                {"position": [x, y, z]} and optionally
                {"size": [sx, sy, sz]} (defaults to a 5cm cube).

        Returns:
            dict with 'added', 'removed', 'kept' counts.
        """
        self._require_planner()
        # ParallelEnsemble doesn't expose direct planning_world handle, but each
        # worker has its own. Forward the update to the workers via a special
        # internal task that mutates each worker's _W_PLANNER's planning_world.
        # We use the existing pool to broadcast.
        pool = self.planner._pool
        if pool is None:
            return {"error": "planner pool closed"}

        # Build the per-worker payload.
        payload = []
        for name, info in (detected or {}).items():
            pos = info.get("position")
            if pos is None or len(pos) < 3:
                continue
            size = info.get("size", [0.05, 0.05, 0.05])
            payload.append((name, list(pos[:3]), list(size[:3])))

        # Each worker runs _worker_update_scene with the same payload.
        # multiprocessing.Pool.map runs the task on N workers (one each).
        from perturbation.skill_level.parallel import _worker_update_scene
        n = self.planner.n_workers
        results = pool.map(_worker_update_scene, [payload] * n)
        # Aggregate (all workers produce the same numbers if successful)
        rep = results[0] if results else {"added": 0, "removed": 0, "kept": 0}
        self._log(
            f"update_scene: +{rep.get('added',0)} -{rep.get('removed',0)} "
            f"={rep.get('kept',0)} (n_workers updated: {sum(1 for r in results if r)})"
        )
        return rep

    # ── Helpers ───────────────────────────────────────────────────────────

    def _require_planner(self) -> None:
        if self.planner is None:
            raise RuntimeError("daemon not initialized — call op='init' first")

    def _on_signal(self, signum, frame) -> None:
        self._log(f"received signal {signum}; shutting down")
        self.running = False

    def _cleanup(self) -> None:
        if self.planner is not None:
            try:
                self.planner.close()
            except Exception:
                pass
            self.planner = None
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass
        self._log("clean exit")

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[plan-daemon] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="OMPL plan daemon")
    ap.add_argument("--socket", required=True, help="Unix socket path to bind")
    args = ap.parse_args()
    PlanDaemon(args.socket).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
