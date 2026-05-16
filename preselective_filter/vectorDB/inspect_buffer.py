"""Inspect a FAISS vector-DB buffer root — integrity + content check.

Verifies that the per-skill vector DB built up during dataset collection is
well-formed: schema is current, jsonl <-> FAISS index counts agree, key/action
dims are consistent, and nearest-neighbour retrieval actually works.

Usage:
    # direct path
    python -m preselective_filter.vectorDB.inspect_buffer <buffer_root>

    # resolve location from a recording config (transport-aware)
    python -m preselective_filter.vectorDB.inspect_buffer --config <yaml>

With --config the buffer location is resolved from the yaml's
`preselective_filter` section:
  - transport: local -> the buffer is in this process's tree; inspect directly.
  - transport: grpc  -> the buffer lives on the SERVER host. If the server is
                        this machine (localhost address) it is inspected
                        locally; if remote, run this inspector on that host.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .faiss_buffer import _SkillShard

# Keys a current-schema (vector-DB) jsonl line must contain.
_REQUIRED_KEYS = {
    "skill_id", "instruction",
    "state", "state_shape",
    "action_chunk", "action_chunk_shape",
    "key_embedding", "key_embedding_shape",
}
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}


def _scan_schema(path: Path) -> tuple[int, list[str]]:
    """Return (n_nonblank_lines, problems). Problems list is empty when OK."""
    problems: list[str] = []
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {lineno}: invalid JSON ({e})")
                continue
            missing = _REQUIRED_KEYS - obj.keys()
            if missing:
                problems.append(
                    f"line {lineno}: missing keys {sorted(missing)} "
                    f"(has {sorted(obj.keys())})"
                )
    return n, problems


def _check_shard(path: Path) -> bool:
    """Print one jsonl shard's integrity report. Returns True if all OK."""
    skill_id = path.stem
    n_lines, problems = _scan_schema(path)
    print(f"\n  skill '{skill_id}'  —  {n_lines} jsonl lines  ({path.name})")

    if n_lines == 0:
        print("    (empty shard)")
        return True

    if problems:
        print(f"    SCHEMA: {len(problems)} bad line(s) — shard NOT loadable:")
        for p in problems[:3]:
            print(f"      {p}")
        if len(problems) > 3:
            print(f"      ... +{len(problems) - 3} more")
        print("    -> stale/old-schema file. Delete or relocate it before "
              "collecting a fresh dataset (current schema needs 'key_embedding').")
        return False

    # Schema OK -> load via _SkillShard (rebuilds the FAISS index from jsonl).
    shard = _SkillShard(path)
    entries = shard.entries
    n = len(entries)
    ok = True

    ntotal = int(shard.index.ntotal) if shard.index is not None else 0
    idx_ok = ntotal == n == n_lines
    ok &= idx_ok
    print(f"    entries={n}  FAISS index.ntotal={ntotal}  "
          f"{'OK' if idx_ok else 'COUNT MISMATCH'}")

    key_dims = {int(np.asarray(e.context.key_embedding).reshape(-1).shape[0])
                for e in entries}
    key_ok = len(key_dims) == 1
    ok &= key_ok
    print(f"    key_embedding dim  : {sorted(key_dims)}  "
          f"{'OK' if key_ok else 'INCONSISTENT'}")

    act_shapes = {np.asarray(e.action_chunk).shape for e in entries}
    act_ok = len(act_shapes) == 1
    ok &= act_ok
    print(f"    action_chunk shape : {sorted(map(str, act_shapes))}  "
          f"{'OK' if act_ok else 'INCONSISTENT'}")

    instrs = {e.context.instruction for e in entries}
    print(f"    distinct instructions: {len(instrs)}")

    # Retrieval self-query — nearest of entry[0]'s key must be itself (d≈0).
    if shard.index is not None and key_ok:
        q = np.ascontiguousarray(
            entries[0].context.key_embedding, dtype=np.float32,
        ).reshape(1, -1)
        dist, idx = shard.index.search(q, min(3, n))
        self_ok = int(idx[0][0]) == 0 and float(dist[0][0]) < 1e-3
        ok &= self_ok
        print(f"    self-query nearest : idx={idx[0].tolist()} "
              f"dist={[round(float(d), 5) for d in dist[0]]}  "
              f"{'OK' if self_ok else 'index BROKEN'}")

    return ok


def inspect(buffer_root: str | Path) -> int:
    root = Path(buffer_root)
    print(f"=== vector-DB inspect: {root.resolve()} ===")
    if not root.exists():
        print("buffer root does not exist — no episodes committed yet.")
        return 1

    shards = sorted(root.glob("*.jsonl"))
    if not shards:
        print("no *.jsonl shards — buffer is empty "
              "(no TRUE-judged episode committed yet).")
        return 0

    print(f"shards: {len(shards)}")
    all_ok = True
    for path in shards:
        all_ok &= _check_shard(path)

    print(f"\n=== {'ALL OK' if all_ok else 'PROBLEMS FOUND'} ===")
    return 0 if all_ok else 2


def _resolve_from_config(config_path: str | Path) -> tuple[str | None, str, str]:
    """Parse a recording config → (buffer_root|None, transport, address).

    buffer_root is None when the yaml has no explicit `buffer.root` — that is
    the normal session-scoped case (the buffer lives under each run's session
    folder, so the path is only known at run time).
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    psf = cfg.get("preselective_filter") or {}
    root = (psf.get("buffer") or {}).get("root")
    transport = str(psf.get("transport", "local")).lower()
    addr = str(psf.get("transport_address", ""))
    return root, transport, addr


def _latest_session_buffer() -> Path | None:
    """Newest ./results/session_*/preselective_buffer, if any exists."""
    sessions = sorted(Path("./results").glob("session_*"))
    for sess in reversed(sessions):
        cand = sess / "preselective_buffer"
        if cand.exists():
            return cand
    return None


def inspect_from_config(config_path: str | Path) -> int:
    """Transport-aware inspection — decide WHERE the buffer lives, then check."""
    root, transport, addr = _resolve_from_config(config_path)
    print(f"config: {config_path}")
    print(f"transport={transport}  buffer.root={root or '(session-scoped)'}")

    if transport == "grpc":
        host = addr.split(":")[0]
        print(f"grpc server: {addr or '(unset)'}")
        if host in _LOCAL_HOSTS:
            print("server is on THIS host → server buffer is locally accessible.")
        else:
            print(
                f"server is REMOTE ({host}) → the vector DB lives on that host.\n"
                "Run this inspector ON the server with its buffer.root, "
                "or copy the buffer dir over."
            )
        if not root:
            print("note: grpc server buffer.root is server-side config — "
                  "pass that path (or --session) to inspect a specific buffer.")

    if root:
        return inspect(root)

    # Session-scoped (no explicit root) → inspect the newest session's buffer.
    latest = _latest_session_buffer()
    if latest is None:
        print("session-scoped buffer, but no ./results/session_*/preselective_buffer "
              "exists yet — run a collection first, or pass --session <dir>.")
        return 1
    print(f"session-scoped → inspecting newest: {latest}")
    return inspect(latest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_buffer",
        description="Inspect a FAISS vector-DB buffer for integrity.",
    )
    parser.add_argument(
        "buffer_root", nargs="?", default=None,
        help="path to a buffer root (per-skill *.jsonl shards)",
    )
    parser.add_argument(
        "--session", metavar="DIR", default=None,
        help="session folder — inspects <DIR>/preselective_buffer",
    )
    parser.add_argument(
        "--config", metavar="YAML", default=None,
        help="recording config yaml — resolve buffer location transport-aware",
    )
    args = parser.parse_args(argv)

    given = [x for x in (args.buffer_root, args.session, args.config) if x]
    if len(given) > 1:
        parser.error("pass only one of: buffer_root, --session, --config")

    if args.config:
        return inspect_from_config(args.config)
    if args.session:
        return inspect(Path(args.session) / "preselective_buffer")
    return inspect(args.buffer_root or "./results/preselective_buffer")


if __name__ == "__main__":
    raise SystemExit(main())
