# Curobo Skill-level Perturbation — Methodology & Design

> Reference for the GPU-accelerated trajectory perturbation backend
> introduced in commits `780ed41 … 68144d0` (master, 2026-05).
> Companion to [`curobo_backend.py`](curobo_backend.py).

## 1. Problem & motivation

The data-collection pipeline records **transit moves** (point-A → point-B
no-contact arm motions) and wants every recorded transit to look slightly
different so the downstream policy learns robust execution rather than
memorising one canonical path. There are two orthogonal perturbation
levels in the codebase:

| Level | What it perturbs | Magnitude | File |
|---|---|---|---|
| **subgoal** | the **target xyz** of `move_to_position` | σ ≈ 5 cm gaussian | `perturbation/subgoal/...` |
| **skill** (this doc) | the **trajectory shape** between start and target | 7–15 cm arcs | `perturbation/skill_level/` |

Both layers fire in production. This document covers only the skill layer.

The skill layer uses curobo as the planning backend, exposing
`plan_batch(start_qpos, goal_qpos, n, seed=...) → List[TrajectoryCandidate]`.
The gRPC adapter (`vla_adaptor/grpc_planner_adapter.py`) duck-types the
same interface so the local in-process backend and the remote H100
server are swappable from yaml. This doc describes the curobo backend.

## 2. Algorithmic core — multi-via, random K, continuous sampling

The curobo backend's diversity comes from **via-point sampling**. Each
plan_batch call returns `n_candidates` trajectory variants:

```
slot 0       : direct  — start → goal in one segment (baseline)
slots 1..n-1 : via    — start → v_1 → … → v_K → goal,
                       K ∈ {1, …, K_max} sampled per slot
```

### Per-candidate K (number of vias)

`max_vias_per_candidate` (`K_max`) is a config knob (`1` default,
`2` enabled in production):

```
slot 0:   K = 0          (always — one direct baseline per batch)
slot ≥1:  K ~ Uniform({1, …, K_max})   (never zero — guaranteed diversity)
```

For `K_max=2` with N=4 the distribution per call is:
- 1 direct + 3 vias, each via slot independently flipping K ∈ {1, 2}

### Per-via parameters (continuous sampling)

Each via location is parameterised as

```
via = (1 − t)·start_ee + t·goal_ee + lat·perp_lat + vert·perp_vert
```

with three independent continuous samples:

| param | distribution | meaning |
|---|---|---|
| `t` | Uniform(0.2, 0.8), **stratified** when K>1 | fraction along start→goal line |
| `lat` | Uniform(−mag, +mag) | perpendicular lateral displacement |
| `vert` | Uniform(0.3·mag, 1.2·mag) | vertical displacement, biased up |

`mag = config.via_offset_mag` (yaml: `curobo_via_offset_mag`, default 0.10 m).

**Stratified t** (when K ≥ 2): split [0.2, 0.8] into K equal subintervals
and draw one t from each. Guarantees min spacing (1/K of range) between
consecutive vias → avoids the degenerate "two vias on top of each
other" collapse.

**Why uniform up-bias on vert**: SO-101's 5-DoF arm reaches *above*
the start-goal line far more reliably than *below* (table proximity +
elbow-up kinematic preference). Empirically, vias sampled below the line
fail IK ~60% of the time on this robot.

### Replacing the old discrete VIA_OFFSETS table

The pre-2026-05 design used a hardcoded 4-entry table of (line, lat, vert)
preset offsets. With N>4 candidates the slots wrapped modulo and produced
near-identical paths. The continuous sampling above provides **O(N)**
spatial diversity instead of O(4).

## 3. Architecture & control flow

### 3.1 Backend dispatch (in `execution_forward_and_reset.py`)

```
yaml.preselective_filter.transport
    │
    ├── "local" → CuroboBackend(urdf, cfg)
    │              ↑ in-process GPU planner, atexit-cleaned
    │
    └── "grpc"  → GrpcPlannerClient(PreselectiveClient(addr), provider)
                   ↑ remote planning + IG·AC selection on H100
```

Both implementations expose:

```python
plan_batch(start_qpos, goal_qpos, n, seed=int) → List[TrajectoryCandidate]
close()
```

so `skills_lerobot.set_skill_planner_client(client)` is the same call
regardless of transport.

### 3.2 Production call site (in `skills/skills_lerobot.py:1787`)

Skill perturbation fires only when **all four** conditions hold:

```python
if (is_transit                                     # 1. caller marks move as transit
    and self._skill_planner_client is not None     # 2. backend attached (yaml.enabled)
    and self._perturbation_rng is not None         # 3. per-episode RNG set
    and trajectory.ik_converged):                  # 4. cartesian IK reached goal
    cands = client.plan_batch(start, goal, n=4, seed=...)
    if cands:
        chosen = rng.choice(cands)                 # RandomSelector
        # ... overrides cartesian trajectory with chosen.waypoints
```

If `cands` is empty (planner failure), the original cartesian-line
trajectory is used. Pick / place descents are `is_transit=False` so they
*never* go through skill perturbation — only transits get curved paths.

### 3.3 Per-plan_batch GPU layout

For N candidates with K_max vias:

```
[step]   Stage                                  Batch dim
─────────────────────────────────────────────────────────
[1] FK   start + goal kinematics                B = 2
[2] IK   all real vias × goalset orientations   B = N·K_max, G = 5 slerp
[3] plan_cspace                                  B = N·(K_max+1)
[4] CPU  per-candidate concat + spline smooth    —
```

All three GPU stages run as **one kernel each** thanks to CUDA graph
capture (see §4). Plan-time wall on RTX 2080 Ti is ~120 ms for N=4,
K_max=2 after the first call's ~5-6 s graph compile.

## 4. Performance optimisations

A chronological summary of the major optimisations applied to the
backend, with the empirical wall-time impact each one had on the
"plan-cached" path (N=4, this robot, default goal pose):

| # | Optimisation | Before | After | Δ |
|---|---|---|---|---|
| 1 | Initial in-process curobo (no CUDA graph) | ~10 s | ~10 s | — |
| 2 | Enable `cuda_graph_reset` runtime flag → graph capture working | 10 s | 273 ms | **35×** |
| 3 | Batched goalset IK (B=n_via, G=5 slerp orientations) | 273 ms | 203 ms | 70 ms saved |
| 4 | Merged seg1+seg2 into ONE plan_cspace batch (B = 2N) | 203 ms | 143 ms | 60 ms saved |
| 5 | Multi-via random K (K_max=2 mode) | 143 ms | 120 ms | depends on K |
| | Continuous via sampling (replaces discrete table) | — | — | diversity ↑ |

### 4.1 CUDA graph activation (commit `780ed41`)

curobo defaults `_src.runtime.cuda_graph_reset = False`. With it off,
`is_cuda_graph_reset_available()` returns False on all CUDA versions, so
the first batch shape captured locks the graph forever and any later
mismatch errors out. The backend flips this flag at top of `__init__`
*before* MotionPlanner is constructed:

```python
if config.use_cuda_graph:
    import curobo.runtime as _rt
    _rt.cuda_graph_reset = True
```

First plan_batch call now pays a one-time ~5-6 s capture cost and every
subsequent call reuses the graph → 35× speedup.

### 4.2 Goalset-based batched IK

5-DoF SO-101 cannot satisfy arbitrary (xyz, quat) pairs — the orientation
must be slid along the slerp arc from start_quat to goal_quat until the
IK converges. Earlier versions retried orientations sequentially.

Now the orientation retries are folded into curobo's native **goalset**
dimension `G`:

```
B = n_via              ← distinct via xyz queries (rows)
G = n_orient = 5       ← (0.5, 0.3, 0.7, 0.0, 1.0) slerp ratios (cols)

pos_t  : (B, 1, 1, G, 3)   ← xyz broadcast across G
quat_t : (B, 1, 1, G, 4)   ← slerp(start, goal, t) for each G
```

Curobo's IK solver picks the most-reachable orientation per batch item
internally. Strictly stronger than CPU-side first-success picking — it
chooses the *best* fit, not just the first that converged. Wall-time win
is ~6× (90 ms sequential → 15 ms batched).

`max_goalset` must be passed to `MotionPlannerCfg.create` to enable
G > 1 (default is 1, which silently fails as "num_goalset=5 exceeds
config.max_goalset=1").

### 4.3 Merged seg1 + seg2 plan_cspace

A K-via trajectory consists of K+1 segments. Naively, you'd run K+1
separate `plan_cspace` calls — but each call has its own GPU launch
overhead and CUDA graph slot.

Instead, **all K+1 segments × N candidates are packed into one
plan_cspace batch**:

```
_cspace_batch = N × (K_max + 1)

Layout (row-major):
  candidate i occupies rows [i·(K_max+1), (i+1)·(K_max+1) )
  rows 0..K_real      = real segments (start→v_1, v_1→v_2, …, v_K→goal)
  rows K_real+1..K_max = dummy goal→goal no-ops (discarded after)
```

Padding to a fixed batch size keeps the CUDA graph shape constant across
calls where individual candidates have different K (e.g., one call with
4 direct + 0 via, next with 1 direct + 2 via1 + 1 via2). Dummy no-ops
take negligible time on the GPU.

### 4.4 Batched FK (start+goal in one compute_kinematics)

Trivial but worth mentioning: rather than two separate FK calls for
start_ee and goal_ee, batch both into one `compute_kinematics(B=2)`. Saves
~3 ms per plan_batch.

### 4.5 Tensor transfer path

`list-of-list → torch.tensor(device=cuda)` was replaced with
`np.asarray → torch.from_numpy → .to(cuda)`. One contiguous host→device
copy instead of N small per-row copies. Marginal but free.

## 5. CUDA graph stability — padding strategy

Curobo's CUDA graphs capture the **exact batch shape** of the first call.
Any later call with a different shape either crashes or re-captures
(slow). The backend defends against this with three padding layers:

1. **IK**: `_ik_solve_batched` pads the input list to `_ik_batch =
   N·max(1, K_max)` slots before calling curobo. Dummy queries copy the
   first real via xyz (cheap, always solvable). Results truncated back
   to the real n_via on return.

2. **plan_cspace**: `plan_batch` pads `merged_start / merged_goal` to
   `_cspace_batch = N · (K_max + 1)`. Tail dummies are
   `start_full → start_full` no-ops.

3. **Inside curobo**: the `MotionPlannerCfg.create(max_batch_size=…)` is
   set to the larger of `_cspace_batch` and `_ik_batch`. Curobo
   internally allocates buffers for that batch size on init.

Net effect: every plan_batch call after the first runs at identical
batch dimensions, and the CUDA graph never re-captures.

## 6. Lifecycle / GPU memory cleanup

CuroboBackend reserves ~800 MB GPU during a session (CUDA graph private
pool + trajopt internal state + collision distance fields). Three
cleanup pathways were added across commits `e50e167`, `3309dac`:

```
Normal pipeline shutdown
    → pipeline._teardown_skill_perturbation()
    → client.close()                          ✓

Ctrl+C after recorder init
    → pipeline SIGINT handler
    → _finalize_recording() → close()         ✓

Ctrl+C before recorder init,
SIGTERM, or unhandled exception
    → Python interpreter shutdown
    → atexit.register'd self._atexit_close()
    → close()                                 ✓

SIGKILL (`kill -9`)
    → not handleable by user space
    → CUDA driver reclaims at process death   (OS, not us)
```

`close()` is **idempotent** (`self._closed` flag) so the pipeline path
and atexit path can both fire safely without double-free.

`close()` drops references to `_planner`, `_default_full`, `_interp_dt`
and calls `torch.cuda.empty_cache() + synchronize()`. Empirical:

```
after init:        allocated=    5   reserved=    6 MiB
after plan_batch:  allocated=  806   reserved= 1350 MiB
after close+gc:    allocated=    0   reserved=  692 MiB
                                            ↑
                          PyTorch CUDA context — only released on exit
```

The 692 MiB residue is the PyTorch CUDA context baseline and can be
re-used by other tensors in the same process. Only process exit releases
it; there is no per-process API to drop it.

## 7. Diversity mechanisms — summary table

The backend layers four orthogonal RNG sources to produce diverse
trajectories. Each row controls a different axis:

| Knob | Range | Effect | Notes |
|---|---|---|---|
| K_via per slot | {1, …, K_max} uniform | path topology (1-arc vs 2-arc) | random across slots in one batch |
| t per via | Uniform(0.2, 0.8), stratified | *where* on the line the via lands | min spacing enforced for K>1 |
| lat per via | Uniform(−mag, +mag) | left/right swing | symmetric, expected \|lat\| = mag/2 |
| vert per via | Uniform(0.3·mag, 1.2·mag) | upward arc height | biased up for reachability |
| slerp t per IK retry | (0.5, 0.3, 0.7, 0.0, 1.0) | orientation along start↔goal arc | not RNG — fixed grid, IK picks best |

Plus the **RandomSelector** in `skills_lerobot.py` which picks 1 of the
N candidates uniformly. Final diversity across the dataset =
(per-batch RNG diversity) × (selector pick variance) × (per-episode RNG
re-seeding).

## 8. Configuration reference

### 8.1 Yaml flags (workspace `pipeline_config/recording_config_*.yaml`)

```yaml
perturbation:
  skill:
    enabled_forward: true          # master switch (false → cartesian-line only)
    enabled_reset: false           # reset-direction perturbation off by default
    n_candidates: 4                # batch size; plan_batch returns ≤ this many cands

    # curobo options:
    curobo_robot_cfg_path: robot_configs/curobo/so101_robot0.yml  # absolute or
                                                                  # project-relative
    curobo_num_trajopt_seeds: 4    # parallel trajopt starts per plan
    curobo_num_ik_seeds: 16        # parallel IK starts per query
    curobo_use_cuda_graph: true    # 1st call ~6s capture, then <0.2s per call
    curobo_via_offset_mag: 0.10    # m — via envelope radius around the line
    curobo_junction_smooth_k: 5    # cubic spline blend window at via junctions
    curobo_max_vias_per_candidate: 2   # K_max — see §2

    fixed_joint_indices: [4]       # wrist_roll lock (parity w/ cartesian IK)
    arm_joint_count: 5             # SO-101 = 5 DoF arm
```

`recording_config_*.yaml` files are **workspace-local** (gitignored —
see `.gitignore:52`). Each WS station maintains its own.

### 8.2 `CuroboBackendConfig` dataclass

The wire format from yaml to CuroboBackendConfig is in
`execution_forward_and_reset.py:471~`. Direct python users can
instantiate the config without yaml (see `tools/test_curobo_backend.py`).

```python
CuroboBackendConfig(
    enabled=True,                      # gates plan_batch entirely
    robot_cfg_path=str(...),            # absolute path required (defensive resolve)
    num_trajopt_seeds=4,
    num_ik_seeds=16,
    use_cuda_graph=True,
    max_batch_size=4,                   # = n_candidates
    interpolation_dt=None,              # None → trajopt default
    via_offset_mag=0.10,
    junction_smooth_k=5,
    fixed_joint_indices=(4,),
    arm_joint_count=5,
    max_vias_per_candidate=2,
)
```

### 8.3 Runtime conditions (gates inside `skills_lerobot.py`)

The flags above only enable the *option*. Perturbation actually *fires*
only when `move_to_position` is called with `is_transit=True` AND a
per-episode RNG is seeded AND IK converged. See §3.2.

## 9. File map

```
perturbation/skill_level/
├── CUROBO_METHODOLOGY.md     ← (this file)
├── __init__.py               ← exports get_curobo_backend(), TrajectoryCandidate
├── curobo_backend.py         ← CuroboBackend + CuroboBackendConfig
└── planner.py                ← TrajectoryCandidate dataclass (shared wire shape)
```

External entry points:

```
execution_forward_and_reset.py:381~      ← yaml → backend dispatch
skills/skills_lerobot.py:1787~           ← per-move plan_batch invocation
tools/test_curobo_backend.py             ← in-process correctness + perf test
tools/test_real_robot_smoke_curobo.py    ← real-robot 4-transit smoke
tools/visualize_curobo_via.py            ← EE-arc PNG generator (offline)
robot_configs/curobo/so101_robot0.yml    ← curobo robot/collision spec
                                          (generated once via curobo build tool)
pipeline_config/recording_config_ws*.yaml ← yaml runtime config (gitignored)
```

## 10. Verification methodology

Tests are layered from cheapest to most-real:

| Test | Scope | Hardware | Cost |
|---|---|---|---|
| `test_curobo_backend.py` | in-process plan_batch, IK timing, diversity | GPU only | seconds |
| `test_real_robot_smoke_curobo.py` | full production call path | GPU + robot | ~30 s |
| `visualize_curobo_via.py` | EE arc PNG for visual inspection | GPU | seconds |
| `run_forward_and_reset_ws*.sh` | full pipeline, dataset recorded | GPU + robot + cameras | minutes/episode |

Each commit pushed to master after `test_curobo_backend.py` passes.
Smoke test before yaml toggle that enables a new mode (e.g. K_max=2).
Pipeline rollout for actual data after smoke passes.

### 10.1 What `test_curobo_backend.py` checks

- `plan_batch(n=4)` returns 4/4 candidates
- `unique_costs == 4` (no collapse onto direct)
- `max wrist_roll deviation < 1e-6` rad (fixed joint actually locked)
- 2nd call ≥10× faster than 1st (CUDA graph cached)
- IK batched vs sequential timing breakdown

### 10.2 What `test_real_robot_smoke_curobo.py` checks

- 4 transits to safe forward targets (z ≥ 18 cm)
- Each transit logs `[Skill Perturbation] curobo:* ...` to stdout
- Mix of `curobo:direct`, `curobo:via1_s*`, `curobo:via2_s*` across the 4
- Motor positions reach within ~10 mm of target (gravity-sag dominated)
- Wall time: ~6 s first transit (graph compile), ~1–3 s after (cached
  + motion-time)

## 11. Known limitations & gotchas

### 11.1 SO-101 reachability

5 DoF is under-actuated for 6-DoF (xyz + quat) poses. The backend
locks wrist_roll via `fixed_joint_indices=[4]` to match the cartesian
IK convention. With wrist_roll locked, only **(x, y, z, pitch)** is
effectively controllable — yaw is geometrically determined by
shoulder_pan reaching the via xyz, and roll is fixed.

K=2 (two-via) paths fail IK ~10-15 % of the time on this robot when the
random vias land outside the reachable annulus. The backend falls back
to the linear-midpoint qpos for that via (so the candidate still
produces *something* — just a less curved path).

### 11.2 GPU memory ceiling

On RTX 2080 Ti (11 GB), the empirical ceiling is **N=16** with
K_max=1 (~9 GB used). With K_max=2 the cspace batch grows to
N · (K_max+1) = 3N, so effective ceiling drops proportionally
(N≤10 or so for K_max=2 on this card).

### 11.3 Per-trajectory candidate ≠ per-trajectory via count

The naming `curobo:via2_s1` describes *that slot's K count*, not
multiple vias per candidate slot. Each candidate has exactly K_slot
vias and K_slot+1 segments. There is no mechanism for K > K_max in a
single candidate.

### 11.4 Diversity collapse from RNG variance

With continuous sampling, some random (t, lat, vert) draws produce
vias close to the line → trajectory collapses to near-direct (cost ≈
direct cost). This is fine in the dataset average (per-episode RNG
diversifies across hundreds of episodes) but can show up within one
plan_batch call. Quick check: `visualize_curobo_via.py --seed K` for
several seeds — visual overlap of a via path with the gray reference
line indicates collapse.

### 11.5 cuda_graph_reset module-level flag

`curobo._src.runtime.cuda_graph_reset = True` is a module-global flag.
Setting it affects every curobo instance in the same process. The
backend sets it once on init; do not toggle it elsewhere.

### 11.6 OMPL backend removed

The earlier CPU OMPL ensemble + `mplib_env` subprocess daemon has been
removed. curobo is now the sole skill-level planner; the gRPC adapter
is the only alternative transport.

## 12. Commit trail (chronological)

```
780ed41  feat: curobo GPU backend (CuroboBackend) — 35× speedup
861f0af  perf(curobo): batched goalset IK + merged seg1+seg2 — 1.9× speedup
ce7cc6d  test(curobo): real-robot smoke test + production interface duck-type fix
c0353df  fix(curobo): resolve robot_cfg_path to absolute before handing to curobo
e50e167  feat(curobo): add CuroboBackend.close() to explicitly free CUDA graph
3309dac  feat(curobo): atexit-register close() + make it idempotent
6cd9b71  feat(curobo): continuous via-point sampling — replace 4-mode discrete table
9cc3e67  feat(curobo): random K_via per candidate (multi-via support, K_max≤2)
2bf895e  test(curobo): smoke test now exercises K_max=2 with 4 transits
68144d0  test(curobo): add EE-arc visualization tool for plan_batch candidates
ea8259e  feat(grpc_server): H100 gRPC server bootstrap script
```

## 13. Future work / open items

- **EntropySelector** (Phase 8): replace `RandomSelector` with policy-
  entropy-driven pick of 1 candidate from N. Requires plumbing the
  current policy's logits into the selector. N=8 then becomes more
  valuable than N=4 (richer candidate pool to pick from).
- **Visualization during pipeline rollout**: optional yaml flag
  (`debug_dump_viz: true`) to dump per-transit candidate PNG.
  Implementation hook would go in `skills_lerobot.py:1787~` after
  `plan_batch`. Cost: +500 ms FK + matplotlib per transit, ~200 KB PNG
  per transit. Recommended OFF in production, ON for first-N transit
  sanity checks.
- **K_max > 2**: theoretically possible but SO-101 reachability makes
  K=3 vias fail IK > 30 % of the time. Would need a more careful
  reachability filter or different robot.
