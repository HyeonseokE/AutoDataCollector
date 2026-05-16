# vla_adaptor — Integration Guide

How to wire `preselective_filter` (via `vla_adaptor`) into the existing
data-collection pipeline. This guide is intentionally **separate from the
code** so the integration changes can be reviewed/approved before
production files are touched.

## Required changes (3 files + 1 yaml schema)

### 1. `execution_forward_and_reset.py`

Add a setup hook mirroring `_setup_skill_perturbation_on_skills`:

```python
def _setup_preselective_filter_on_skills(self) -> None:
    """Build Selector from recording_config and inject into skills."""
    if not self.recording_config:
        return
    try:
        import yaml as _yaml
        with open(self.recording_config, "r") as f:
            full_cfg = _yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[preselective_filter] failed to read recording_config: {e}")
        return

    try:
        from vla_adaptor import setup_preselective_filter
        selector = setup_preselective_filter(full_cfg)
    except Exception as e:
        print(f"[preselective_filter] setup failed: {e}")
        return

    if selector is None:
        return  # disabled
    self._preselective_selector = selector
    self._skills.set_preselective_selector(selector)
    print("[preselective_filter] selector attached to skills")


def _teardown_preselective_filter(self) -> None:
    sel = getattr(self, "_preselective_selector", None)
    if sel is None:
        return
    from vla_adaptor import teardown_preselective_filter
    teardown_preselective_filter(sel)
    self._preselective_selector = None
```

Call `self._setup_preselective_filter_on_skills()` in the same place as
`self._setup_skill_perturbation_on_skills()` (around line 313).
Call `self._teardown_preselective_filter()` near
`self._teardown_skill_perturbation()` (around line 786).

### 2. `skills/skills_lerobot.py`

Add to `LeRobotSkills.__init__`:

```python
self._preselective_selector = None
self._last_preselection: dict[int, "Selection"] = {}  # keyed by skill step index
```

Add setter:

```python
def set_preselective_selector(self, selector) -> None:
    self._preselective_selector = selector
```

Modify the call site at `skills_lerobot.py:1787` (currently
`rng.choice(cands)`):

```python
if (is_transit
    and self._skill_planner_client is not None
    and self._perturbation_rng is not None
    and trajectory.ik_converged):
    cands = client.plan_batch(current_joints, goal_joint_rad, n=4, seed=...)
    if cands:
        if self._preselective_selector is not None:
            # ---- Method 3 path ----
            from vla_adaptor import trajectory_to_action_chunk
            from preselective_filter import Candidate, Context

            chunk_size = self._preselective_selector.policy.policy.config.chunk_size
            action_dim = self._preselective_selector.policy.policy.config.max_action_dim
            fps = self._recording_fps  # piped from execution_forward_and_reset
            current_gripper = self._current_gripper_value()  # implement helper
            obs_now = self._current_observation()             # implement helper
            state_now = self._current_state()                 # implement helper

            wrapped = [
                Candidate(
                    skill_id=self._current_skill_id(),
                    action_chunk=trajectory_to_action_chunk(
                        waypoints=c.waypoints,
                        times=c.times,
                        chunk_size=chunk_size,
                        action_dim=action_dim,
                        fps=fps,
                        current_gripper=current_gripper,
                        arm_dof=self._arm_dof,
                    ),
                    payload=c,
                )
                for c in cands
            ]
            ctx = Context(
                observation=obs_now,
                state=state_now,
                instruction=self._current_instruction(),
                skill_id=self._current_skill_id(),
            )
            selection = self._preselective_selector.select(ctx, wrapped)
            chosen = selection.chosen_candidate.payload  # original TrajectoryCandidate
            # Store for add_to_buffer at judge-True checkpoint
            self._last_preselection[self._skill_step_idx] = (ctx, selection)
        else:
            # ---- Original random path (unchanged) ----
            chosen = cands[rng.integers(0, len(cands))]

        trajectory.joint_positions = chosen.waypoints
        # (rest of time-parameterization unchanged)
```

### 3. Recorder hook (judge True → add_to_buffer)

In whatever module handles the post-judge episode finalization (likely
`record_dataset/recorder.py` or wherever `save_episode` is called):

```python
def finalize_episode(judge_result, ...):
    if not judge_result:
        return  # drop episode, no buffer add either

    selector = getattr(skills, "_preselective_selector", None)
    if selector is not None:
        for skill_step_idx, (ctx, selection) in skills._last_preselection.items():
            selector.add_to_buffer(ctx, selection)
    skills._last_preselection.clear()

    recorder.save_episode(...)  # existing LeRobot dataset save
```

The order — `add_to_buffer` then `save_episode` — matches decision #6
(judge True → buffer → dataset).

### 4. `pipeline_config/recording_config_ws*.yaml`

Add the section (copy from the schema docstring in `pipeline_setup.py`):

```yaml
preselective_filter:
  enabled: false                                       # opt-in
  policy:
    checkpoint: "CoRL2026-CSI/smol_CaP_pnp_10fps"
    device: "cuda"
  buffer:
    root: "./results/<session>/preselective_buffer"
  selector:
    alpha: 0.5
    lam: 0.5
    n_vla_samples: 8
    max_modes: null
    context_k: 8
  adapter:
    n_fm_mc_samples: 8
    z_pool: "mean"
```

Keep `enabled: false` by default. Flip to `true` per workspace yaml after
the SmolVLA checkpoint is in place locally.

---

## Verification checklist after wiring

1. `python -m preselective_filter.tests.test_selector_smoke` still passes (no regression in pure module)
2. `python -c "from vla_adaptor import setup_preselective_filter"` imports without errors
3. With `enabled: false`, the pipeline behaves identically to today
4. With `enabled: true` and a tiny test episode:
   - First skill step takes ~700 ms extra (π₀ forward × 4 candidates + sample_actions)
   - Cold-start: AC_buffer = 1.0 for all candidates (logged in ScoreReport)
   - After 1 successful episode, `<results>/<session>/preselective_buffer/<skill>.jsonl` exists with one line per skill step
   - Second episode: AC_buffer values vary across candidates

## Things to confirm during real-world bring-up

- `chunk_size`, `max_action_dim`, `fps`, `arm_dof` reads match the policy actually being loaded
- `_current_observation/_current_state/_current_gripper/_current_instruction/_current_skill_id` helpers exist or need adding to LeRobotSkills
- Per-skill-step latency budget (~700 ms) acceptable for the recording_fps cadence
- SmolVLA GPU memory + ongoing perturbation backend (curobo) coexist

## Out-of-scope for this guide

- Auto-promoting `enabled: true` per workspace (manual toggle for now)
- Buffer compaction / retention policy
- Clustering-based `max_modes > None` (decision (a) — explicitly deferred)
- Anything covered in `preselective_filter/LIMITATIONS.md`
