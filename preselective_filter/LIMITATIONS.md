# preselective_filter — Known Limitations

## L1. AC_model cannot distinguish multi-modal π₀ from OOD π₀

### Issue

`AC_model` uses
$$
D_{\text{model-AC}}^j = \min_{p_r \in \mathcal{M}_{\pi_0}(x_m)} \lVert \mathrm{vec}(A_{\xi_m^j}) - p_r \rVert_2^2
$$
where $\mathcal{M}_{\pi_0}(x_m) = \{\hat A_1, \dots, \hat A_M\}$ are M
samples from $\pi_0(\cdot \mid O_m, S_m, I)$.

The `min` operator does not see whether those M samples are:

- **응집된 modes** — π₀ is confident, recognizes K valid action modes
  for this context (e.g., 8 samples → LEFT×4 ∪ RIGHT×4)
- **흩어진 noise** — π₀ is in OOD and outputs unstructured samples

In both cases, almost any candidate has a small `min` distance to *some*
sample → AC_model is high. Combined with high IG (novelty), the final
score `IG · AC` can label genuinely OOD regions as "Useful Diversity".

### Why we accept this (decision (i))

1. **Scope** — Method 3 operates on the "novel but in-distribution"
   region. Fully-OOD states are out of scope.
2. **Upstream guard** — Method 2 perturbation layers (skill-level
   OMPL/Curobo + subgoal Gaussian) are tuned to keep candidates
   task-relevant. Candidates drifting into hard OOD are not Method 3's
   responsibility to detect.
3. **No-threshold selection** — Per spec, every skill step must produce
   one choice. Even an OOD-tainted choice degrades to "least bad", not
   to a corrupted dataset gate.

### Possible future mitigations (not in v1)

If this assumption breaks in practice, the following gates can be added
without changing the public Protocols:

| Idea | What it measures | Config addition |
|---|---|---|
| Sample variance gate | `std(samples) > τ` → drop AC_model | `model_ood_threshold` |
| Cluster structure penalty | intra/inter cluster ratio | `cluster_quality_weight` |
| Context-level U_π₀ check | mean `L_FM` across candidates | `context_ood_threshold` |
| Buffer cross-check | distribution similarity between π₀ samples and `M_B(x_m)` | `buffer_crosscheck_weight` |

Clustering-based gates require an external dependency (scipy/sklearn)
and are tracked separately from the core module.

### Operational monitoring suggestions

- Log `std(D_model-AC)` per skill step; sudden spikes hint at OOD.
- Inspect rollouts flagged as "Useful Diversity" in low-coverage skill
  stages — if many fail or look chaotic, the AC_model signal is being
  poisoned by OOD samples.
