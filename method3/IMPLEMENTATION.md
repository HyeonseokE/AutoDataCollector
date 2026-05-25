# Method3 구현 가이드

> 이 문서는 method3 (online pre-selective real-world data acquisition) 의 **구현자 관점 가이드** 다.
> 모듈 인벤토리는 [`README.md`](./README.md), 명세는 [`../ours_method/final_method3_spec.md`](../ours_method/final_method3_spec.md) 를 참조.

## 0. 문서 가이드

| 문서 | 역할 | 분량 |
|------|------|------|
| [`README.md`](./README.md) | 모듈 인벤토리 — 각 파일의 한 줄 역할 + 폴더 구조 | 짧음 |
| **`IMPLEMENTATION.md`** (이 문서) | 알고리즘 흐름 + 핵심 수식 + paper 에 없는 구현 디테일 | 중간 |
| [`final_method3_spec.md`](../ours_method/final_method3_spec.md) | Paper-level 명세 (§0-16) | 길음 |
| [`paper_implementation_gaps.md`](../ours_method/paper_implementation_gaps.md) | Paper ↔ 구현 갭 audit | 참고 |
| [`paper_addition_proposals.md`](../ours_method/paper_addition_proposals.md) | Paper 추가 권고 (VLA fine-tune / DCT / MI) | 참고 |

각 절에는 두 종 reference 가 붙는다:

- `[code: path:line]` — 실제 코드 위치
- `[spec: §X]` — final_method3_spec 의 section

---

## 1. Overview

### Two-phase acquisition

```text
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: State Coverage Seeding                                │
│  ───────────────────────────────                                │
│  • 목적: state/subgoal support 형성 (H(A|S) 안정 추정 불가 → 우회)│
│  • 후보: K개 subgoal perturbation                                │
│  • Path: canonical (InterpPlan) — diversify subgoals, not paths │
│  • Output: TRUE 만 buffer 에 flush → P_phase1 seed              │
└───────────────────┬─────────────────────────────────────────────┘
                    ↓ phase1 종료 후
       ┌────────────────────────────────┐
       │  VLA Fine-tune (offline)        │
       │  Skill-wise traj → DCT_50 target│
       │  SmolVLA + single-step denoise  │
       │  → π_θ^(1) (Phase1-trained VLA) │
       └────────────────┬───────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Useful-OOD Selection                                  │
│  ──────────────────────────────                                 │
│  • 후보: Curobo + via-point + joint perturbation                │
│  • 평가 1 (buffer-side): M_MI = β·ΔH_A − λ·ΔH_A|S               │
│  • 평가 2 (model-side): U_VLA = single-step denoise (DCT space) │
│  • 선택: argmax U_VLA s.t. M̃_MI ≥ τ_MI (Useful-OOD)            │
│  • Output: accept → D_phase2_raw + vector DB ingest             │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 입출력

| Phase | 입력 | 출력 |
|-------|------|------|
| Phase1 | Initial scene + skill 정의 | `D_phase1_raw`, `SubgoalBuffer` (TRUE-only) |
| VLA fine-tune | `D_phase1_raw` (skill-segment 단위) | `π_θ^(1)` checkpoint (DCT_50 target) |
| Phase2 | `π_θ^(1)` + `P_phase1` (vector DB seed) | `D_phase2_raw` + 성장된 vector DB |

[spec: §0-2, §15]

---

## 2. Phase1: State Coverage Seeding

### 2.1 Skill-wise subgoal buffer

Skill 호출 단위로 buffer 를 분리한다. Skill key 는 **ordinal** (skill_0, skill_1, ...) — episode 내 호출 순서로 매김. Skill type (semantic label) 은 metadata 로만 보존.

```python
# method3/phase1_state_seeding/subgoal_buffer.py
class SubgoalBuffer:
    """skill_id (ordinal) → list[SubgoalEntry] mapping"""
```

`SubgoalEntry` 는 §5.2 필드 (subgoal xyz / canonical descriptor / episode_id) 를 영속화한다. Phase 종료 시 `SubgoalBuffer.save/load` 로 디스크 영속 — Phase1 ↔ Phase2 사이 보존.

[code: `phase1_state_seeding/subgoal_buffer.py`] [spec: §5]

### 2.2 Subgoal candidate 생성 (perturbation)

각 step 마다 현재 state $S_t$ 와 base subgoal $g_0$ 에 perturbation 을 가해 K개 후보 $\mathcal{G} = \{g'_1, ..., g'_K\}$ 를 생성. 두 모드:

| 모드 | 분포 | 용도 |
|------|------|------|
| `uniform_ball` | radius r 의 ball 내 uniform | default |
| `gaussian` | $\mathcal{N}(g_0, \sigma^2 I)$ | 작은 perturbation 한정 |

각 후보는 `subgoal_validity` 의 세 술어 (reachable / safe / inside AABB) 를 통과해야 valid set $\mathcal{G}_{valid}$ 에 진입.

[code: `phase1_state_seeding/subgoal_candidates.py`, `subgoal_validity.py`] [spec: §4.2]

### 2.3 Canonical preview + Terminal region descriptor T_end

각 valid 후보 $g'_j$ 마다 canonical trajectory 를 preview 한다:

$$\xi_j = \text{InterpPlan}(S_t, g'_j)$$

`interp_plan` 은 joint-space linear interpolation + path constraints (skill_id 기반) 으로 deterministic trajectory 를 만든다 — perturbation 은 subgoal 에만, path 는 canonical 유지 (§3.3 원칙: *Diversify subgoals, but keep paths canonical*).

Trajectory 의 **마지막 일정 비율** (`end_fraction`, default 0.2) 만 `T_end` 로 추출:

```python
# method3/phase1_state_seeding/canonical_preview.py:35
def last_segment(trajectory: np.ndarray, end_fraction: float) -> np.ndarray:
    """trajectory 의 마지막 end_fraction 부분 슬라이스"""
```

T_end 의 각 timestep 을 `terminal_descriptor.φ_goal` 로 geometric descriptor $\hat{h}_\tau$ 변환.

[code: `canonical_preview.py:12 interp_plan`, `:35 last_segment`, `terminal_descriptor.py`] [spec: §5.3-5.4]

### 2.4 Novelty scoring G_S^goal

T_end descriptor 들의 buffer 대비 novelty 평균을 후보 score 로 사용:

$$G_S^{goal}(g'_j) = \frac{1}{|T_{end}|} \sum_{\tau \in T_{end}} n_g(\hat{h}_\tau)$$

여기서 $n_g(\cdot)$ 는 neighbor-density 기반 novelty kernel — buffer 안 가까운 descriptor 수가 적을수록 ↑. 선택:

$$g^* = \arg\max_{g'_j \in \mathcal{G}_{valid}} G_S^{goal}(g'_j)$$

Cold-start 정책 (buffer 비었을 때): random 우선, scored 면 gain 내림차순.

[code: `subgoal_selector.py:109 Phase1SubgoalSelector`, `:322` 의 gain 계산] [spec: §5.4]

### 2.5 TRUE-only staging

선택된 subgoal 실행 후 episode judge 결과에 따라 buffer 반영을 분기:

```python
# method3/phase1_state_seeding/subgoal_selector.py:93
class _PendingMove:
    """staged 상태 — flush 또는 discard 대기"""
```

- `stage_executed(...)` — episode 끝까지 staged 보관
- `flush_episode(episode_id=...)` — judge=TRUE 면 buffer 에 commit
- `discard_episode()` — FALSE/UNCERTAIN 면 폐기

이로써 buffer 는 **검증된 성공 episode 의 subgoal 만** 으로 성장. Resume 시 episode_id 로 reconcile.

[code: `subgoal_selector.py:93 _PendingMove`, `:444 stage_executed`, `:488 flush_episode`] [spec: §5.5]

---

## 3. VLA Fine-tuning on Skill-wise DCT

Phase1 종료 후 Phase2 의 평가자가 될 `π_θ^(1)` 을 학습한다. 이 단계는 paper 에 거의 누락된 핵심 — [`paper_addition_proposals.md` §A](../ours_method/paper_addition_proposals.md) 참조.

### 3.1 Skill segmentation → Dataset

LeRobot v3 episode 들을 **skill 호출 단위** 로 분리한다. 한 sample = 한 skill segment.

| Field | 내용 |
|-------|------|
| 입력 | `(o_τ, I, m)` — segment 시작 frame observation + instruction + skill type |
| 정답 | segment 전체 action sequence (T, 6) — 가변 길이 |

```python
# method3/reembedding/lerobot_skill_segment_adapter.py:39
class LeRobotPhase1SkillSegmentAdapter:
    """LeRobot dataset → skill-segment 단위 RawTrajectoryDataset adapter"""
```

[code: `reembedding/lerobot_skill_segment_adapter.py`, `dct/skill_dataset.py`]

### 3.2 DCT_50 target

가변 길이 action sequence 를 고정 shape 으로 만들기 위해 truncated DCT 적용. 출력은 **(6, 50)** = top-50 low-frequency coefficient × 6-DoF action.

$$z^a_\xi = \text{DCT}_{L_0=50}(A_\xi) \in \mathbb{R}^{6 \times 50}$$

세 효과 동시 달성:

1. Skill-unit 단위 sample 화 (H-window sliding chunk 가 아닌)
2. 가변 길이 → fixed shape
3. Low-frequency 보존 → high-freq noise 제거

```python
# method3/dct/skill_dataset.py:218
dct_target = traj_to_dct(seg_actions, L0=50)
```

[code: `dct/skill_dataset.py:218`, `dct/build_skill_dct.py`]

### 3.3 SmolVLA fine-tune

Pretrained SmolVLA 의 head 를 **DCT_50 target** 으로 교체하여 single-step denoising loss 로 fine-tune.

$$\pi_\theta^{(1)}: (o_\tau, I, m) \;\longrightarrow\; \hat{z}^a \in \mathbb{R}^{6 \times 50}$$

학습 spec 예시 (최근 run):
- dataset: `CoRL2026-CSI/pnp_phase1_30_table2`
- step: ~1400
- final loss: ~0.016
- checkpoint: `smolvla_dct_20260524_113901/001400`

이 checkpoint 가 Phase2 의 `phase1_trained_vla_path` config 로 주입된다.

[code: `dct/validate_dct_inference.py` 의 validation 흐름] [spec: §6 Step 1-2, paper §3.4]

---

## 4. Phase2: Useful-OOD Selection

### 4.1 Phase1 vector DB 재구축 (re-embedding)

`D_phase1_raw` 의 모든 skill segment 를 **frozen** `π_θ^(1)` 로 re-embedding 하여 Phase2 의 retrieval index $P_{phase1}^{(m)}$ 를 만든다.

```python
# method3/reembedding/seed_builder.py
e_i = [φ_VLA^(1)(o_i, I_i); p_i]   # state key, dim 965
z_i^a = ψ_EE-delta-DCT(A_i)         # action descriptor, dim 300
```

- **state_key** (965 dim) = SmolVLA-DCT encoder hidden ⊕ proprio (절대좌표, image-conditioned)
- **action_descriptor** (300 dim) = EE delta DCT (translation-invariant motion shape)

두 representation 의 출처가 다른 이유는 §4.3 참조.

[code: `reembedding/seed_builder.py`, `reembedding/vla_encoder.py`] [spec: §6]

### 4.2 Curobo candidate generation

각 subgoal seed 별로 K-batch trajectory 를 Curobo motion planner 로 생성. 다양화는 두 layer:

| Layer | Mechanism | Default |
|-------|-----------|---------|
| Via-point insertion | skill 중간에 0~`max_vias` 개 via-point 추가 | `max_vias=2` |
| Joint perturbation | via-point joint config 에 noise 추가 | config |

```python
# method3/phase2_mi_selection/curobo_candidate_gen.py:62
class CurobogenConfig:
    ...
```

Arm-only 5-DoF (gripper 제외) 로 통일된 후 후속 처리에서 6-DoF 로 패딩. Joint→servo 변환은 `JointServoConverter` (paradigm A.3).

EE FK 로 EE pose chunk 를 얻고 `ee_delta_dct_from_poses` 로 (50, 6) EE delta DCT 생성 → action_descriptor.

[code: `phase2_mi_selection/curobo_candidate_gen.py:62 CurobogenConfig`, `:160 candidates_from_trajectory_list`, `dct/ee_features.py`]

### 4.3 두 descriptor 분리 (핵심 변경점)

Paper 의 단일 descriptor 가정과 달리, 실제 구현은 **두 descriptor 를 명시적으로 분리** 한다.

| 용도 | 표현 | Shape | Frame |
|------|------|-------|-------|
| state retrieval (neighbor) | SmolVLA-DCT hidden ⊕ proprio | 965 | 절대 (image-conditioned) |
| M_MI 계산 (ΔH_A, ΔH_A\|S) | **EE delta DCT** | 300 = 50×6 | **translation-invariant** |
| U_VLA denoise space | **joint DCT** (VLA target 과 동일) | 300 = 50×6 | joint space |

EE delta DCT 는 `np.diff(poses, axis=0)` 후 DCT — position 의 절대좌표를 제거하고 **motion shape 만** 보존. 같은 motion 을 다른 위치에서 수행한 trajectory 가 같은 descriptor 값을 가지므로, state coverage 와 action diversity 가 분리 측정 가능.

[code: `phase2_mi_selection/mi_selector.py:53 Phase2Candidate`, `dct/ee_features.py`] [spec: §4.2 보강 필요 — paper_addition_proposals §B]

### 4.4 M_MI = β·ΔH_A − λ·ΔH_A|S

#### 4.4.1 ΔH_A (action coverage gain)

후보의 action descriptor 가 buffer 의 NN 과 얼마나 먼지 — top-q novelty.

[code: `phase2_mi_selection/action_coverage.py`] [spec: §8]

#### 4.4.2 ΔH_A|S (conditional ambiguity)

Covered window — state neighbor 가 `k_min` 이상인 window — 안에서만 평가:

```python
# method3/phase2_mi_selection/conditional_ambiguity.py:155
for tau in range(n_windows):
    if int(covered_count[tau]) < k_min:
        continue                             # under-covered → skip
    neighbors = np.flatnonzero(covered_mask[tau])
    d_min = float(z_dist[cz_idx, neighbors].min())     # nearest action distance
    s_a   = max(mean_nn_distance(db_z[neighbors]), s_min)  # support spread
    deltas.append(max(math.log(d_min / (s_a + eps)), 0.0))  # log-ratio
```

식:

$$\widehat{\Delta H}_{A|S} = \text{agg}_{\tau \in \text{covered}}\, \max\!\left(\log\frac{d_{\min}^a(\tau)}{s_a(\tau) + \epsilon},\; 0\right)$$

- $d_{\min}^a(\tau)$ — covered neighbor 중 가장 가까운 action descriptor 거리
- $s_a(\tau)$ — neighbor 부분집합의 mean-NN distance (support 내부 spread), `s_min=1e-3` floor
- agg ∈ {mean, max} (default mean)
- Default `k_min=3`

Covered window 0개면 `ΔH_A|S = 0` 반환 — under-covered branch 처리는 호출부 책임.

[code: `conditional_ambiguity.py:86 conditional_ambiguity`, `:54 covered_windows`] [spec: §9]

#### 4.4.3 M̃_MI batch z-score

Candidate batch 안에서 M_MI 를 z-score 정규화:

$$\widetilde{M}_{MI} = \frac{M_{MI} - \mu_{batch}}{\sigma_{batch} + \epsilon}$$

τ_MI 는 절대값이 아닌 **z-score threshold** (default `τ_MI=0.0` = "batch mean 이상").

```python
# method3/phase2_mi_selection/mi_selector.py:378-379
mu, sigma = float(m_mi.mean()), float(m_mi.std())
m_mi_norm = (m_mi - mu) / (sigma + cfg.eps)
```

[code: `mi_selector.py:378`] [spec: §13.2]

### 4.5 U_VLA single-step denoise loss (DCT space)

후보의 joint DCT representation $z^a_{\xi}$ 에 대해 `π_θ^(1)` 의 single-step denoising loss 를 평가.

$$U_{VLA}(\xi) = L_{\text{denoise}}(o_\tau, I, m, z^a_{\xi}; \pi_\theta^{(1)})$$

```python
# method3/phase2_mi_selection/vla_informativeness.py:96
class LeRobotVLAInformativenessScorer:
    """policy.forward(batch, reduction='mean') 으로 단일 candidate U_VLA 계산"""
```

- `mode="dct"` — candidate.dct_target 을 batch["action"] 자리에 inject, **R=1 강제** + optional `sigma` (deterministic time)
- `mode="default"` — raw action chunk (paper 식 (188) 의 R회 평균)

DCT mode 의 의의: VLA 가 학습한 target space (DCT) 와 candidate representation space 가 **shape · semantic 모두 일치** 하여 denoise loss 가 well-defined.

[code: `vla_informativeness.py:96`, `:132 score`, `:180 score_batch`] [spec: §12, paper_addition_proposals §B]

### 4.6 Selection policy

기본 (Q1 — Useful-OOD):

$$\xi^* = \arg\max_{\xi}\, U_{VLA}(\xi) \quad \text{s.t.}\quad \widetilde{M}_{MI}(\xi) \ge \tau_{MI},\; \text{not under-covered}$$

`selection_mode` config 로 4개 모드 ablation 가능:

| Mode | argmax/min | M̃_MI 조건 | 의미 |
|------|-----------|-----------|------|
| **Q1** (default) | argmax U_VLA | ≥ +τ_MI | Useful OOD |
| Q2 | argmax U_VLA | ≤ −τ_MI | Harmful OOD |
| Q3 | argmin U_VLA | ≥ +τ_MI | Useful ID |
| Q4 | argmin U_VLA | ≤ −τ_MI | Redundant ID |

**Fallback** (eligible 후보 없을 때): Q1/Q3 → argmax M_MI, Q2/Q4 → argmin M_MI. 이 경우 `accepted=False` 마크 — episode 는 dataset 에 들어가지만 selection 통계상 별도 카테고리.

[code: `mi_selector.py:342 select`, `:91 Phase2MIConfig`] [spec: §11-13]

---

## 5. System Architecture

### 5.1 gRPC server-client (H100 ↔ robot)

VLA inference 와 candidate scoring 은 H100 GPU 서버에서 수행. Robot client 는 candidate trajectory 와 observation 을 보내고 selection 결과만 받는다.

```text
[robot client] ──candidate trajs──→ [H100 server]
                                        │
                                        ├─ VLA forward (π_θ^(1))
                                        ├─ U_VLA / M_MI 계산
                                        └─ argmax + accept_to_buffer
   ◀──── chosen index + telemetry ──────┘
```

| 파일 | 역할 |
|------|------|
| `grpc_server/preselective.proto` | RPC 스키마 |
| `grpc_server/server.py` | server-side scoring + buffer ingest |
| `grpc_server/client.py` | client-side stub |
| `grpc_server/_codec.py` | encode/decode utilities |
| `grpc_server/launch_remote_server.sh` | 원격 server bring-up |

**Server-side accept_to_buffer** — useful-OOD accept 된 demo frame 을 server-side vector DB 에 push. Client 는 IngestEpisode 를 호출하지 않음 (z-space dimensionality 정합 보존).

[code: `grpc_server/`] [spec: 누락 — paper_implementation_gaps §A1-A2]

### 5.2 Episode lifecycle 관리

Episode 단위로 buffer / vector DB / raw dataset 을 reconcile.

```python
# method3/episode_lifecycle/
episode_id      # episode 단위 식별자 (resume 가능하도록 stamp)
reconciler      # buffer ↔ dataset 정합성 회복
migrate_legacy_buffer  # legacy 구조 → 신규 episode-aware 구조
```

- Resume 시 batch_info.json 의 `judge` 값으로 episode 생존 여부 판정 — TRUE 만 keep, FALSE/UNCERTAIN/PENDING 은 cleanup
- buffer 의 staged `_PendingMove` 도 episode_id 로 reconcile

[code: `method3/episode_lifecycle/`] [spec: 누락]

### 5.3 Vector DB + raw dataset 분리

두 저장 layer 가 명확히 분리되어 있다.

| Layer | 역할 | 파일 |
|-------|------|------|
| `D_phase{1,2}_raw` | source-of-truth — 모든 raw trajectory + metadata | `method3/storage/raw_dataset.py` |
| `B_t^{(m)}` (vector DB) | searchable index — re-embedding 가능한 lightweight key/descriptor | `method3/phase2_mi_selection/vector_db.py` |

`RawDatasetEntry` 는 image 를 직접 저장하지 않고 `observation_ref` (pointer) 로만 가리킴 — 이미지가 무거우므로 LeRobot dataset 안에 있고 pointer 로 복원. 이 덕분에 frozen VLA 변경 시 vector DB 만 재계산하면 됨.

[code: `storage/raw_dataset.py:33 RawDatasetEntry`, `:77 RawTrajectoryDataset`, `phase2_mi_selection/vector_db.py`] [spec: §3, §3.1]

---

## 6. Operational Policies

### 6.1 Stack task override

Top-down 단일 카메라에서 vertical stacking 판정은 occlusion 만으로 추론해야 하므로 VLM judge 가 자주 false-negative. Instruction 에 stacking 키워드 있으면 judge=FALSE 라도 TRUE 로 강제하여 데이터 레코딩 진행.

```python
# execution_forward_and_reset.py:3548
_is_stacking = any(
    k in _instr_lower
    for k in ("stack", "on top of", "from bottom to top", "from top to bottom")
)
if _is_stacking and judge_prediction != "TRUE":
    result['judge']['_stack_override'] = True
    result['judge']['_original_prediction'] = result['judge'].get('prediction')
    result['judge']['prediction'] = "TRUE"
    judge_prediction = "TRUE"
```

원본 판정은 `_original_prediction` + `_stack_override` 로 보존 — 후속 분석/ablation 에서 차감 가능.

Judge prompt 자체에도 stacking heuristic 섹션 추가 — top-down 한계 명시 + occlusion 기준 가이드.

[code: `execution_forward_and_reset.py:3548`, `judge/forward_execution/prompt.py` Vertical Stacking Heuristic 섹션]

### 6.2 Reset edge-margin 검증

Reset seed candidate 의 bbox 전체가 green margin (30px inset) 안에 있어야 valid — 부피가 이미지 가장자리를 침범 못함.

```python
# code_gen_lerobot/reset_execution/workspace.py:239-242
# [ACTIVE] Strict policy: bbox 전체가 edge_margin 안쪽 강제 → 부피 침범 금지
if (cu - hw < edge_margin or cu + hw >= img_w - edge_margin or
    cv - hh < edge_margin or cv + hh >= img_h - edge_margin):
    continue
```

`edge_margin = 30`, image (img_w, img_h) = (640, 480). 4-corner 모두 검사.

[code: `code_gen_lerobot/reset_execution/workspace.py:239-242`]

### 6.3 Pick/place compensation

so101 robot 의 URDF 캘리브레이션 잔존 offset (real fingertip ≈ commanded + 5~10mm) 을 두 layer 로 보상:

| Parameter | 의미 | 기본값 |
|-----------|------|--------|
| `pick_offset` | object top 으로부터 descent 깊이 | 0.025 (2.5 cm) |
| `pick_z_offset` | URDF offset 보상 (signed) | 0.0 (필요 시 −0.005~−0.020) |
| `pick_xy_offset` | radial undershoot 보상 | 0.010 (1 cm outward) |
| `gravity_sag.gain` | high-z droop LUT | 0.1711 |

Pick/place formula: `fingertip_z = object_top − pick_offset + pick_z_offset`. Stacking 같은 정밀 task 에서는 `pick_z_offset` 의 음수 조정이 필요.

[code: `robot_configs/motor_calibration/so101/robot4_compensation.json`]

---

## 7. Configuration & Entry Points

### 7.1 Pipeline config (YAML)

| 파일 | 대상 | 로드 |
|------|------|------|
| `pipeline_config/phase1_config.yaml` | Phase1 subgoal seeding | recording 파이프라인 자동 로드 |
| `pipeline_config/phase2_config.yaml` | Phase2 MI + 전환 + re-embed | `method3.config.load_phase2_config()` |

Phase2 config 의 핵심 키:
- `phase1_trained_vla_path` — fine-tuned VLA checkpoint 경로
- `phase1_dataset_path` — re-embedding source
- `selection_mode` — Q1/Q2/Q3/Q4
- `tau_MI` — z-score threshold (default 0.0)
- `skill_planner_transport` — `grpc` (원격) | `local`

### 7.2 execution_forward_and_reset.py 분기

```python
# execution_forward_and_reset.py:186
def __init__(self, method3_phase: str = "phase1", ...):
    # method3_phase: "phase1" | "phase2"
```

`method3_phase` 에 따라:
- Phase1 → `Phase1SubgoalSelector` active
- Phase2 → `Phase2MISelector` + `SkillVectorDB` load + (gRPC mode 면) remote server 통신

Episode 저장도 자동 분기 — `phase2/episode_NN/` subdir.

[code: `execution_forward_and_reset.py:159 ForwardAndResetPipeline`, `:186 method3_phase`]

### 7.3 Launcher scripts

```bash
./run_forward_and_reset_ws3.sh    # WS3 (현재 active 환경)
./run_forward_and_reset_ws{1,2,4,8}.sh   # 다른 worker station
./scripts/phase2_prep_chain.sh    # Phase1 → fine-tune → Phase2 preparation chain
```

### 7.4 Offline dry-run

```bash
conda activate lerobot_cap
python -m method3.acquisition.demo_offline
```

하드웨어 없이 전체 acquisition 흐름 (Phase1 → readiness → re-embed → Phase2) 을 in-memory env 로 시뮬레이션.

---

## 8. Testing & Verification

### 8.1 Unit test

```bash
conda activate lerobot_cap
python -m pytest method3/ -q                          # 전체
python -m pytest method3/phase2_mi_selection/tests/   # Phase2 만
python -m pytest method3/dct/tests/                   # DCT 만
```

각 module 의 `tests/` 폴더에 spec § 별로 unit test — `test_phase1_subgoal_scoring.py` (§5), `test_phase2_mi_selector.py` (§11-12) 등.

### 8.2 End-to-end 검증 절차

1. **Phase1 run**: `phase1_config.yaml` 편집 → `./run_forward_and_reset_ws3.sh` (방법론_phase=phase1)
2. **VLA fine-tune**: `python -m method3.dct.build_skill_dct ... && python -m method3.dct.train ...`
3. **Re-embedding 검증**: `python -m method3.dct.validate_dct_inference` (predicted DCT ↔ GT DCT space MSE)
4. **Phase2 run**: `phase2_config.yaml` 편집 (`phase1_trained_vla_path` 지정) → `./run_forward_and_reset_ws3.sh` (`method3_phase=phase2`)
5. **결과 확인**: `results/session_*/episode_*/forward/judge_result.json`, vector DB 성장 (`grpc_server/buffer/`)

---

## Appendix A: 모듈별 file:line reference table

| 모듈 | 핵심 entry | spec § |
|------|-----------|--------|
| Phase1 selector | `phase1_state_seeding/subgoal_selector.py:109 Phase1SubgoalSelector` | §5.4 |
| Phase1 staging | `subgoal_selector.py:93 _PendingMove`, `:488 flush_episode` | §5.5 |
| Canonical preview | `phase1_state_seeding/canonical_preview.py:12 interp_plan` | §5.4 |
| Subgoal buffer | `phase1_state_seeding/subgoal_buffer.py SubgoalBuffer` | §5 |
| Skill segment adapter | `reembedding/lerobot_skill_segment_adapter.py:39` | §6 |
| DCT_50 target | `dct/skill_dataset.py:218 traj_to_dct(L0=50)` | (paper §3.4 보강) |
| EE delta DCT | `dct/ee_features.py:44 ee_delta_dct_from_poses` | (paper §3.4 보강) |
| Seed builder | `reembedding/seed_builder.py build_phase1_vector_db` | §6 |
| Phase2 selector | `phase2_mi_selection/mi_selector.py:179 Phase2MISelector` | §11-13 |
| Phase2 candidate | `mi_selector.py:53 Phase2Candidate` | §7 |
| Conditional amb | `phase2_mi_selection/conditional_ambiguity.py:86 conditional_ambiguity` | §9 |
| Covered windows | `conditional_ambiguity.py:54 covered_windows` | §9.1 |
| U_VLA scorer | `phase2_mi_selection/vla_informativeness.py:96 LeRobotVLAInformativenessScorer` | §12 |
| Curobo gen | `phase2_mi_selection/curobo_candidate_gen.py:62 CurobogenConfig`, `:160` | (paper §3.4 보강) |
| Raw dataset | `storage/raw_dataset.py:33 RawDatasetEntry`, `:77 RawTrajectoryDataset` | §3.1 |
| Episode lifecycle | `episode_lifecycle/episode_ref.py`, `reconciler.py` | (누락) |
| gRPC server | `grpc_server/server.py`, `preselective.proto` | (누락) |
| Pipeline 분기 | `execution_forward_and_reset.py:186 method3_phase param` | — |
| Stack override | `execution_forward_and_reset.py:3548` | (누락) |
| Reset edge margin | `code_gen_lerobot/reset_execution/workspace.py:239-242` | (누락) |

## Appendix B: Hyperparameter table

| Symbol | Meaning | Default | 위치 |
|--------|---------|---------|------|
| `L0` | DCT coefficient 수 | 50 | `dct/skill_dataset.py:109` |
| `D_e` | state key dimension | 965 | SmolVLA hidden + proprio |
| `D_z` | action descriptor dimension | 300 | EE delta DCT 50×6 |
| `k_min` | covered window 판정 최소 neighbor | 3 | `conditional_ambiguity.py:92` |
| `s_min` | s_a floor (penalty 폭주 방지) | 1e-3 | `conditional_ambiguity.py:93` |
| `ρ_m` | state-neighborhood radius | yaml | config |
| `τ_MI` | M̃_MI threshold (z-score) | 0.0 | `mi_selector.py:104` |
| `β, λ` | M_MI weights (ΔH_A, ΔH_A\|S) | yaml | config |
| `max_vias` | Curobo via-point 최대 수 | 2 | recent commit 82393ec |
| `R` | stochastic denoise 평가 횟수 | 1 (dct mode 강제) | `vla_informativeness.py:118` |
| `end_fraction` | T_end 슬라이스 비율 | 0.2 | spec §5.3 |
| `edge_margin` | reset bbox green margin | 30 px | `workspace.py:236` |
| `pick_offset` | pick descent 깊이 | 0.025 m | robot4_compensation.json |
| ARM_DOF | 통일 차원 (gripper 제외) | 5 | `lerobot_skill_segment_adapter.py:126` |

---

## 변경 로그

| 날짜 | 변경 |
|------|------|
| 2026-05-24 | 최초 작성 — Phase1/Phase2/VLA/system/operational 6 영역 가이드 |
