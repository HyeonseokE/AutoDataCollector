# Paper ↔ Implementation Gap Audit

> Reference: [`final_method3_paper.md`](./final_method3_paper.md)
> Audit date: 2026-05-24
> Scope: §3.3 Phase1 / §3.4 Phase2 / §3.5 Acquisition Policy + system layers paper 에 누락된 부분

---

## 📑 Table of Contents

1. [Phase1 (§3.3) 부족 내용](#1-phase1-33-부족-내용)
2. [Covered window §9.1 + k_min default](#2-covered-window-91--k_min-default)
3. [ΔH_A|S §9.3 실제 식 + M̃_MI z-score](#3-δh_as-93-실제-식--m̃_mi-z-score)
4. [Curobo + via-point + joint perturbation mechanism](#4-curobo--via-point--joint-perturbation-mechanism)
5. [VLA 임베딩 디테일 (D 카테고리 통합)](#5-vla-임베딩-디테일-d-카테고리-통합)
6. [Paper 작성 우선순위 (★★★★★ 두 항목 핵심)](#6-paper-작성-우선순위-두-항목-핵심)
7. [부록: 카테고리별 전체 갭 매트릭스](#부록-카테고리별-전체-갭-매트릭스)

---

## 1. Phase1 (§3.3) 부족 내용

| # | 갭 | 실제 위치 | Paper 상태 |
|---|----|-----------|------------|
| 1.1 | **InterpPlan mechanism 부재** | `method3/phase1_state_seeding/canonical_preview.py: interp_plan` | §3.3 식 (122) 표기만 |
| 1.2 | **Terminal region T_end 정의 부재** | `subgoal_selector._terminal_region` (마지막 segment 추출) | T_end 자체 미등장 |
| 1.3 | **G_S^goal 실제 식 누락** | `gain = mean(n_g(ĥ_τ) for τ ∈ T_end)` | §3.3 식 (114-117) 추상 표기 |
| 1.4 | **n_g (novelty kernel) form 미정의** | neighbor-density 기반 novelty | functional form / radius / scaling 0 |
| 1.5 | **TRUE-only staging 정책 부재** | `_PendingMove` → `flush_episode` / `discard_episode` | buffer 가 항상 자라는 것처럼 묘사 |

### 핵심 영향

- Paper 만 보고 Phase1 재현 시 **subgoal scoring 의 수치적 정의 부재** 로 동일 결과 도출 불가.
- Cold-start 정책 (buffer 비었을 때 random 우선, scored 면 gain 내림차순) 도 paper 미반영.

---

## 2. Covered window §9.1 + k_min default

| # | 갭 | 디테일 |
|---|----|--------|
| 2.1 | **Covered window 정의 부재** | state-neighbor 가 `k_min` 이상인 window 만 ΔH_A|S 합산에 포함 |
| 2.2 | **k_min=3 hardcoded default** | paper §3.4 식 (178) 은 `k_min^(m)` 으로 skill 별처럼 표기, 실제는 global |
| 2.3 | **`covered_windows()` 헬퍼 미정의** | `[(window_idx, neighbor_indices), ...]` 자료구조 paper 에 0 |
| 2.4 | **n_covered=0 fallback** | ΔH_A|S=0.0 반환 → under-covered 분기는 호출부 책임 |
| 2.5 | **radius ρ_m sensitivity 미분석** | D_e=965 high-dim space 에서 default / scaling 분석 0 |

### 코드 근거

```python
# method3/phase2_mi_selection/conditional_ambiguity.py:155
for tau in range(n_windows):
    if int(covered_count[tau]) < k_min:
        continue            # ← covered window 만 ΔH_A|S 에 기여
```

---

## 3. ΔH_A|S §9.3 실제 식 + M̃_MI z-score

| # | 갭 | 실제 식 |
|---|----|---------|
| 3.1 | **ΔH_A|S 실제 식** | `max(log(d_min^a / (s_a + ε)), 0)` per covered window 의 mean/max |
| 3.2 | **d_min^a 정의** | neighbor 중 nearest action descriptor 거리 |
| 3.3 | **s_a 정의** | neighbor 부분집합의 mean-NN distance, `s_min=1e-3` floor |
| 3.4 | **M̃_MI batch z-score** | `(m_mi - μ_batch) / (σ_batch + ε)` |
| 3.5 | **agg = "mean"\|"max" 옵션** | paper 단일 aggregation 가정, 실제 config 가능 |

### 코드 근거

```python
# method3/phase2_mi_selection/conditional_ambiguity.py:159-173
d_min = float(z_dist[cz_idx, neighbors].min())            # §9.3 d_min^a
s_a   = max(mean_nn_distance(db_z[neighbors]), s_min)
deltas.append(max(math.log(d_min / (s_a + eps)), 0.0))    # §9.3
```

```python
# method3/phase2_mi_selection/mi_selector.py:378-379
mu, sigma  = float(m_mi.mean()), float(m_mi.std())
m_mi_norm  = (m_mi - mu) / (sigma + cfg.eps)              # M̃_MI = z-score
```

### Paper 와의 의미적 차이

> Paper §3.4 식 (164): `M_MI = β·ΔH_A − λ·ΔH_A|S` 의 τ_MI 가 **절대값** 처럼 표기됨.
> 실제로는 `m_mi_norm ≥ τ_MI` (z-score threshold) 으로 작동 → `τ_MI=0` = "batch mean 이상".

---

## 4. Curobo + via-point + joint perturbation mechanism

| # | 단계 | 디테일 |
|---|------|--------|
| 4.1 | **K-batch trajectory generation** | Curobo MotionPlanner 가 subgoal seed g 에 대해 K개 candidate trajectory batch 생성 |
| 4.2 | **Via-point insertion** | skill 중간에 1~`max_vias` via-point 추가 (현재 `max_vias=2`, commit 82393ec 에서 3→2 — trajopt fail 감소) |
| 4.3 | **Joint perturbation** | via-point joint config 에 noise → IK 다양화, arm-only 5-DoF (gripper 제외) |
| 4.4 | **Joint→servo 변환** | `JointServoConverter`: curobo joint radian → LeRobot servo position (±100), paradigm A.3 |
| 4.5 | **EE FK + dct_target_ee** | 변환된 joint chunk 에 FK 적용 → EE pose chunk → `ee_delta_dct_from_poses` → (50, 6) EE delta DCT |

### 흐름도

```text
subgoal seed g
   ↓
[Curobo MotionPlanner] ── K-batch trajopt
   ↓
trajectory list (arm-only 5-DoF, radian)
   ↓ via-point + joint perturbation
candidate joint chunk (T, 5)
   ↓ joint→servo (A.3)             ↓ EE FK
proprio/action (servo space)        EE pose chunk (T, 6)
   ↓ pad gripper                       ↓ ee_delta_dct_from_poses
joint DCT  (50, 6)                   EE delta DCT (50, 6)
   ↓ for U_VLA                          ↓ for M_MI / VectorDB
```

### Paper 와의 차이

Paper §3.4 는 "skill-level trajectory perturbation" 한 줄로 추상화 — backend, via-point 수, joint noise scale, FK chain, dimensionality 정합화 모두 누락.

---

## 5. VLA 임베딩 디테일 (D 카테고리 통합)

| # | 갭 | 실제 | Paper |
|---|----|------|-------|
| 5.1 | **DCT_50 정의** | top-50 low-frequency coefficient (L0=50), chunk_size 동기화, skill 단위 = single sample | "descriptor-space proxy" — DCT 자체 미언급 |
| 5.2 | **두 DCT 공존 (목적 분리)** | joint DCT (50×6, VLA target + U_VLA) + EE delta DCT (50×6=300, action_descriptor + M_MI) | 단일 descriptor 가정 |
| 5.3 | **arm-only 5-DoF 통일** | gripper 축 제외, LeRobot 6-DoF 는 패딩으로만 호환 | action_dim 추상 |
| 5.4 | **U_VLA mode="dct" R=1 강제** | `sigma` 옵션 시 deterministic single-step time | §3.4 식 (188): R회 stochastic 평균 |
| 5.5 | **State key φ_VLA(o_τ, I) black box** | SmolVLA-DCT forward → encoder hidden ⊕ proprio (965 dim), 학습 spec (dataset/step/loss) 명시 0 | 식 (170) 만 |

### 두 DCT 의 dimensionality table

| 용도 | 표현 | Shape | 출처 |
|------|------|-------|------|
| VLA 학습 target | joint DCT | (50, 6) = 300 | `method3/dct/skill_dataset.py: traj_to_dct(seg_actions, L0=50)` |
| U_VLA denoise space | joint DCT | (50, 6) = 300 | `vla_informativeness.LeRobotVLAInformativenessScorer(mode="dct")` |
| VectorDB action_descriptor | **EE delta DCT** | (50, 6) = 300 | `method3/dct/ee_features.py: ee_delta_dct_from_poses` |
| M_MI 계산 (ΔH_A, ΔH_A|S) | **EE delta DCT** | (50, 6) = 300 | `mi_selector.action_descriptors` |

### Translation invariance motivation

EE delta DCT 는 `np.diff(poses, axis=0)` 후 DCT — 즉 **position 의 절대좌표 제거** 후 motion shape 만 보존.
→ 같은 motion 을 다른 위치에서 수행한 trajectory 는 같은 descriptor 값 → table 1 (state coverage) 와 table 2 (action diversity) 가 분리 측정 가능.

---

## 6. Paper 작성 우선순위 (두 항목 핵심)

### ★★★★★ 항목 1: §3.4 VLA 학습 sub-section 신설

**내용**
- DCT paradigm 자체 (skill segment → DCT_50 압축)
- L0=50 정의 + chunk_size 동기화
- joint DCT vs EE delta DCT 분리 motivation
- SmolVLA-DCT 학습 spec (dataset/step/target/loss)

**왜 ★★★★★**
> Paper §3.4 의 "Phase1-trained VLA" 만 보고는 어떤 VLA 모델 / 어떤 target / 어떤 dataset / 얼마나 학습인지 알 수 없어 method 재현 불가.

### ★★★★★ 항목 2: §3.4 의 e_τ 식 옆 두 descriptor 분리 명시

**내용**
- `state_key` (절대좌표, image-conditioned, neighbor retrieval 용) 정의
- `action_descriptor` (EE delta DCT, translation-invariant, M_MI 용) 정의
- 분리 이유 (motion shape diversity vs state-neighborhood retrieval) 식 단위 기록

**왜 ★★★★★**
> Paper 의 단일 descriptor 가정이 실제로는 두 목적으로 명시적 분리됐고, dimensionality 도 다름 (state 965 vs action 300). 두 항목 추가 없이 paper 만 보고 구현하면 shape mismatch 와 retrieval semantic 부정합으로 reproduce 실패.

### 권장 추가 표 (paper 본문 hyperparameter table)

| Symbol | Meaning | Default | Source |
|--------|---------|---------|--------|
| `L0` | DCT coefficient 수 | 50 | `skill_dataset.py:109` |
| `D_e` | state key dimension | 965 | SmolVLA hidden + proprio |
| `D_z` | action descriptor dimension | 300 | EE delta DCT 50×6 |
| `k_min` | covered window 판정 최소 neighbor | 3 | `conditional_ambiguity.py:92` |
| `s_min` | s_a floor | 1e-3 | `conditional_ambiguity.py:93` |
| `ρ_m` | state-neighborhood radius | config | yaml |
| `τ_MI` | M̃_MI threshold (z-score) | 0.0 | `mi_selector.py:104` |
| `β, λ` | M_MI weights | config | yaml |
| `max_vias` | via-point 최대 수 | 2 | commit 82393ec |
| `R` | stochastic denoise 평가 횟수 | 1 (dct mode 강제) | `vla_informativeness.py:118` |

---

## 부록: 카테고리별 전체 갭 매트릭스

### A. 🔴 Paper 에 0% 반영된 시스템 컴포넌트

| 갭 | 실제 위치 |
|----|-----------|
| gRPC server-client 아키텍처 (H100 ↔ robot) | `grpc_server/` |
| Server-side accept_to_buffer | `grpc_server/server.py` |
| Episode lifecycle (reconcile, resume) | `method3/episode_lifecycle/` |
| TRUE-only staging | `subgoal_selector.stage_executed` |

### B. 🟠 Phase1 (§3.3) — 본문 [1] 참조

### C. 🟠 Phase2 (§3.4) — 본문 [2], [3] 참조 + 추가

| 갭 | Paper |
|----|-------|
| β, λ default 값 | 식 (162) 정의만 |
| fallback (eligible=∅): argmax M_MI + `accepted=False` | 식 (234) 만 |

### D. 🟠 VLA 임베딩 — 본문 [5] 참조

### E. 🟡 Storage / Engineering

| 갭 | 실제 |
|----|------|
| Vector DB 구조 (append-only index.jsonl + arrays/.npz) | `method3/storage/raw_dataset.py` |
| D_phase1_raw / D_phase2_raw (re-embedding source-of-truth) | paper: 한 entity |
| observation_ref pointer 규약 | paper: 없음 |

### F. 🟡 Operational policy

| 갭 | 실제 |
|----|------|
| Stack task override (judge=FALSE 라도 TRUE 강제) | `execution_forward_and_reset.py:3534-3559` |
| Reset edge-margin 검증 (bbox 전체 green margin 안) | `code_gen_lerobot/reset_execution/workspace.py:236-247` |
| Object overlap / collision rule (allow_overlap, max_iou) | `workspace.py: generate_random_position` |

---

## 변경 로그

| 날짜 | 변경 |
|------|------|
| 2026-05-24 | 최초 작성 — paper §3.3 ~ §3.5 + system layers 갭 분석 |
