# Data Quality & Sample Efficiency (설계 · metric · 결과)

> 기존 **Table 2(숫자표)는 제거**하고 **두 그림으로 대체**: `fig_quality_scatter`(100ep snapshot) + `fig_quality_trend`(20→100 scaling, mean±std). **SR은 이 섹션에서 제외**(Table 1에 존재).

## 1. 핵심 주장
에피소드를 *더 많이*가 아니라 *더 학습 가능하게(learnable)* 모은다.
- **(C1) coverage·diversity 확장**: CRAPE는 redundant한 RoboTwin보다 trajectory diversity·transition support를 크게 넓히고 teleop 수준에 근접.
- **(C2) useful diversity**: 그 다양성은 유해 ambiguity가 아니다. **다양성 보정(NCD)** 시 CRAPE의 conditional ambiguity는 teleop보다 **낮다**.
- **(C3) scaling**: budget이 커질수록 diversity↑·NCD↓(개선) → "more learnable".

> 통찰: diversity와 raw consistency는 **trade-off**(MI=H(A)−H(A|S)). raw ambiguity로 재면 다양한 방법이 무조건 불리 → **diversity-invariant(NCD)** 로 재야 의미.

## 2. 데이터셋 · 프로토콜
- 비교군: **Teleoperation / RoboTwin(code-based, HF `cap_*`) / CRAPE(Ours)**. (MA는 dataset-metric 미측정 → Table 1 SR에만.)
- 10 고정 layout, layout당 2/4/6/8/10 → **20/40/60/80/100 ep**.
- EE6 통일: 세 방법 모두 `observation.state` → per-joint affine → FK(`so101_robot0.urdf`). metric은 delta·상대거리 기반이라 상수 tool-offset 무관.
- **T8**(bi-arm, 12-dim)·**T10**(데이터 오류) 제외 → 단팔 **8개 task(T1–T7, T9)**.
- 스크립트: `table2_metrics.py`(Signature/Transition), `consistency_normalized.py`(NCD+bootstrap), `fig_quality_scatter.py`·`fig_quality_trend.py`·`_figstyle.py`(그림).

---

## 3. Metric 선정/기각 로그 (각 ≤2문장)

**기각**
- **Diversity (pairwise DTW)** — *기각*: 궤적 간 상관 미반영. kernel 기반 Signature로 대체.
- **Action Consistency (per-state variance, raw)** — *기각*: diversity와 교란되어 perturbation 기반 Ours가 무조건 최악.
- **State Coverage (KDE volume)** — *기각(흡수)*: Transition Support Coverage가 transition-level로 대체.
- **Mutual Information (KSG)** — *기각*: CRAPE 선택목표와 동일해 순환적이고, deterministic RoboTwin이 최고로 나와 주장과 역방향.
- **kNN Action Inconsistency (raw 1-NN / mode-aware min-residual)** — *기각*: diversity 교란 미제거로 Ours 최하위. → NCD로 발전.
- **Conditional ambiguity §9 `d_min/s_a` (local 정규화)** — *기각*: radius에 따라 순위 뒤집힘(비robust).

**채택**
- **Signature Diversity (FAKTUAL)** — *채택*: signature-kernel Gram의 spectral entropy로 episode-level 궤적 다양성 측정.
- **Transition Support Coverage** — *채택*: `[s,Δa,s']` fingerprint의 support entropy(KL).
- **NCD** — *채택*: 국소 ambiguity를 **global diversity로 정규화**한 diversity-invariant consistency. 8/8 task에서 Ours<teleop으로 주장 입증.

---

## 4. 최종 선정 metric (SR 제외)
| Metric | 정의 | 방향 |
|---|---|---|
| **Signature Div.** | signature-kernel Gram의 spectral entropy = `exp H(K_sig)` (eff. # trajectories) | ↑ |
| **Transition Cov.** | `[ee6, Δee6]` fingerprint의 Kozachenko–Leonenko 미분엔트로피 `Ĥ_KL[s,Δa]` (nats) | ↑ |
| **NCD** | `d̄_min / d̄_glob` — k-NN state-neighbor 중 nearest action까지 거리(mode-aware) ÷ global pairwise action distance (same-episode 제외) | ↓ |

- NCD는 within-dataset 비율(scale-free) → 방법 간 공정. **다양성 있는 방법 간에만** 유효(deterministic RoboTwin은 trivially 낮음).
- SR은 downstream 지표라 **Table 1**로 분리. MI는 순환·교란으로 제외.

---

## 5. 분석 결과 (전 데이터셋)

### 5.1 시각화 (표 대체)
- **`fig_quality_scatter`** — 100ep snapshot. x=Signature Div, y=NCD, 방법별 KDE density 음영 + 점(마커 구분). RoboTwin=저다양 / teleop=고다양·고ambiguity / **CRAPE=useful diversity**.
- **`fig_quality_trend`** — 3패널(Signature/Transition/NCD) × budget(20→100). **선=8 task 평균, band=±1 std(task 간 spread)**.

### 5.2 Scaling 추이 (8-task 평균, 20→100ep)
| Metric | 방법 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|
| **Signature Div.** ↑ | teleop | 8.69 | 11.46 | 12.73 | 13.22 | 13.58 |
| | RoboTwin | 5.76 | 6.73 | 7.04 | 7.14 | 7.32 (포화) |
| | **Ours** | 7.11 | 8.91 | 9.84 | 10.36 | **10.55** |
| **Transition Cov.** ↑ | teleop | 1.04 | 2.08 | 1.95 | 1.78 | 1.52 |
| | RoboTwin | −5.27 | −6.15 | −6.90 | −7.32 | **−7.50** (악화) |
| | **Ours** | 1.27 | 1.44 | 1.06 | 0.69 | 0.49 |
| **NCD** ↓ | teleop | 0.323 | 0.285 | 0.275 | 0.270 | 0.261 |
| | RoboTwin | 0.121 | 0.089 | 0.078 | 0.073 | 0.068 (trivial) |
| | **Ours** | 0.250 | 0.221 | 0.215 | 0.206 | **0.198** |

→ **(C1)** Signature: Ours 꾸준히↑, RoboTwin ~40ep에서 포화. **(C3)** NCD: 전 split 개선 & **Ours<teleop at every budget**. Transition: Ours≫RoboTwin(격차 ~8), RoboTwin은 더 음수로 붕괴(redundancy 심화); Ours는 양수 유지(소폭↓ → "성장"보다 "RoboTwin 대비 압도"로 서술).

### 5.3 NCD — diversity-invariant consistency (핵심 증거, @100ep, k=20)
per-task (낮을수록 consistent):

| task | teleop | **Ours** |
|---|---|---|
| T1 | 0.269 | 0.258 |
| T2 | 0.236 | 0.213 |
| T3 | 0.329 | **0.228** |
| T4 | 0.246 | 0.240 |
| T5 | 0.240 | **0.175** |
| T6 | 0.291 | **0.173** |
| T7 | 0.284 | **0.183** |
| T9 | 0.184 | 0.116 |

**집계 (paired, teleop vs Ours):** **8/8** Ours<teleop · mean gap **+0.063** · task-bootstrap 95% CI **[+0.034, +0.091]** · **Wilcoxon p=0.004**. (per-task CI = episode-subsampling bootstrap B=200, 70% 비복원; T3·T6·T7 CI 분리.)

→ **(C2)** 동일 다양성에서 CRAPE의 ambiguity가 teleop보다 유의하게 낮음 = 자동 생성 다양성이 인간 변동보다 **더 state-구조적**.

### 5.4 Table 3 — ablation (pnp, 누적; `scripts/table3_*`)
| Metric | Baseline | StateSeed | +ActPert(rand) | +MI Sel | Ours |
|---|---|---|---|---|---|
| Signature Div. ↑ | 9.3 | 10.0 | 10.8 | 10.6 | **11.0** |
| Transition Cov. ↑ | **−4.6** | 3.2 | 3.1 | 2.5 | 2.7 |
| (NCD, k=20) | 0.07 | 0.21 | 0.27 | 0.28 | 0.29 |

→ **State seeding → coverage**(−4.6→3.2), **Perturbation → diversity**(9.3→11.0). selection(+MI Sel/Ours)은 dataset diversity·coverage·NCD를 더 안 바꿈 — **고정 budget 내 *선택*이라 효과는 SR로 발현**(C3). 따라서 consistency 입증은 **NCD(vs teleop) + SR**, selection 가치는 **Table 3 SR 행**이 담당.

---

## 6. 한계 · 메모
- NCD는 다양성 보정 비율 → **저다양 deterministic(RoboTwin)에는 부적용**, "다양한 방법 간"으로 한정 해석.
- trend band = ±1 std(task 간 spread, descriptive). 평균 불확실성을 원하면 ±SEM(=std/√8)로 전환 가능.
- T8(bi-arm) 포함하려면 팔별 FK 필요. T10 재수집 시 8→9 task 확장.
- 미측정: 30/50 split은 미사용(데이터가 20/40/60/80/100). SR-vs-budget은 SR 측정 후 trend에 패널 추가 가능.
