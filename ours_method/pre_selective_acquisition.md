# Pre-selective Real-world Data Acquisition (사전선별적 수집)

> Method 3. `stochastic_perturbation.md` 의 (2) skill-level / (3) subgoal-level
> perturbation 으로 생성된 candidate trajectory 풀에서, **실행 전(online)** 에
> 학습에 유용한 후보 하나만 골라 실행 + 수집하는 단.

---

## 문제

자동화된 real-world 수집은 가능해졌지만, 모든 생성 trajectory를 무차별 수집·학습하면 두 측면에서 비효율:

1. **비용** — 실세계 수집은 로봇 embodiment를 완전 점유(하드웨어 마모, 운영 비용). 중복 trajectory까지 저장·학습하면 storage / GPU cost 가 낭비됨.
2. **모델 학습** — diversity 자체가 항상 유용한 supervision은 아님. 중복은 정보이득이 없고, 유사 obs/instruction 에서 incompatible action mode 가 섞이면 **action ambiguity** 가 증가해 flow-matching / diffusion 기반 VLA 의 학습·closed-loop 실행이 불안정해진다.

---

## 기대 효과

- 수집된 trajectory 단위당 정보이득(IG) 극대화 → 로봇 점유시간·storage·GPU 비용 절감
- VLA policy entropy 가 비정상적으로 커지는 trajectory 차단 → flow/diffusion VLA 의 mode collapse / action chunk instability 방지
- "다양성 최대화" 가 아니라 **useful diversity 의 선택적 누적** 이 목표

---

## (1) Diversity 의 3-분류

생성된 후보 $ξ$ 에 대해 (학습된 dataset $D$, 미학습 buffer $B$, 기학습 VLA $π_0$) 를 기준으로 다음과 같이 분류:

| 분류 | 조건 | 의미 |
|---|---|---|
| **Redundant Diversity** | $IG(ξ;D,B) ≈ 0$ | $D$ / $B$ 가 이미 커버 — 추가 학습정보 없음 |
| **Ambiguous Novelty** | $IG(ξ;D,B) > ε_{IG}\ \land\ H_{π_0}^{cand} > ε_H$ | novel 하지만 유사 obs 에서 action mode 가 갈라져 ambiguity ↑ |
| **Useful Diversity** | $IG(ξ;D) > δ_{IG}\ \land\ H_{π_0} < ε_H$ | novel + action-consistent — **수집 대상** |

→ 목표는 diversity 최대화가 아니라 **Useful Diversity Acquisition**.

---

## (2) 두 축의 측정 기준

선별 기준은 두 축의 동시 만족:

### (2-1) Information Gain (IG) — "유용한 다양성" 측정

- 기학습 VLA 의 candidate 에 대한 uncertainty
- 기수집 dataset / buffer 대비 novelty (예: latent 거리, k-NN, model disagreement)

### (2-2) Action-Consistency (AC) — "다양성이 ambiguity 로 변질" 방지

- 기학습 VLA 의 action chunk entropy: $H_{π_0}(A^{1:H} \mid O, I)$
- 유사 obs/instruction 에서 action chunk 분포가 과도하게 multi-modal 인 후보는 거부

**핵심**: IG 단독 사용 시 ambiguous novelty 가 통과돼버린다. AC 가 그 gate.

→ 즉 Method 3 = **Information Gain Maximization with Policy Entropy Constraint**

---

## (3) Skill-wise Online Selection

전체 rollout 을 뽑아 사후 평가하는 방식이 **아니라**, 매 skill step 에서 생성된 candidate segment 중 하나를 **사전 선택 후 실행**.

```
[skill m 진입]
  ├── 현재 위치 → 목표 위치 도달 가능한 후보 trajectory 생성
  │     S_m = {ξ_m^1, ξ_m^2, …, ξ_m^K}
  │     (perturbation (2) skill-level 의 plan_batch + (3) subgoal offset 으로 풀 구성)
  │
  ├── 각 ξ_m^j 에 대해:
  │     IG_j  = IG(ξ_m^j ; D, B)
  │     H_j   = H_{π_0}(A^{1:H} | O_j, I)
  │
  ├── 필터:
  │     IG_j > δ_IG  ∧  H_j < ε_H
  │     (∧ optional: buffer redundancy 기준)
  │
  ├── ξ_m^* = argmax over surviving candidates  (policy: top-IG / Pareto / …)
  │
  ├── 실행 ξ_m^* → target 도달
  └── (B 에 적재) → 다음 skill m+1 에서 반복
```

- 풀 source: `stochastic_perturbation.md` (2) skill-level OMPL ensemble + (3) subgoal Gaussian offset
- 선택 단위: **skill segment** (전체 rollout 아님). 매 skill 마다 독립 선택.
- 실행된 segment 만 수집 → 사후 폐기 비용 없음

---

## (4) 파일 목록 *(TODO — 미구현)*

| 경로 (예상) | 목적 |
|---|---|
| `acquisition/__init__.py` | top-level export |
| `acquisition/scorer.py` | `InformationGainScorer`, `ActionConsistencyScorer` — Protocol 기반, 합성 가능 |
| `acquisition/policy.py` | TopIG / Pareto / threshold gate 선택 정책 |
| `acquisition/online_selector.py` | skill-step 별 candidate 풀 입력 → ξ\* 출력 |
| `skills/skills_lerobot.py` | (변경) `move_to_position` 안 OMPL `plan_batch` 결과를 `online_selector` 로 라우팅 — 현재는 RNG 무작위 선택 |
| `execution_forward_and_reset.py` | (변경) 기학습 VLA(`π_0`) + dataset/buffer reference 를 `online_selector` 에 주입 |
| `pipeline_config/recording_config_ws*.yaml` | (변경) `acquisition:` 섹션 ($δ_{IG}$, $ε_H$, scorer/policy 설정) |

---

## (5) 설정 스키마 *(예상)*

```yaml
acquisition:
  enabled: false              # true → skill-wise pre-selective acquisition 활성화
  scorer:
    information_gain:
      kind: vla_uncertainty   # vla_uncertainty | dataset_knn | bc_disagreement
      threshold_delta: 0.10   # δ_IG
    action_consistency:
      kind: policy_entropy    # H_{π_0}(A^{1:H} | O, I)
      threshold_epsilon: 0.50 # ε_H
  policy:
    kind: top_ig              # top_ig | pareto | uniform_within_band
    k: 1                      # 실행할 segment 수 (skill 당)
  buffer:
    enabled: true             # B 에 적재된 미학습 후보까지 redundancy 비교에 사용
```

---

## (6) Perturbation 과의 관계

| 단계 | 역할 | 출력 |
|---|---|---|
| (2) Skill-level OMPL ensemble | "어떻게 갈지" 모드 다양성 | candidate 풀 (algo × seed) |
| (3) Subgoal-level Gaussian | "어디로 갈지" 위상학적 다양성 | candidate 풀 (offset 분포) |
| **Method 3 (이 문서)** | **풀에서 useful 만 선별** | 실행 + 적재되는 단일 segment |

→ Perturbation 은 풀의 **폭** 을 만들고, Method 3 은 그 폭에서 **유용한 단면** 만 추출. 직교 단(orthogonal stages).
