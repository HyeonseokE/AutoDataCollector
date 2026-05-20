# Method3 Final Spec: Pre-selective Real-world Data Acquisition

## 0. 핵심 문제정의

Ours의 자동 데이터 취득 파이프라인은 지속적인 real-world data collection을 가능하게 한다. 그러나 명확한 기준 없이 모든 trajectory를 수집하는 것은 두 가지 측면에서 비효율적이다.

첫째, **비용 측면의 비효율**이다. 실세계 데이터 수집은 로봇 embodiment를 완전히 점유하고, 하드웨어 마모와 운영 비용을 수반한다. 또한 중복 trajectory까지 모두 저장하고 학습에 포함하면 storage cost와 GPU training cost가 불필요하게 증가한다.

둘째, **모델 학습 측면의 비효율**이다. 적절한 diversity는 distribution shift와 OOD robustness에 필요하지만, 단순히 rollout 수나 trajectory coverage를 늘리는 것이 항상 유용한 supervision을 의미하지는 않는다. 중복 trajectory는 정보 이득이 낮고, 유사한 observation / instruction에서 incompatible action mode가 섞인 trajectory는 action ambiguity를 증가시켜 flow-matching / diffusion 기반 VLA의 학습과 closed-loop execution을 불안정하게 만들 수 있다.

따라서 Method3의 목표는 다음과 같다.

$$
\boxed{
\text{현재 buffer가 충분히 커버하지 못한 state/action region을 확장하면서도, 유사 state에서 action ambiguity를 과도하게 증가시키지 않는 trajectory를 선별 수집한다.}
}
$$

---

# 1. Offline Curation과 Online Acquisition의 차이

기존 robot data curation 방법은 이미 수집된 demonstration pool에서 학습에 유용한 demonstration을 선별하는 **offline filtering problem**을 다룬다. 이 setting에서는 데이터셋의 state coverage가 이미 결정되어 있으며, curation method가 직접 제어하는 대상이 아니다.

따라서 DemInf와 같은 방법은 주어진 pool 안에서 behavior cloning이 잘 학습할 수 있는 empirical expert distribution을 정의하는 데 집중한다. 이를 위해 mutual information을 데이터셋 quality metric으로 사용한다.

$$
I(S;A)=H(A)-H(A|S)
$$

여기서:

$$
H(A)
=
\text{dataset 전체의 action diversity}
$$

$$
H(A|S)
=
\text{state가 주어졌을 때 action ambiguity}
$$

즉, 좋은 데이터셋은 전체 action은 다양하지만, 같은 state에서는 action이 예측 가능하고 일관적인 데이터셋이다.

하지만 우리의 setting은 다르다. 우리는 이미 모인 pool에서 subset을 고르는 것이 아니라, real-world interaction을 통해 dataset 자체를 점진적으로 구축하는 **online acquisition problem**을 다룬다.

즉, 기존 curation은 이미 모인 pool에서 어떤 데이터를 남길지 묻는 문제지만, 우리의 acquisition은 다음에 어떤 state-action region을 실제로 방문하고 추가할지를 결정하는 문제다.

따라서 기존 기준을 그대로 적용하기 전에, 먼저 충분한 state support를 만들어야 한다.

$$
\boxed{
\text{Phase1: 먼저 } H(S) \text{를 키워 state support를 만든다.}
}
$$

$$
\boxed{
\text{Phase2: 그 state support 안에서 } H(A)-H(A|S) \text{ 기준으로 action distribution을 정제한다.}
}
$$

---

# 2. Overall Framework: Two-phase Online Acquisition

Method3는 두 단계로 구성된다.

## Phase1: State Coverage Seeding

Phase1은 state coverage를 확보하는 warm-start 단계다.

$$
\boxed{
\text{Phase1 objective: } H(S)\uparrow
}
$$

아직 local action distribution이 충분하지 않으므로, Phase1에서는 다음 항을 안정적으로 추정할 수 없다.

$$
H(A|S)
$$

즉, cold-start 문제로 특정 state에 대한 action 샘플 분포가 아직 없다.

$$
p(A|S)
$$

따라서 Phase1에서는 action-consistency filter를 적용하지 않는다.

대신 다음 원칙을 따른다.

$$
\boxed{
\text{subgoal은 다양하게, path는 canonical하게}
}
$$

즉, state diversity는 subgoal diversity로 만들고, 같은 state에서 여러 action mode가 생기는 것을 막기 위해 skill trajectory는 최대한 단조로운 canonical path로 유지한다.

---

## Phase2: MI-based Consistent Diversity Acquisition

Phase2는 Phase1에서 만들어진 state support 안에서, 생성된 후보 trajectory 중 학습에 유용한 trajectory만 선별적으로 수집하는 단계다.

> **갱신 (`phase1_seed_anchor_logic.md`)**: Phase2는 새로운 subgoal/state region을 탐색하지 않는다. Phase1이 확보한 skill-wise seed subgoal 집합 $\mathcal{G}_{seed}^{(m)}$ 을 anchor로 고정하고, 각 seed 주변에서 action variation 후보만 생성·큐레이션한다 (§7 참조).

Phase2의 핵심 목표는 다음과 같다.

$$
\boxed{
\text{action coverage는 확장하되, 유사 state에서 action ambiguity는 증가시키지 않는다.}
}
$$

---

# 3. Data Storage Overview

Method3에서는 데이터 저장 구조를 두 층으로 분리한다.

$$
\boxed{
\text{Raw Trajectory Dataset = source of truth}
}
$$

$$
\boxed{
\text{Vector DB = retrieval / scoring을 위한 searchable index}
}
$$

Raw dataset에는 re-embedding과 action descriptor 재계산에 필요한 원본 정보를 저장한다.

Vector DB에는 retrieval key, action descriptor, metadata, raw dataset pointer를 저장한다.

---

## 3.1 Raw Dataset에 저장할 정보

Phase1과 Phase2 raw dataset은 다음 정보를 복원 가능하게 저장해야 한다.

```text
episode_id
phase
skill_id
instruction
subgoal
time_index
observation
proprioception
action_chunk
raw_action_sequence
planner_or_generator_type
success_flag
validity_flag
environment_metadata
object_metadata
```

수식적으로는 다음처럼 쓸 수 있다.

$$
D_{\mathrm{raw}}
=
\{
(o_\tau,p_\tau,I,g,m,A_{\tau:\tau+H-1},meta_\tau)
\}_{\tau}
$$

re-embedding에 필요한 최소 정보는 다음이다.

$$
(o_\tau,I,p_\tau)
$$

action descriptor 재계산에 필요한 정보는 다음이다.

$$
A_{\tau:\tau+H-1}
$$

skill-wise indexing과 분석에 필요한 정보는 다음이다.

$$
(m,g,meta_\tau)
$$

---

# 4. Phase1: State Coverage Seeding

## 4.1 Phase1에서 단조로운 trajectory를 사용하는 이유

skill-traj perturbation은 양면성이 있다.

첫째, 여러 형상의 경로를 만든다는 점에서 state coverage를 키울 수 있다.

$$
H(S)\uparrow
$$

둘째, 같은 state 근처에서 여러 action mode를 만들 수 있기 때문에 action ambiguity를 키울 수 있다.

$$
H(A|S)\uparrow
$$

그런데 Phase1에서는 아직 local action distribution이 충분하지 않으므로, 이 ambiguity를 안정적으로 계산하거나 제어할 수 없다.

따라서 Phase1에서는 state diversity를 키울 때 **subgoal diversity만 적극적으로 활용**하고, skill trajectory는 가능한 한 단조로운 canonical interpolation path를 사용한다.

---

## 4.2 State-Coverage-Guided Subgoal Sampling

Phase1에서는 여러 subgoal 후보를 만들고, 각 후보까지 canonical preview trajectory를 생성한 뒤, 해당 후보가 도달하는 terminal state region이 기존에 덜 커버된 영역인지 평가한다.

최종 선택식은 다음과 같다.

$$
g^*
=
\arg\max_{g'_j\in\mathcal{G}_{valid}}
G_S^{goal}(g'_j)
$$

선택된 subgoal까지는 canonical interpolation trajectory로 실행한다.

$$
\xi^*
=
\mathrm{InterpPlan}(S_t,g^*)
$$

여기서 valid subgoal set은 다음과 같다.

$$
\mathcal{G}_{valid}
=
\{
g'_j\in\mathcal{G}
\mid
\mathrm{reachable}(g'_j),\ \mathrm{safe}(g'_j),\ \mathrm{is\_transit}(g'_j)
\}
$$

---

# 5. Skill-wise Subgoal Buffer for Phase1

Phase1에서는 full skill-wise vector DB가 아니라, 별도의 **skill-wise subgoal buffer**를 사용한다.

$$
\boxed{
B_{g,t}^{(m)}
=
\text{skill } m \text{에서 이미 seed한 subgoal-side terminal state region들의 buffer}
}
$$

이 buffer는 Phase1에서 어떤 subgoal / terminal state region을 이미 방문했는지 추적하기 위한 lightweight coverage memory다.

Full skill-wise vector DB와는 목적이 다르다.

$$
\boxed{
B_{g,t}^{(m)}
\neq
B_t^{(m)}
}
$$

- \(B_{g,t}^{(m)}\): Phase1 subgoal-side state coverage tracking용 buffer
- \(B_t^{(m)}\): Phase2 state-action retrieval / MI-style scoring용 full vector DB

---

## 5.1 Skill-wise Subgoal Buffer가 필요한 이유

같은 subgoal 위치라도 skill이 다르면 의미가 달라질 수 있다.

예를 들어 같은 object 근처 위치라도:

- reach skill에서는 end-effector 접근 목표
- grasp skill에서는 grasp pose 근처 목표
- wipe skill에서는 contact / wiping target
- place skill에서는 placement target

처럼 해석된다.

따라서 subgoal buffer도 skill별로 구성한다.

$$
B_{g,t}
=
\{B_{g,t}^{(1)},B_{g,t}^{(2)},\dots,B_{g,t}^{(M)}\}
$$

각 skill \(m\)에 대해:

$$
B_{g,t}^{(m)}
=
\{c_i^{(m)}\}_{i=1}^{N_m}
$$

---

## 5.2 Subgoal Buffer Entry Format

각 entry는 full trajectory가 아니라, subgoal 근처 terminal region을 나타내는 정보를 저장한다.

```text
subgoal_buffer_entry = {
  skill_id: m,
  subgoal: g_i,
  terminal_region_key: h_i,
  end_state_keys: {h_i_tau | tau in T_end},
  episode_id: episode_id,
  start_t: start_t,
  end_t: end_t,
  success_flag: true,
  planner_type: "InterpPlan",
  phase: "phase1"
}
```

여기서 가장 중요한 항목은 다음이다.

- \(g_i\): 실제 선택된 subgoal
- \(h_i\): subgoal 근처 terminal region descriptor
- \(\{h_i^\tau\}\): 마지막 구간 terminal state descriptor set
- \(episode_id, start_t, end_t\): raw dataset pointer

---

## 5.3 Terminal Region Descriptor

Phase1에서는 전체 path를 평가하지 않고, subgoal 근처 마지막 구간만 본다.

$$
\mathcal{T}_{end}
=
\text{last 20\% of preview trajectory}
$$

각 terminal preview state를 descriptor로 변환한다.

$$
\hat h_\tau^j
=
\phi_{goal}(\hat S_\tau^j,g'_j,m)
$$

초기 구현에서는 VLA embedding 대신 geometric descriptor를 사용한다.

예시:

$$
\hat h_\tau^j
=
[q_\tau^j,\ x_{\tau}^{EE,j},\ x_{\tau}^{EE,j}-g'_j,\ m]
$$

subgoal buffer에는 마지막 구간 descriptor의 평균을 저장할 수 있다.

$$
h_i
=
\frac{1}{|\mathcal{T}_{end}|}
\sum_{\tau\in\mathcal{T}_{end}}
h_i^\tau
$$

또는 마지막 구간 descriptor set 자체를 함께 저장할 수 있다.

$$
\{h_i^\tau\}_{\tau\in\mathcal{T}_{end}}
$$

---

## 5.4 Subgoal Candidate Scoring

각 후보 \(g'_j\)에 대해 canonical preview trajectory를 만든다.

$$
\xi_j
=
\mathrm{InterpPlan}(S_t,g'_j)
=
\{\hat S_1^j,\hat S_2^j,\dots,\hat S_T^j\}
$$

마지막 구간만 평가한다.

$$
\mathcal{T}_{end}
=
\text{last 20\% of preview trajectory}
$$

각 마지막 구간 state를 descriptor로 바꾼다.

$$
\hat h_\tau^j
=
\phi_{goal}(\hat S_\tau^j,g'_j,m)
$$

같은 skill의 subgoal buffer와 비교한다.

$$
B_{g,t}^{(m)}
=
\{h_1,h_2,\dots,h_N\}
$$

각 descriptor에 대해 kNN 평균 거리를 계산한다.

$$
d_k^g(\hat h_\tau^j,B_{g,t}^{(m)})
=
\frac{1}{k}
\sum_{i\in\mathcal{N}_k^g(\hat h_\tau^j)}
d_g(\hat h_\tau^j,h_i)
$$

거리 scale을 정규화하기 위해 subgoal buffer 내부의 평균 nearest-neighbor 거리를 사용한다.

$$
\bar d_{NN}^{g}(B_{g,t}^{(m)})
=
\frac{1}{|B_{g,t}^{(m)}|}
\sum_{h_i\in B_{g,t}^{(m)}}
\min_{l\ne i}
d_g(h_i,h_l)
$$

초기 buffer가 너무 작다면 minimum scale을 둔다.

$$
s_g^{(m)}
=
\max
\left(
\bar d_{NN}^{g}(B_{g,t}^{(m)}),
s_{\min}^{g}
\right)
$$

point-level novelty는 다음과 같다.

$$
n_g(\hat h_\tau^j)
=
\left[
\log
\frac{
d_k^g(\hat h_\tau^j,B_{g,t}^{(m)})
}{
s_g^{(m)}+\epsilon
}
\right]_+
$$

subgoal 후보의 score는 마지막 구간 novelty 평균으로 계산한다.

$$
G_S^{goal}(g'_j)
=
\frac{1}{|\mathcal{T}_{end}|}
\sum_{\tau\in\mathcal{T}_{end}}
n_g(\hat h_\tau^j)
$$

최종 선택은 다음과 같다.

$$
g^*
=
\arg\max_{g'_j\in\mathcal{G}_{valid}}
G_S^{goal}(g'_j)
$$

$$
\xi^*
=
\mathrm{InterpPlan}(S_t,g^*)
$$

---

## 5.5 Subgoal Buffer Update

선택된 subgoal \(g^*\)에 대해 canonical trajectory를 실제 실행한 뒤, 성공한 경우에만 subgoal buffer를 업데이트한다.

실행 trajectory의 마지막 구간을 가져온다.

$$
\mathcal{T}_{end}^{exec}
=
\text{last 20\% of executed trajectory}
$$

각 terminal state를 descriptor로 변환한다.

$$
h_\tau^*
=
\phi_{goal}(S_\tau^{exec},g^*,m)
$$

대표 terminal region key를 만든다.

$$
h^*
=
\frac{1}{|\mathcal{T}_{end}^{exec}|}
\sum_{\tau\in\mathcal{T}_{end}^{exec}}
h_\tau^*
$$

그 다음 skill-wise subgoal buffer에 추가한다.

$$
B_{g,t+1}^{(m)}
=
B_{g,t}^{(m)}
\cup
\{c_*^{(m)}\}
$$

실패한 trajectory는 subgoal coverage seed로 쓰지 않는다.

이렇게 누적된 skill-wise subgoal buffer의 subgoal들이 Phase2의 seed anchor 집합 $\mathcal{G}_{seed}^{(m)}$ 가 된다 (§7, `phase1_seed_anchor_logic.md`).

---

## 5.6 Cold-start 처리

Phase1 초반에는 \(B_{g,t}^{(m)}\)가 비어 있거나 매우 작을 수 있다.

기본 구현에서는 각 skill마다 최소 seed 수를 채울 때까지 random valid subgoal warm-start를 사용한다.

$$
|B_{g,t}^{(m)}|<N_{seed}
\Rightarrow
g^*\sim \mathcal{G}_{valid}
$$

필요하면 farthest-first initialization을 ablation으로 사용할 수 있다.

$$
g^*
=
\arg\max_{g'_j\in\mathcal{G}_{valid}}
d_g(\hat h^j,B_{g,t}^{(m)})
$$

---

# 6. Phase1 종료 후 Full Skill-wise Vector DB 구축

Phase1 중에는 다음 두 가지가 병행된다.

$$
\boxed{
\text{Skill-wise subgoal buffer: online subgoal selection용}
}
$$

$$
\boxed{
\text{Phase1 raw dataset: Phase1 VLA 학습 및 full vector DB re-embedding용}
}
$$

Phase1 종료 후에는 다음 절차를 따른다.

1. Phase1 raw dataset으로 VLA encoder 또는 adapter를 학습한다.
2. Phase1-trained encoder를 freeze한다.
3. Phase1 raw dataset을 다시 읽어 re-embedding한다.
4. full skill-wise vector DB seed를 구축한다.

state retrieval key는 다음과 같다.

$$
e_i^{vla}
=
\phi_{\mathrm{VLA}}^{(1)}(o_i,I_i)
$$

$$
e_i
=
[e_i^{vla};p_i]
$$

action descriptor는 raw action chunk에서 계산한다.

$$
z_i^a
=
\psi(A_{i:i+H-1})
$$

Phase1 seed vector DB는 다음과 같다.

$$
P_{\mathrm{phase1}}^{(m)}
=
\{(e_i,z_i^a,ref_i,meta_i)\}_{i\in D_{\mathrm{phase1}}^{(m)}}
$$

---

# 7. Phase2 Mutual Information-based Candidate Selection

> **Phase2 Seed Anchoring (`phase1_seed_anchor_logic.md`)** — Phase2는 새 subgoal을 탐색하지 않고, Phase1 subgoal buffer가 모은 seed subgoal 집합 $\mathcal{G}_{seed}^{(m)}=\{g_1^{(m)},\dots,g_N^{(m)}\}$ 을 anchor로 쓴다. 각 seed $g$ 주변에서 skill-level trajectory perturbation으로 action variation 후보 $\Xi_t^{(m)}(g)$ 를 만들고, 그 안에서 MI-style score로 큐레이션한다. 이로써 $\widehat{\Delta H}_A$ 의 증가가 state novelty가 아니라 action-level diversity임이 보장되어 $Q_2$ 의 해석이 명확해진다.

## 7.1 Online MI Gain

Phase2에서는 후보 trajectory를 현재 buffer에 추가했을 때 dataset-level mutual information이 얼마나 증가하는지를 본다.

$$
Gain_{MI}(D_t,\xi)
=
I(D_t\cup\{\xi\})-I(D_t)
$$

dataset-level mutual information은 다음과 같다.

$$
I(D)=H_D(A)-H_D(A|S)
$$

따라서 다음처럼 분해된다.

$$
Gain_{MI}(D_t,\xi)
=
\Delta H_A(D_t,\xi)
-
\Delta H_{A|S}(D_t,\xi)
$$

실제 continuous trajectory entropy를 직접 추정하기 어렵기 때문에 proxy를 사용한다.

$$
\Delta H_A(D_t,\xi)
\approx
\text{action descriptor novelty}
$$

$$
\Delta H_{A|S}(D_t,\xi)
\approx
\text{local action support expansion in similar states}
$$

---

## 7.2 Phase2 Reference Buffer

Phase2에서는 full skill-wise vector DB를 사용한다.

$$
B_t^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t}^{(m)}
$$

각 vector DB entry는 다음과 같다.

$$
b_i^{(m)}
=
(e_i,z_i^a,ref_i,meta_i)
$$

여기서:

- \(e_i\): state retrieval key
- \(z_i^a\): action descriptor
- \(ref_i\): raw dataset pointer
- \(meta_i\): phase, skill id, subgoal, score 등 metadata

---

## 7.3 Candidate Representation

후보 skill trajectory segment를 다음처럼 둔다.

$$
\xi
=
\{(S_\tau,A_{\tau:\tau+H-1})\}_{\tau=1}^{T}
$$

각 window \(\tau\)에 대해 state key와 action descriptor를 만든다.

VLA는 observation과 instruction만 입력으로 받는다.

$$
e_\tau^{vla}
=
\phi_{\mathrm{VLA}}^{(1)}(o_\tau,I)
$$

proprioception은 VLA embedding 뒤에 concat한다.

$$
e_\tau
=
[e_\tau^{vla};p_\tau]
$$

따라서 최종 retrieval key는 다음과 같다.

$$
\boxed{
e_\tau
=
[\phi_{\mathrm{VLA}}^{(1)}(o_\tau,I);p_\tau]
}
$$

action chunk는 DCT descriptor로 변환한다.

$$
A_{\tau:\tau+H-1}\in\mathbb{R}^{50\times 6}
$$

$$
C_\tau
=
\mathrm{DCT}_{time}(A_{\tau:\tau+H-1})
$$

$$
\tilde C_\tau
=
C_\tau[1:K,:]
$$

$$
z_\tau^a
=
\psi(A_{\tau:\tau+H-1})
=
\mathrm{vec}(\tilde C_\tau)
$$

예를 들어 \(K=3\)이면:

$$
z_\tau^a\in\mathbb{R}^{18}
$$

---

# 8. Estimating Action Coverage Gain

skill \(m\)의 action descriptor buffer를 다음처럼 둔다.

$$
B_A^{(m)}
=
\{z_i^a \mid i\in B_t^{(m)}\}
$$

각 candidate action descriptor \(z_\tau^a\)에 대해 action-space kNN 거리를 계산한다.

$$
d_k^a(z_\tau^a,B_A^{(m)})
=
\frac{1}{k}
\sum_{i\in\mathcal{N}_k^a(z_\tau^a)}
d_a(z_\tau^a,z_i^a)
$$

buffer 내부의 typical action spacing을 계산한다.

$$
\bar d_{NN}^{a}(B_A^{(m)})
=
\frac{1}{|B_A^{(m)}|}
\sum_{z_i^a\in B_A^{(m)}}
\min_{l\ne i}
d_a(z_i^a,z_l^a)
$$

candidate action novelty는 다음과 같다.

$$
n_a(z_\tau^a)
=
\left[
\log
\frac{
d_k^a(z_\tau^a,B_A^{(m)})
}{
\bar d_{NN}^{a}(B_A^{(m)})+\epsilon
}
\right]_+
$$

trajectory-level action coverage gain은 top-quantile mean으로 계산한다.

$$
\widehat{\Delta H}_A(D_t,\xi)
=
\frac{1}{|\mathcal{T}_q|}
\sum_{\tau\in\mathcal{T}_q}
n_a(z_\tau^a)
$$

여기서 \(\mathcal{T}_q\)는 action novelty가 높은 상위 \(q\%\) window index이다.

---

# 9. Estimating Conditional Action Ambiguity Increase

## 9.1 Covered State Condition

각 candidate state key \(e_\tau\)에 대해 같은 skill buffer에서 state-neighbor를 검색한다.

$$
\mathcal{N}_{\rho_m}^{s}(e_\tau)
=
\{i\in B_t^{(m)}
:
d_s(e_\tau,e_i)<\rho_m
\}
$$

데이터가 적은 초기 setting을 고려하여 최소 neighbor 수는 다음처럼 둔다.

$$
\boxed{
k_{\min}^{(m)}=3
}
$$

따라서 covered window 조건은 다음과 같다.

$$
\boxed{
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|\ge3
}
$$

covered window set은 다음과 같다.

$$
\mathcal{T}_{covered}(\xi)
=
\left\{
\tau:
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|\ge3
\right\}
$$

covered window가 충분하지 않으면 해당 후보는 Phase2 MI selector로 신뢰성 있게 평가하지 않는다.

$$
|\mathcal{T}_{covered}(\xi)|<T_{\min}^{(m)}
\Rightarrow
\text{MI score로 신뢰성 있게 평가하지 않는다 (accept 제외)}
$$

`phase1_seed_anchor_logic.md` 갱신에 따라 Phase2는 state coverage를 확장하지 않으므로, under-covered 후보는 state-seeding 대상이 아니라 단순히 accept에서 제외한다. 후보가 모두 Phase1 seed anchor 주변에서 생성되므로 under-coverage는 정상 경로에선 드물다.

---

## 9.2 Local Action Support

covered state \(e_\tau\)에 대해 state-neighbor들의 action descriptor를 모은다.

$$
\mathcal{A}_{e_\tau}
=
\{z_i^a:
i\in\mathcal{N}_{\rho_m}^{s}(e_\tau)
\}
$$

이 set이 candidate state 주변의 local action support이다.

---

## 9.3 Nearest-support Expansion

후보 action \(z_\tau^a\)가 local action support 바깥에 있는지 계산한다.

가장 가까운 local action까지의 거리는 다음과 같다.

$$
d_{min}^{a}(z_\tau^a,\mathcal{A}_{e_\tau})
=
\min_{z_i^a\in\mathcal{A}_{e_\tau}}
d_a(z_\tau^a,z_i^a)
$$

local action support 내부의 typical spacing은 다음과 같다.

$$
\bar d_{NN}^{a}(\mathcal{A}_{e_\tau})
=
\frac{1}{|\mathcal{A}_{e_\tau}|}
\sum_{z_i^a\in\mathcal{A}_{e_\tau}}
\min_{z_l^a\in\mathcal{A}_{e_\tau},l\ne i}
d_a(z_i^a,z_l^a)
$$

local support가 너무 좁을 때 penalty가 과도하게 커지는 것을 막기 위해 scale floor를 사용한다.

$$
s_a(e_\tau)
=
\max
\left(
\bar d_{NN}^{a}(\mathcal{A}_{e_\tau}),
s_{\min}^{a}
\right)
$$

local action ambiguity increase는 다음과 같다.

$$
\Delta h_{A|S}(e_\tau,z_\tau^a)
=
\left[
\log
\frac{
d_{min}^{a}(z_\tau^a,\mathcal{A}_{e_\tau})
}{
s_a(e_\tau)+\epsilon
}
\right]_+
$$

해석은 다음과 같다.

- \(\Delta h_{A|S}\approx0\): 후보 action이 기존 local action support 안에 있음
- \(\Delta h_{A|S}>0\): 후보 action이 유사 state에서 기존 action support를 새롭게 확장함

---

## 9.4 Trajectory-level Conditional Ambiguity

candidate trajectory 전체의 conditional ambiguity increase는 covered windows에서만 aggregation한다.

초기 구현에서는 mean을 사용한다.

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
\frac{1}{|\mathcal{T}_{covered}(\xi)|}
\sum_{\tau\in\mathcal{T}_{covered}(\xi)}
\Delta h_{A|S}(e_\tau,z_\tau^a)
$$

보수적인 rejection이 필요하면 max aggregation을 ablation으로 사용한다.

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
\max_{\tau\in\mathcal{T}_{covered}(\xi)}
\Delta h_{A|S}(e_\tau,z_\tau^a)
$$

---

# 10. Multi-modal Action Support

Nearest-support 방식은 multi-modal action을 자연스럽게 허용한다.

유사 state 안에 여러 valid action mode가 있더라도, 후보 action이 그중 하나와 가까우면 penalty가 작다.

$$
d_{min}^{a}(z_\tau^a,\mathcal{A}_{e_\tau})
$$

반대로 기존 어느 mode와도 멀면 penalty가 커진다.

초기 구현에서는 clustering 없는 nearest-sample 방식을 사용한다.

필요하면 local action support를 clustering하여 nearest-mode 방식으로 확장할 수 있다.

$$
\mathcal{M}_{e_\tau}
=
\{M_1,\dots,M_L\}
$$

$$
d_{min}^{a}(z_\tau^a,\mathcal{M}_{e_\tau})
=
\min_{r\in\mathcal{M}_{e_\tau}}
d_a(z_\tau^a,\mu_r)
$$

---

# 11. Phase2 Final Score

최종 Phase2 score는 다음과 같다.

$$
Q_2(\xi)
=
\beta \widehat{\Delta H}_A(D_t,\xi)
-
\lambda \widehat{\Delta H}_{A|S}(D_t,\xi)
$$

최종 선택은 다음과 같다.

$$
\xi^*
=
\arg\max_{\xi\in\Xi_t^{(m)}}
Q_2(\xi)
$$

여기서 \(\Xi_t^{(m)}\)는 현재 skill \(m\)에서 생성된 후보 trajectory set이다.

---

# 12. Score Normalization and Accept Rule

\(Q_2\)의 scale은 task와 candidate set에 따라 달라질 수 있으므로 후보 batch 내부에서 normalize한다.

현재 round의 candidate set을 \(\Xi_t^{(m)}\)라고 하자.

$$
\mu_Q
=
\frac{1}{|\Xi_t^{(m)}|}
\sum_{\xi\in\Xi_t^{(m)}}
Q_2(\xi)
$$

$$
\sigma_Q
=
\sqrt{
\frac{1}{|\Xi_t^{(m)}|}
\sum_{\xi\in\Xi_t^{(m)}}
(Q_2(\xi)-\mu_Q)^2
}
$$

normalized score는 다음과 같다.

$$
\tilde Q_2(\xi)
=
\frac{
Q_2(\xi)-\mu_Q
}{
\sigma_Q+\epsilon
}
$$

accept / reject rule은 다음 중 하나로 둔다.

## Option A. Component-wise threshold

$$
\widehat{\Delta H}_A(D_t,\xi)>\tau_A
$$

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)<\tau_{amb}
$$

## Option B. Normalized score threshold

$$
\tilde Q_2(\xi)>\tau_{\tilde Q}
$$

초기 구현에서는 Option B를 사용한다.

---

# 13. Optional VLA Uncertainty Gate

Phase1으로 학습한 VLA를 model-side estimator로 사용할 수 있다.

후보 trajectory \(\xi\)의 action chunk에 대해 single-step denoising loss를 계산한다.

$$
U_{\pi_0}(\xi)
=
\mathrm{Agg}_{\tau}
[
\mathcal{L}_{denoise}
(S_\tau,A_{\tau:\tau+H-1};\pi_0)
]
$$

이 값은 \(H(A|S)\) 자체가 아니라, Phase1 VLA가 학습한 action manifold로부터 후보가 얼마나 벗어났는지를 나타내는 model-side compatibility / uncertainty score이다.

초기 구현에서는 main \(Q_2\) score에 직접 섞기보다 eligibility gate로 사용한다.

Phase1 training dataset의 loss distribution에서 quantile threshold를 만든다.

$$
\tau_{low}
=
Q_{q_{low}}(\mathcal{U}_{train})
$$

$$
\tau_{high}
=
Q_{q_{high}}(\mathcal{U}_{train})
$$

후보는 다음 범위 안에 있으면 model-side eligible로 본다.

$$
\tau_{low}
\le
U_{\pi_0}(\xi)
\le
\tau_{high}
$$

해석은 다음과 같다.

- 너무 낮은 loss: Phase1과 거의 중복된 redundant 후보
- 적당한 loss: Phase1 action manifold 안의 learnable novelty
- 너무 높은 loss: out-of-manifold 가능성이 큰 후보

---

# 14. Buffer Update

Accepted candidate는 먼저 raw dataset에 저장한다.

$$
D_{\mathrm{phase2},t+1}^{(m)}
=
D_{\mathrm{phase2},t}^{(m)}
\cup
\{\xi_{\mathrm{accepted}}\}
$$

동시에 같은 sample에 대한 vector DB entry를 생성한다.

$$
b_{\mathrm{new}}^{(m)}
=
(e_{\mathrm{new}},z_{\mathrm{new}}^a,ref_{\mathrm{new}},meta_{\mathrm{new}})
$$

reference buffer는 다음처럼 업데이트된다.

$$
B_{t+1}^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t+1}^{(m)}
$$

여기서 \(D_{\mathrm{phase2},t+1}^{(m)}\)는 raw dataset 자체가 아니라 Phase2 accepted entries의 vector DB index를 의미한다.

---

# 15. Phase Saturation Rules

## 15.1 Phase1 Stop: Phase2-readiness

Phase1은 Phase2에서 local action consistency를 평가할 수 있을 만큼 state support가 준비되면 종료한다.

후보 trajectory의 covered ratio는 다음과 같다.

$$
R_{\mathrm{cov}}(\xi)
=
\frac{
|\mathcal{T}_{covered}(\xi)|
}{
|\mathcal{T}(\xi)|
}
$$

skill \(m\)의 평균 covered ratio는 다음과 같다.

$$
\bar R_{\mathrm{cov}}^{(m)}
=
\frac{1}{|\Xi_{\mathrm{probe}}^{(m)}|}
\sum_{\xi\in\Xi_{\mathrm{probe}}^{(m)}}
R_{\mathrm{cov}}(\xi)
$$

전체 readiness score는 다음과 같다.

$$
R_{\mathrm{ready}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\bar R_{\mathrm{cov}}^{(m)}
$$

Phase1 종료 조건은 다음과 같다.

$$
\boxed{
R_{\mathrm{ready}}>\tau_{\mathrm{ready}}
}
$$

초기 기본값은 다음과 같다.

$$
\tau_{\mathrm{ready}}=0.7
$$

---

## 15.2 Phase2 Stop: MI-style Gain Saturation

최근 \(W\)개 accepted trajectory의 평균 normalized score를 계산한다.

$$
\bar{\tilde Q}_2^{(W)}
=
\frac{1}{W}
\sum_{i=t-W+1}^{t}
\tilde Q_2(\xi_i^*)
$$

Phase2 종료 조건은 다음과 같다.

$$
\boxed{
\bar{\tilde Q}_2^{(W)}<0
}
$$

즉, 최근 accepted 후보들이 candidate batch 평균보다 더 이상 좋지 않으면 saturation으로 판단한다.

---

## 15.3 Experiment-level Rule

실험에서는 baseline과 공정하게 비교하기 위해 전체 acquisition budget을 고정한다.

$$
B=100
$$

Phase1에는 minimum / maximum guard를 둔다.

$$
B_{1,\min}\le B_1\le B_{1,\max}
$$

예시는 다음과 같다.

$$
B_{1,\min}=20,\quad B_{1,\max}=50
$$

실험에서 Phase1 transition rule은 다음과 같다.

$$
\boxed{
(t\ge B_{1,\min}\land R_{\mathrm{ready}}>\tau_{\mathrm{ready}})
\lor
(t=B_{1,\max})
}
$$

Phase2 stopping rule은 다음과 같다.

$$
\boxed{
t=B
\lor
\bar{\tilde Q}_2^{(W)}<0
}
$$

공정 비교를 엄격히 할 경우 early stop을 끄고 \(t=B\)까지만 진행할 수 있다.

---

# 16. Radius Setting

skill \(m\)의 embedding set을 다음처럼 둔다.

$$
E^{(m)}
=
\{e_i\}_{i\in B_t^{(m)}
}
$$

각 point의 \(k\)-NN distance를 계산한다.

$$
r_i^{(m)}
=
d_k^s(e_i,E^{(m)})
$$

state-neighborhood radius는 skill별 nearest-neighbor distance distribution의 quantile로 정한다.

$$
\boxed{
\rho_m
=
\mathrm{Quantile}_{0.7}
\left(
\{r_i^{(m)}\}_{i=1}^{|E^{(m)}|}
\right)
}
$$

즉, skill별 local neighborhood scale을 buffer에서 자동으로 잡는다.

---

# 17. Final Dataset

최종 VLA 학습 데이터셋은 Phase1과 Phase2 raw dataset을 모두 포함한다.

$$
D_{\mathrm{final}}
=
D_{\mathrm{phase1}}
\cup
D_{\mathrm{phase2}}
$$

Phase1 데이터도 valid demonstration이며, state support를 확보하는 역할을 했기 때문에 최종 학습에 포함한다.

---

# 18. Final Summary

Method3는 기존 offline curation의 전제였던 다양한 state pool을 online acquisition 과정에서 직접 만든 뒤, 그 pool 안에서 MI-style score를 이용해 action diversity는 높고 action ambiguity는 낮은 데이터를 선별 수집한다.

핵심 흐름은 다음과 같다.

1. Phase1에서 skill-wise subgoal buffer를 이용해 subgoal-side state coverage를 확장한다.
2. Phase1에서는 path를 canonical하게 유지하여 불필요한 action multimodality를 억제한다.
3. Phase1 raw dataset으로 VLA encoder를 학습하고, 이를 freeze한다.
4. Phase1 raw dataset을 re-embedding하여 full skill-wise vector DB seed를 만든다.
5. Phase2에서는 full skill-wise vector DB를 이용해 action coverage gain과 conditional ambiguity increase를 계산한다.
6. Accepted Phase2 candidate는 raw dataset과 vector DB에 모두 누적한다.
7. 최종 학습 데이터는 Phase1과 Phase2 raw dataset을 모두 사용한다.

최종적으로 Method3의 선택 기준은 다음과 같다.

$$
\boxed{
\text{under-covered state/action region을 확장하면서도, 유사 state에서 action ambiguity를 증가시키지 않는 trajectory를 선택한다.}
}
$$
