# method3 version2(구현명세정리)

마감여부: 시작 전

## **method3 - pre-selective real-world data acquisition**(**사전선별적 수집)**

### **[sub-problem 3]**

Ours의 자동 데이터 취득 파이프라인은 지속적인 real-world data collection을 가능하게 한다. 그러나 명확한 기준 없이 모든 trajectory를 수집하는 것은 두 가지 측면에서 비효율적이다.

첫째, **비용 측면의 비효율**이다. 실세계 데이터 수집은 로봇 embodiment를 완전히 점유하고, 하드웨어 마모와 운영 비용을 수반한다. 또한 중복 trajectory까지 모두 저장하고 학습에 포함하면 storage cost와 GPU training cost가 불필요하게 증가한다.

둘째, **모델 학습 측면의 비효율**이다. 적절한 diversity는 distribution shift와 OOD robustness에 필요하지만, 단순히 rollout 수나 trajectory coverage를 늘리는 것이 항상 유용한 supervision을 의미하지는 않는다. 중복 trajectory는 정보 이득이 낮고, 유사한 observation/instruction에서 incompatible action mode가 섞인 trajectory는 action ambiguity를 증가시켜 flow-matching / diffusion 기반 VLA의 학습과 closed-loop execution을 불안정하게 만들 수 있다.

- **citation 및 주장 근거 정리**
    
    데이터셋의 적절한 다양성 확보는 정책의 distribution shift에 필수적이지만, 단순히 rollout을 많이 만들거나 trajectory의 커버리지를 제한없이 높이는 것이 모델학습 측면에서도 유용한 것은 아니다. 오히려 무분별한 다양성은 flow-matching / diffusion 기반 VLA 학습을 방해할 수 있다.
    
    **[1] Data Quality in Imitation Learning - Neurips2023**
    : 이 논문은 state diversity가 항상 유용하지 않다고 명시하고, action divergence와 transition diversity가 데이터 품질을 결정하는 중요한 요인이라고 분석. demo data는 더 consistent한 action을 갖도록 curated되어야 하며, expert/demo policy entropy를 줄이는 관점을 제안해. 특히 state에서 expert action entropy가 높으면 learned policy가 action을 잘 match하기 어렵다고 설명함.
    
    ---
    
    **[2] Robot Data Curation with Mutual Information Estimators - RSS 2025**
    : demonstration quality를 **state diversity**와 **action predictability**를 동시에 반영하는 mutual information으로 평가함 → **다양하지만 예측가능한 데이터가 좋은 데이터**
    
    ![image.png](method3%20version2(%EA%B5%AC%ED%98%84%EB%AA%85%EC%84%B8%EC%A0%95%EB%A6%AC)/image.png)
    
    > 
    > 
    > - $H_{\pi_{\theta}}(A):$ 기학습 VLA의 action entropy, 무질서도
    > - $H_{\pi_{\theta}}(A|S):$ 기학습 VLA의 주어진state일 때 action의 entropy, 무질서도
    - $H_{\pi_{\theta}}(A)$가 크다
    → 전체 action chunk 분포자체는 다양함.
    - 그런데 $H_{\pi_{\theta}}(A|S)$도 크다
    → 같은 observation/state에서 action이 불확실함.
    
    좋은 데이터는 단순히 action의 entropy가 높지만, 특정 state에 대한 action entropy는 작아서 일관되고 예측 가능한 데이터이다.
    
    ---
    
    **[3] Diff-DAgger - ICRA2025**
    
    : **Diff-DAgger**는 상용VLA의 action head인 diffusion/flow matching기반 head가 다중모드를 잘 다루지만, 여전히 OOD failure, compounding error, limited extrapolation 문제가 있다고 지적. 
    
    ![image.png](method3%20version2(%EA%B5%AC%ED%98%84%EB%AA%85%EC%84%B8%EC%A0%95%EB%A6%AC)/image%201.png)
    
     closed-loop execution에서 uncertainty가 커지는 상태는 action chunk 선택이 불안정해지고, 작은 오류가 다음 observation을 OOD로 밀어낼 수 있어 여전히 위험.
    
    → 같은 observation/state에서 action이 불확실함.
    
    ---
    
    **[4] VFP: Variational Flow-Matching Policy for Mulit-Modal Robot Manipuation - arxiv 2025.8**
    
    : flow matching도 excessive multimodality에 완전히 robust하지 않다. complex manipulation task에서 **averaged or ambiguous behaviors**로 collapse할 수 있다고 직접 지적함.
    
    → Uncontrolled diversity → $P_{data}(A|S)$ becomes overly multimodal → High $H_{\pi_{\theta}}(A|S)$ → ambiguous / unstable execution
    

따라서, Method3를 통해 **사전선별적인 데이터 수집**을 수행한다. 

기존 robot data curation 방법은 이미 수집된 demonstration pool에서 학습에 유용한 demonstration을 선별하는 **offline filtering problem**을 다룬다. 이 setting에서는 데이터셋의 state coverage가 이미 결정되어 있으며, curation method가 직접 제어하는 대상이 아니다. 

따라서 (DemInf)와 같은 방법은 주어진 pool 안에서 behavior cloning이 잘 학습할 수 있는 empirical expert distribution을 정의하는 데 집중한다. 이를 위해 mutual information(상호정보량)을 데이터셋 quality metric으로 사용한다.

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

이 metric은 $H(A)$를 통해 dataset 전체의 action diversity를 유지하고, 낮은 $H(A∣S)$를 통해 state가 주어졌을 때 action이 예측 가능하도록 만든다. 즉, **이미 충분히 다양한 state를 포함한 pool이 존재한다면**, $I(S;A)$기반 filtering은 그 state coverage 안에서 action diversity를 유지하면서도, 동일하거나 유사한 state에서는 일관된 action을 갖는 demonstration을 선별하는 데 적합하다. 

즉, 기존 curation setting에서는 state diversity를 새롭게 만들어내는 것이 아니라, 이미 존재하는 state pool 안에서 **learnable and predictable state-action relation**을 가진 데이터를 고르는 것이 핵심이다.

반면 우리의 setting은 고정된 demonstration pool에서 subset을 고르는 것이 아니라, real-world interaction을 통해 dataset 자체를 점진적으로 구축하는 **online acquisition problem**이다. 

즉, 기존 curation은 이미 모인 pool에서 어떤 데이터를 남길지 묻는 문제지만, 우리의 acquisition은 다음에 어떤 state-action region을 실제로 방문하고 추가할지를 결정하는 문제다. 이 경우 충분히 다양한 state pool이 이미 주어져 있다는 보장이 없기 때문에, 

$I(S;A)=H(A)−H(A∣S)$만으로는 부족하다.

따라서 기존 기준을 적용하려면, 먼저 충분한 state support를 만들어야 한다.

우리는 under-covered state를 적극적으로 확장하면서도, 각 state 안에서는 action이 일관되도록 데이터를 수집한다.

### **[주장]**

따라서 수집의 목적 함수는 현재 buffer가 충분히 커버하지 못한 state region을 확장하면서도, 유사한 state에서 action ambiguity를 과도하게 증가시키지 않는 데이터를 모으는 것이다.

따라서 우리의 acquisition objective는 이 직관을 확장하여, 데이터셋의 $H(S)$를 높이면서도 mutual information $I(S;A)=H(A)−H(A∣S)$을 크게 유지되는 방향으로 수집되어야 한다.

offline curation에서는 전체 dataset $D$가 이미 주어져 있기 때문에 dataset 전체의 MI를 직접구할 수 있다.

$I_{D}​(S;A)=H_{D}​(A)−H_{D}​(A∣S)$

그러나, 우리는 완성된 dataset의 MI를 재는 게 아니라, 후보 trajectory $\xi$를 현재 버퍼 $D_t$에 추가하면 dataset quality가 얼마나 좋아지는가?를 바탕으로 후보를 선정할 지 말 지를 고르는 문제이다.

즉, 다음을 의미.

$Gain_{MI}​(D_{t}​,ξ)=I(D_{t}​∪{ξ})−I(D_{t}​)$

여기서 $I(D)$는 dataset $D$에서의 state-action mutual information이다.

$I(D)=H_{D}​(A)−H_{D}​(A∣S)$

따라서, $Gain_{MI}(D_t,\xi)$ MI 정의를 대입하면:

$Gain_{MI}​(D_{t}​,ξ)​=[H_{D_{t}}​∪{ξ}​(A)−H_{D_{t}}​∪{ξ}​(A∣S)]−[H_{D_{t}}​​(A)−H_{D_{t}}​​(A∣S)]$

이를 항별로 정리하면:

$Gain_{MI}​(D_{t}​,ξ)​=[H_{D_{t}}​∪{ξ}​(A)−H_{D_{t}}​​(A)]−[H_{D_{t}}​∪{ξ}​(A∣S)−H_{D_{t}}​​(A∣S)]​$

여기서 각각을 다음처럼 정의한다.

$ΔH_{A}​(D_{t}​,ξ)=H_{D_{t}}​∪{ξ}​(A)−H_{D_{t}}​​(A)$

$ΔH_{A∣S}​(D_{t}​,ξ)=H_{D_{t}}​∪{ξ}​(A∣S)−H_{D_{t}}​​(A∣S)$

최종적으로, $Gain_{MI​}(D_{t}​,ξ)=ΔH_{A}​(D_{t}​,ξ)−ΔH_{A∣S}​(D_{t}​,ξ)$

그러나, 실제 continuous trajectory entropy $H$를 직접 추정하기 어렵다. 

연속적인 고차원 robot trajectory에서 $H(A)$, $H(A∣S)$를 직접 안정적으로 추정하는 것은 어렵다. 특히 action chunk가 $50\times6$이면 고차원 continuous distribution이라 density estimation이 불안정해짐.

그래서 실제 구현에서는 entropy 자체를 직접 구하지 않고, 다음처럼 근사함.

$ΔH(A)≈action descriptor novelty$

$ΔH(A∣S)≈local~state~neighborhood에서 action~support~expansion$

marginal proxy의 논리는:

1. 원래 목표는 후보 추가에 따른 MI 증가량이다.
2. 이것은 ΔH(A)−ΔH(A∣S)로 분해된다.
3. 실제 continuous trajectory의 entropy는 직접 추정하기 어렵다.
4. 따라서 kNN 거리 기반 action novelty와 local action support expansion으로 근사한다.

### [전체 과정의 데이터저장:adaptive skill-wise vector DB]

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

Raw dataset에는 re-embedding과 action descriptor 재계산에 필요한 원본 정보를 저장한다. 포멧은 lerobot dataset v3.0 포멧이다. 해당 포멧을 기반으로 약간의 확장된 label들을 함께 로깅한다.

Vector DB에는 retrieval key, action descriptor, metadata, raw dataset pointer를 저장한다.

---

### 3.1 Raw Dataset에 저장할 정보

[method3_vector_db_construction_fixed](https://www.notion.so/method3_vector_db_construction_fixed-3645fe31cfec803c8244c6a061544940?pvs=21)

## 목적

Method3에서는 Phase1과 Phase2에서 수집된 데이터를 **같은 skill-wise vector DB** 안에서 누적하여 사용한다.

다만 vector DB가 원본 데이터 저장소가 되는 것은 아니다. 전체 구조는 다음처럼 분리한다.

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

즉, raw dataset에는 observation, instruction, proprioception, action chunk 등 re-embedding과 재계산에 필요한 원본 정보를 저장하고, vector DB에는 retrieval key, action descriptor, metadata, raw dataset pointer를 저장한다.

핵심 원칙은 다음과 같다.

$$
\boxed{
\text{Phase1과 Phase2 데이터는 같은 vector DB에 누적하되, 동일한 frozen key extractor로 embedding해야 한다.}
}
$$

---

# 1. 전체 저장 구조

Method3에서는 두 종류의 저장소를 사용한다.

## 1.1 Raw Trajectory Dataset

Raw dataset은 원본 trajectory 정보를 저장하는 source of truth이다.

예시는 다음과 같다.

```
D_phase1_raw/
D_phase2_raw/
```

Raw dataset은 나중에 VLA 학습, re-embedding, action descriptor 재계산, ablation analysis에 사용된다.

---

## 1.2 Skill-wise Vector DB

Vector DB는 Phase2에서 후보 trajectory를 평가하기 위한 searchable index이다.

Vector DB는 skill-wise로 partition한다.

$$
B_t
=
\{B_t^{(1)},B_t^{(2)},\dots,B_t^{(M)}\}
$$

skill (m)에 대한 reference buffer는 다음과 같다.

$$
B_t^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t}^{(m)}
$$

여기서:

- (P_{}^{(m)}): Phase1에서 수집된 skill (m)의 seed / prototype buffer
- (D_{,t}^{(m)}): Phase2에서 현재까지 accepted된 skill (m)의 buffer

따라서 Phase2에서 후보를 평가할 때는 Phase1 데이터와 Phase2 데이터를 모두 reference로 사용한다.

$$
\boxed{
\text{Phase2 reference buffer} = \text{Phase1 seed} + \text{accepted Phase2 data}
}
$$

---

# 2. Phase1에서 저장해야 하는 정보

Phase1 데이터는 나중에 두 가지 용도로 사용된다.

1. Phase1 VLA 또는 adapter 학습
2. Phase1-trained VLA encoder를 이용한 re-embedding

따라서 Phase1에서는 단순히 action trajectory만 저장하면 안 된다. 이후 re-embedding과 action descriptor 재계산이 가능하도록 최소 원본 정보를 함께 저장해야 한다.

---

## 2.1 Phase1 Raw Dataset Entry

Phase1 raw dataset의 각 sample 또는 window는 다음 정보를 포함해야 한다.

```
phase: "phase1"
episode_id
skill_id
instruction
subgoal
time_index
observation
proprioception
action_chunk
raw_action_sequence
planner_type
success_flag
validity_flag
environment_metadata
object_metadata
```

수식적으로는 다음처럼 표현할 수 있다.

$$
D_{\mathrm{phase1}}
=
\{
(o_\tau,p_\tau,I,g,m,A_{\tau:\tau+H-1},meta_\tau)
\}_{\tau}
$$

여기서 re-embedding에 필요한 정보는 다음이다.

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

## 2.2 왜 raw 정보를 저장해야 하는가?

Phase1이 끝나면 Phase1 데이터로 VLA encoder를 학습하거나 adapter를 학습한다. 그 뒤 Phase1 raw dataset을 다시 불러와서 같은 encoder로 key를 다시 계산해야 한다.

Phase1-trained VLA embedding은 다음처럼 계산된다.

$$
e_\tau^{vla}
=
\phi_{\mathrm{VLA}}^{(1)}(o_\tau,I)
$$

proprioception은 VLA encoder에 넣지 않고, VLA embedding 뒤에 concat한다.

$$
e_\tau
=
[e_\tau^{vla};p_\tau]
$$

즉, Phase1 raw dataset에 (o_), (I), (p_)가 없으면 re-embedding을 할 수 없다.

따라서 Phase1에서 “raw form으로 저장한다”는 의미는 다음과 같다.

$$
\boxed{
\text{나중에 같은 encoder로 key를 다시 만들 수 있도록 observation, instruction, proprioception, action chunk, skill/subgoal metadata를 저장한다.}
}
$$

---

# 3. Vector DB Entry Format

Vector DB entry는 원본 데이터를 모두 들고 있을 필요는 없다. 대신 retrieval과 scoring에 필요한 compact feature와 raw dataset pointer를 저장한다.

각 entry는 다음과 같이 표현한다.

$$
b_i^{(m)}
=
(e_i,z_i^a,ref_i,meta_i)
$$

각 항의 의미는 다음과 같다.

- (e_i): state retrieval key
- (z_i^a): action chunk descriptor
- (ref_i): raw dataset pointer
- (meta_i): phase, skill id, subgoal, score 등 metadata

---

## 3.1 Recommended Vector DB Entry

```
vector_db_entry = {
  key: e_i,
  action_descriptor: z_i^a,
  dataset_ref: {
    dataset_name,
    episode_id,
    skill_id,
    start_t,
    end_t,
    horizon
  },
  meta: {
    phase,
    subgoal,
    planner_type,
    accepted_by,
    score,
    created_at
  }
}
```

---

## 3.2 반드시 저장할 항목

```
key
action_descriptor
dataset_ref
phase
skill_id
subgoal
accepted_by
```

---

## 3.3 선택적으로 저장할 항목

```
raw_action_chunk
Q2_score
Delta_H_A_score
Delta_H_A_given_S_score
VLA_uncertainty_score
neighbor_count
covered_ratio
```

raw action chunk는 vector DB에도 저장하면 빠르게 접근할 수 있다. 하지만 raw dataset pointer로 복원 가능하다면 vector DB에는 pointer만 저장해도 된다.

---

# 4. Dataset Pointer만 저장해도 되는 조건

Vector DB에 raw observation이나 raw action 전체를 저장하지 않고 pointer만 저장하려면, 해당 pointer를 통해 다음 정보를 반드시 복원할 수 있어야 한다.

```
observation
instruction
proprioception
action_chunk
skill_id
subgoal
phase
```

예시는 다음과 같다.

```
dataset_ref = {
  dataset: "D_phase1_raw",
  episode_id: "ep_0032",
  skill_id: "reach",
  start_t: 120,
  horizon: 50
}
```

이 pointer로부터 다음을 복원할 수 있어야 한다.

$$
o_\tau,\quad I,\quad p_\tau,\quad A_{\tau:\tau+H-1}
$$

그러면 언제든 Phase1-trained encoder로 key를 다시 만들 수 있다.

$$
e_\tau
=
[\phi_{\mathrm{VLA}}^{(1)}(o_\tau,I);p_\tau]
$$

---

# 5. State Retrieval Key

현재 VLA는 observation과 instruction만 입력으로 받아 embedding을 생성한다.

따라서 VLA embedding은 다음과 같이 계산한다.

$$
e_i^{vla}
=
\phi_{\mathrm{VLA}}(o_i,I_i)
$$

여기서:

- (o_i): image / observation
- (I_i): instruction

Proprioceptive state는 VLA encoder에 직접 넣지 않고, VLA embedding 뒤에 concat한다.

$$
e_i
=
[e_i^{vla};p_i]
$$

따라서 최종 retrieval key는 다음과 같다.

$$
\boxed{
e_i
=
[\phi_{\mathrm{VLA}}(o_i,I_i);p_i]
}
$$

여기서:

- (p_i): proprioceptive state
- (e_i): vector DB에서 neighbor search에 사용하는 최종 key

---

# 6. 왜 동일한 key extractor가 필요한가?

Phase2에서 covered state를 판단할 때는 candidate key와 buffer key 사이의 거리를 계산한다.

$$
\mathcal{N}_{\rho_m}^{s}(e_\tau)
=
\{i\in B_t^{(m)}:
d_s(e_\tau,e_i)<\rho_m
\}
$$

이때 (e_)와 (e_i)가 같은 embedding space에 있어야 거리 (d_s(e_,e_i))가 의미를 갖는다.

만약 Phase1은 한 embedding model로 key를 만들고, Phase2는 다른 embedding model로 key를 만들면, 두 embedding의 scale과 neighborhood structure가 달라질 수 있다.

따라서 다음 원칙을 따른다.

$$
\boxed{
\text{Phase1과 Phase2의 모든 key는 동일한 embedding model과 동일한 preprocessing pipeline으로 생성한다.}
}
$$

---

# 7. 구축 절차

## Step 1. Phase1 데이터 수집

Phase1에서는 state coverage seeding을 위해 canonical trajectory를 수집한다.

$$
\xi^*
=
\mathrm{InterpPlan}(S_t,g^*)
$$

이때 수집된 데이터는 action만 저장하는 것이 아니라, 이후 re-embedding과 action descriptor 재계산이 가능하도록 raw dataset에 저장한다.

$$
D_{\mathrm{phase1}}
=
\{
(o_\tau,p_\tau,I,g,m,A_{\tau:\tau+H-1},meta_\tau)
\}_{\tau}
$$

이 단계에서 vector DB key는 임시 key를 사용할 수 있다. 하지만 최종 Phase2 reference DB는 Phase1-trained VLA encoder로 다시 구축한다.

---

## Step 2. Phase1 VLA 학습

Phase1 데이터로 VLA 또는 adapter를 학습한다.

학습이 끝난 뒤, 이 encoder를 Phase2에서 사용할 retrieval encoder로 고정한다.

$$
\phi_{\mathrm{VLA}}^{(1)}
=
\text{Phase1-trained VLA encoder}
$$

---

## Step 3. Phase1 데이터 re-embedding

Phase1 종료 후에는 반드시 Phase1 raw dataset을 다시 읽어와서 Phase1-trained VLA encoder로 모든 Phase1 데이터를 re-embedding한다.

$$
e_i
=
[\phi_{\mathrm{VLA}}^{(1)}(o_i,I_i);p_i]
$$

action descriptor도 raw action chunk에서 다시 계산한다.

$$
z_i^a
=
\psi(A_{i:i+H-1})
$$

그 다음 skill-wise vector DB seed를 구축한다.

$$
P_{\mathrm{phase1}}^{(m)}
=
\{(e_i,z_i^a,ref_i,meta_i)\}_{i\in D_{\mathrm{phase1}}^{(m)}}
$$

이를 통해 Phase1 seed buffer가 Phase2 candidate와 같은 metric space에 놓이게 된다.

---

## Step 4. Phase2 동안 encoder freeze

Phase2 동안 retrieval encoder는 고정한다.

$$
\boxed{
\phi_{\mathrm{VLA}}^{(1)}
=
\text{fixed during Phase2}
}
$$

Phase2 중 새로 accepted된 데이터도 같은 encoder로 key를 생성한다.

$$
e_{\mathrm{new}}
=
[\phi_{\mathrm{VLA}}^{(1)}(o_{\mathrm{new}},I_{\mathrm{new}});p_{\mathrm{new}}]
$$

이렇게 해야 Phase1 data와 Phase2 data가 동일한 vector DB 안에서 일관되게 비교된다.

---

## Step 5. Phase2 accepted data 저장 및 누적

Phase2에서 accepted된 candidate는 먼저 Phase2 raw dataset에 저장한다.

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

skill-wise reference buffer는 다음처럼 업데이트된다.

$$
B_{t+1}^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t+1}^{(m)}
$$

여기서 (D_{,t+1}^{(m)})는 raw dataset 자체가 아니라 Phase2 accepted entries의 vector DB index를 의미한다.

---

# 8. Skill-wise Partition

Vector DB는 skill별로 partition한다.

$$
B_t
=
\{B_t^{(1)},B_t^{(2)},\dots,B_t^{(M)}\}
$$

후보 trajectory가 skill (m)에 해당하면, neighbor search는 해당 skill buffer 안에서만 수행한다.

$$
e_\tau
\rightarrow
B_t^{(m)}
$$

이렇게 하면 서로 다른 skill에서 발생하는 action mode가 같은 local neighborhood에 섞이는 것을 방지할 수 있다.

---

# 9. Phase Label의 필요성

Phase1과 Phase2 데이터를 같은 buffer에 누적하더라도, entry마다 phase label을 저장해야 한다.

이유는 다음과 같다.

## (1) 역할이 다르다

Phase1 데이터는 state support seed 역할을 한다.

$$
P_{\mathrm{phase1}}^{(m)}
$$

Phase2 데이터는 MI selector를 통과한 refinement data 역할을 한다.

$$
D_{\mathrm{phase2},t}^{(m)}
$$

## (2) Ablation이 가능하다

phase label을 저장하면 다음 비교가 가능하다.

- Phase1 only
- Phase2 only
- Phase1 + Phase2
- Phase1 seed buffer를 reference로 쓰는 효과
- Phase2 selector가 실제로 데이터를 정제했는지

## (3) 필요하면 weighting을 다르게 줄 수 있다

초기 구현에서는 동일 weight를 사용한다.

필요하면 이후 실험에서 Phase2 데이터에 더 높은 weight를 줄 수 있다.

$$
w_i =
\begin{cases}
1.0 & \text{if } i\in D_{\mathrm{phase2}} \\
\gamma & \text{if } i\in P_{\mathrm{phase1}}
\end{cases}
$$

초기 구현에서는 다음처럼 둔다.

$$
\gamma=1.0
$$

즉, Phase1과 Phase2 데이터를 동일 weight로 사용한다.

---

# 10. Neighbor Search

candidate window ()의 retrieval key를 다음처럼 계산한다.

$$
e_\tau
=
[\phi_{\mathrm{VLA}}^{(1)}(o_\tau,I);p_\tau]
$$

같은 skill buffer 안에서 state-neighbor를 검색한다.

$$
\mathcal{N}_{\rho_m}^{s}(e_\tau)
=
\{i\in B_t^{(m)}:
d_s(e_\tau,e_i)<\rho_m
\}
$$

covered state 판단 기준은 다음과 같다.

$$
\boxed{
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|\ge3
}
$$

데이터가 적은 초기 setting을 고려하여 최소 neighbor 수는 3으로 둔다.

---

# 11. Radius Setting

skill (m)의 embedding set을 다음처럼 둔다.

$$
E^{(m)}
=
\{e_i\}_{i\in B_t^{(m)}}
$$

각 point의 (k)-NN distance를 계산한다.

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

# 12. Final Dataset

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

# 13. 최종 구현 원칙

## 원칙 1. Raw dataset과 vector DB를 분리한다

$$
\boxed{
\text{Raw dataset은 원본 저장소, vector DB는 retrieval index}
}
$$

## 원칙 2. Phase1과 Phase2는 같은 key extractor를 사용한다

$$
e
=
[\phi_{\mathrm{VLA}}^{(1)}(o,I);p]
$$

## 원칙 3. Phase1 종료 후 Phase1 raw data를 re-embed한다

$$
D_{\mathrm{phase1}}
\rightarrow
P_{\mathrm{phase1}}
$$

## 원칙 4. Phase2 동안 retrieval encoder는 freeze한다

$$
\phi_{\mathrm{VLA}}^{(1)}
=
\text{fixed}
$$

## 원칙 5. Vector DB는 skill-wise로 partition한다

$$
B_t^{(m)}
$$

## 원칙 6. Phase label은 metadata로 저장한다

```
phase: "phase1" or "phase2"
```

## 원칙 7. Phase2 reference buffer는 Phase1과 accepted Phase2를 모두 사용한다

$$
B_t^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t}^{(m)}
$$

---

# Final Summary

Method3에서 vector DB는 Phase1과 Phase2 데이터를 연결하는 핵심 구조다.

Phase1에서 수집한 raw data는 Phase2의 state support seed가 되고, Phase2에서 accepted된 data는 같은 vector DB에 누적된다.

다만, 두 phase의 데이터가 같은 neighbor search에 사용되려면 반드시 동일한 key extractor를 사용해야 한다.

따라서 최종적으로는 다음 절차를 따른다.

1. Phase1 raw data를 수집한다.
2. Phase1 raw data로 VLA encoder를 학습한다.
3. Phase1 raw data를 Phase1-trained encoder로 re-embed한다.
4. Phase2 동안 encoder를 freeze한다.
5. Phase2 accepted raw data도 저장하고, 동일 encoder로 embedding하여 vector DB에 누적한다.

핵심은 다음 한 줄이다.

$$
\boxed{
\text{원본 데이터는 raw dataset에 저장하고, vector DB에는 동일한 frozen encoder로 만든 key와 raw dataset pointer를 저장한다.}
}
$$

Phase1과 Phase2 raw dataset은 다음 정보를 복원 가능하게 저장해야 한다.

```
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

### **[Overall framework: Two-phase online acquisiton]**

- **phase1: State Coverage Seeding
→** 먼저 H(S)를 키워 state support를 만든다.
- **phase2: Mutual Information-based Consistent Diversity Acquisition** 
→ 그 state support pool안에서 H(A)−H(A∣S) 기준으로 action distribution을 선별취득한다.

→기존 offline curation의 state 다양성 전제를 online acquisition 안에서 직접 만들어 주는 방식.
→(1) 다양한 state pool을 만들고, (2) 그 pool 안에서 MI을 고려한 분포를 만든다.

### **[Phase1: State Coverage Seeding]**

**목표**: (1) state pool확보, $maximize~H(S)$, 단조로운 state coverage를 최대 확보하며 phase2를 위한 warm-start과정

**방식**: 다양한 state region을 방문하되, 각 region에 대해 가능한 한 대표적이고 단조로운 trajectory를 seed로 저장하는 과정

**H(S)만 확보하는 이유**: 아직 local action distribution이 충분하지 않으므로, $H(A∣S)$를 안정적으로 추정할 수 없음.(cold-start문제로 state에 대한 action 샘플 $p(A|S)$이 없음.)
따라서, Phase1에서는 $H(A∣S)$ filter를 적용하지 않음. 

**단조로운 traj만 취득하는 이유**:skill-traj perturbation은 양면성이 있음. 여러 형상의 경로를 만든다는 점에서 $H(S)$를 키우는 것에 관여함. 그런데 동시에 같은 state 근처에서 여러 action mode를 만들 수 있기 때문에 $H(A∣S)$를 높이는데 관여함.

![image.png](method3%20version2(%EA%B5%AC%ED%98%84%EB%AA%85%EC%84%B8%EC%A0%95%EB%A6%AC)/image%202.png)

skill-traj의 다양성을 warm-start를 위해 state coverage 샘플을 모으는 과정에서 아직 local action distribution이 충분하지 않으므로, 이 ambiguity를 안정적으로 계산하거나 제어할 수 없다.
($H(A∣S)$계산 불가)

그러므로, Phase1에서 state diversity $H(S)$를 키울 때는 subgoal diversity만을 적용하고, skill-traj는 최대한 단조로운 경로만을 채택한다. (실제 경로는 단순 interpolation planning을 그럼 기존처럼 버퍼쪽이랑 같이 쓰는데, 그럼 기존처럼 버퍼쪽이랑 같이 쓰는데, 이용)

**State-Coverage-Guided Subgoal Sampling 목적함수:**
 $\boxed{
g^*
=\arg\max_{g'j\in\mathcal{G}{\mathrm{valid}}}
G_S^{\mathrm{goal}}(g'_j)
}$

,  $\boxed{
\xi^*
=\mathrm{InterpPlan}(S_t,g^*)
}$

여기서 $\mathcal{G}_{\mathrm{valid}}$는 workspace, reachability, collision, contact-safety 조건을 만족하는 subgoal 후보 집합.

### **State-Coverage-Guided Subgoal Sampling 구현 관련**

### 1. Phase1에서는 subgoal만 다양화하고 path는 canonical하게 유지

네가 말한 핵심을 반영하면, Phase1에서는 skill-traj 다양성을 강하게 reward하지 않는다.

즉, 후보는 이렇게 만든다.

$$
g'_j \in \mathcal{G}
$$

$$
\xi_j = \mathrm{InterpPlan}(S_t, g'_j)
$$

그리고 $g’_{j}$의 점수는 이 canonical trajectory가 얼마나 새로운 state region을 지나가는지로 평가한다.

$$
\widehat{\Delta H}(S; g'_j \mid D_t)
=
G_S(\mathrm{InterpPlan}(S_t, g'_j))
$$

따라서 subgoal 다양화는 **endpoint/state-region 다양화**를 만들고, canonical planner는 **action path를 일관되게 유지**하는 역할을 한다.

---

### 2. Subgoal 후보를 어떻게 점수화할까?

각 subgoal 후보 $g’_{j}$에 대해 canonical preview trajectory를 만든다.

$$
\xi_j
=
\mathrm{InterpPlan}(S_t,g'_j)
=
\{\hat S^j_1,\ldots,\hat S^j_T\}
$$

각 preview state를 embedding한다.

$$
\hat e^j_\tau
=
\phi_{state}(\hat S^j_\tau,I,g'_j,m)
$$

그 다음 기존 skill-wise subgoal buffer와 비교해서 state novelty를 계산한다.

$$
n_s(\hat e^j_\tau)
=
\left[
\log
\frac{
d_k^s(\hat e^j_\tau,B_t^{(m)})
}{
\bar d_{NN}^s(B_t^{(m)})+\epsilon
}
\right]_+
$$

이 값은 preview state 하나가 기존 buffer보다 얼마나 새로운지를 나타낸다.

그리고 subgoal 후보의 state gain은:

$$
G_S(g'_j)
=
\mathrm{Agg}_{\tau}
[
n_s(\hat e^j_\tau)
]
$$

로 둔다.

Phase1에서는 path 다양성을 많이 보상하고 싶지 않으니까, aggregation은 전체 path 평균보다 **subgoal 근처 state**를 중심으로 보는 게 더 좋다.

$$
G_S^{goal}(g'_j)
=
\mathrm{Agg}_{\tau\in\mathcal{T}_{end}}
[
n_s(\hat e^j_\tau)
]
$$

여기서 $\tau_{end}$는 trajectory 마지막 20% 구간 정도야.

즉, 질문은 이거야.

> 이 subgoal로 가면 최종적으로 도달하는 state region이 기존 buffer에 없는 새로운 region인가?
> 

---

### 3. 최종 Phase1 subgoal 선택식

가장 깔끔하게는 이렇게 쓰면 돼.

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

여기서 $\mathcal{G}_{valid}$는 workspace, reachability, collision, contact-safety 조건을 만족하는 subgoal 후보 집합이야.

$$
\mathcal{G}_{valid}
=
\{g'\in\mathcal{G}
\mid
\mathrm{reachable}(g'),\ \mathrm{safe}(g'),\ \mathrm{is\_transit}=\mathrm{True}
\}
$$

실행 비용 같은 디테일을 objective에 넣기 싫다면, 이렇게 hard constraint로 분리하면 된다.

Phase1에서는 state 다양성은 키우지만, 같은 state에서 여러 종류의 action이 생기는 것을 일부러 억제한다는 게 핵심.

### 4. Phase1 subgoal 누적 로깅 버퍼: skill-wise subgoal buffer

[skill_wise_subgoal_buffer_notion_safe](https://www.notion.so/skill_wise_subgoal_buffer_notion_safe-3645fe31cfec80e1ad83d56da884e370?pvs=21)

## 목적

Phase1에서는 path 다양성을 크게 보상하지 않고, **subgoal-side state coverage**를 넓히는 것이 목표다.

즉, 여러 subgoal 후보 중에서 다음 질문에 답한다.

> 이 subgoal로 가면 최종적으로 도달하는 state region이 기존 skill-wise subgoal buffer에 비해 얼마나 새로운가?
> 

이를 위해 Phase1에서는 **skill-wise subgoal buffer**를 사용한다.

$$
\boxed{
\text{skill-wise subgoal buffer는 Phase1에서 이미 seed한 terminal state region을 추적하기 위한 lightweight buffer이다.}
}
$$

---

# 1. Skill-wise Subgoal Buffer와 Full Skill-wise Vector DB의 차이

Skill-wise subgoal buffer와 full skill-wise vector DB는 모두 skill별로 나누지만, 목적이 다르다.

$$
\boxed{
B_{g,t}^{(m)}
\neq
B_t^{(m)}
}
$$

- (B_{g,t}^{(m)}): Phase1 subgoal-side state coverage 추적용 buffer
- (B_t^{(m)}): Phase2 state-action retrieval / MI-style scoring용 full vector DB

즉, skill-wise subgoal buffer는 Phase1에서 **어떤 subgoal / terminal state region을 이미 seed했는가**를 보는 용도이고, full skill-wise vector DB는 Phase2에서 **유사 state-action pair와 action consistency를 평가하는 용도**다.

---

# 2. 왜 subgoal buffer도 skill-wise인가?

같은 subgoal 위치라도 skill이 다르면 의미가 달라질 수 있다.

예를 들어 같은 object 근처 위치라도:

- reach skill에서는 end-effector 접근 목표
- grasp skill에서는 grasp pose 근처 목표
- wipe skill에서는 contact / wiping target
- place skill에서는 placement target

처럼 해석된다.

따라서 서로 다른 skill의 subgoal region을 하나의 buffer에서 비교하면 잘못된 novelty 판단이 생길 수 있다.

그래서 subgoal buffer는 skill별로 구성한다.

$$
B_{g,t}
=
\{B_{g,t}^{(1)},B_{g,t}^{(2)},\dots,B_{g,t}^{(M)}\}
$$

각 skill (m)에 대해:

$$
B_{g,t}^{(m)}
=
\{c_i^{(m)}\}_{i=1}^{N_m}
$$

---

# 3. Subgoal Buffer Entry Format

각 entry (c_i^{(m)})는 full state-action trajectory가 아니라, **subgoal 근처 terminal region**을 나타내는 정보를 저장한다.

추천 entry format은 다음과 같다.

```
subgoal_buffer_entry = {
  skill_id: m,
  subgoal: g_i,
  terminal_region_key: h_i,
  end_state_keys: {h_i^tau | tau in T_end},
  episode_id: episode_id,
  start_t: start_t,
  end_t: end_t,
  success_flag: true,
  planner_type: "InterpPlan",
  phase: "phase1"
}
```

여기서 가장 중요한 항목은 다음이다.

- (g_i): 실제 선택된 subgoal
- (h_i): subgoal 근처 terminal region descriptor
- ({h_i^}): 마지막 구간의 terminal state descriptors
- (episode_id, start_t, end_t): raw dataset pointer

---

# 4. Terminal Region Descriptor

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

초기 구현에서는 VLA embedding 대신 geometric descriptor를 사용해도 된다.

예시 descriptor는 다음과 같다.

$$
\hat h_\tau^j
=
[q_\tau^j,\ x_{\tau}^{EE,j},\ x_{\tau}^{EE,j}-g'_j,\ m]
$$

여기서:

- (q_^j): preview joint state
- (x_{}^{EE,j}): preview end-effector pose
- (g’_j): candidate subgoal
- (m): skill id

subgoal buffer에는 마지막 구간 descriptor들의 평균을 저장할 수 있다.

$$
h_i
=
\frac{1}{|\mathcal{T}_{end}|}
\sum_{\tau\in\mathcal{T}_{end}}
h_i^\tau
$$

또는 마지막 구간 descriptor set 자체를 저장할 수 있다.

$$
\{h_i^\tau\}_{\tau\in\mathcal{T}_{end}}
$$

초기 구현에서는 평균 descriptor (h_i)를 기본으로 저장하고, 필요하면 end-state descriptor set을 함께 저장한다.

---

# 5. Subgoal 후보 생성

현재 nominal subgoal을 (g)라고 할 때, 주변에 여러 후보를 만든다.

$$
\mathcal{G}
=
\{g'_1,g'_2,\dots,g'_K\}
$$

후보 생성 방법 예시는 다음과 같다.

- Gaussian offset
- shell sampling
- object-relative directional sampling

그 다음 reachable / safe하지 않은 후보는 제거한다.

$$
\mathcal{G}_{valid}
=
\{
g'_j\in\mathcal{G}
\mid
\mathrm{reachable}(g'_j),\ \mathrm{safe}(g'_j)
\}
$$

필요하면 transit motion 조건도 hard constraint로 추가한다.

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

# 6. Canonical Preview Trajectory 생성

각 후보 (g’_j)에 대해 현재 state (S_t)에서 해당 subgoal까지의 canonical interpolation trajectory를 만든다.

$$
\xi_j
=
\mathrm{InterpPlan}(S_t,g'_j)
=
\{\hat S_1^j,\hat S_2^j,\dots,\hat S_T^j\}
$$

여기서 (S_^j)는 실제 실행 후 observation이 아니라, planning 시점에서 얻는 preview state이다.

Phase1에서는 path 다양성을 크게 보상하지 않기 때문에, 모든 후보에 대해 동일한 canonical planner를 사용한다.

$$
\boxed{
\text{subgoal은 다양화하지만, path는 canonical하게 유지한다.}
}
$$

---

# 7. Subgoal Buffer와 비교하기

후보 (g’_j)의 마지막 구간 preview state descriptor를 만든다.

$$
\hat h_\tau^j
=
\phi_{goal}(\hat S_\tau^j,g'_j,m)
$$

이 descriptor를 같은 skill의 subgoal buffer와 비교한다.

$$
B_{g,t}^{(m)}
=
\{h_1,h_2,\dots,h_N\}
$$

각 preview descriptor에 대해 (k)-NN 평균 거리를 계산한다.

$$
d_k^g(\hat h_\tau^j,B_{g,t}^{(m)})
=
\frac{1}{k}
\sum_{i\in\mathcal{N}_k^g(\hat h_\tau^j)}
d_g(\hat h_\tau^j,h_i)
$$

여기서:

- (d_g): goal-side descriptor distance
- (_k^g): subgoal buffer 안에서의 (k)-nearest neighbors

---

# 8. Buffer 내부 기준 거리 계산

거리 scale을 정규화하기 위해 subgoal buffer 내부의 평균 nearest-neighbor 거리를 사용한다.

$$
\bar d_{NN}^{g}(B_{g,t}^{(m)})
=
\frac{1}{|B_{g,t}^{(m)}|}
\sum_{h_i\in B_{g,t}^{(m)}}
\min_{l\ne i}
d_g(h_i,h_l)
$$

이 값은 현재 skill (m)의 subgoal buffer 안에서 terminal region들이 보통 어느 정도 떨어져 있는지를 나타낸다.

초기 buffer가 너무 작아서 nearest-neighbor scale을 안정적으로 계산하기 어렵다면, minimum scale을 둔다.

$$
s_g^{(m)}
=
\max
\left(
\bar d_{NN}^{g}(B_{g,t}^{(m)}),
s_{\min}^{g}
\right)
$$

---

# 9. Point-level Subgoal-side Novelty

각 terminal preview descriptor의 novelty를 다음처럼 계산한다.

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

해석은 다음과 같다.

- (n_g ): 기존 subgoal buffer에 이미 가까운 terminal region이 있음
- (n_g > 0): 기존 subgoal buffer 바깥쪽의 novel / under-covered terminal region

---

# 10. Subgoal-level State Gain

subgoal 후보 (g’_j)의 score는 마지막 구간 preview states의 novelty를 aggregation해서 계산한다.

$$
G_S^{goal}(g'_j)
=
\mathrm{Agg}_{\tau\in\mathcal{T}_{end}}
[
n_g(\hat h_\tau^j)
]
$$

가장 단순한 구현은 평균이다.

$$
G_S^{goal}(g'_j)
=
\frac{1}{|\mathcal{T}_{end}|}
\sum_{\tau\in\mathcal{T}_{end}}
n_g(\hat h_\tau^j)
$$

이 score가 높다는 것은 해당 subgoal이 기존 skill-wise subgoal buffer에 비해 새로운 terminal state region으로 이어진다는 뜻이다.

---

# 11. 최종 Subgoal 선택

가장 높은 goal-side state gain을 갖는 subgoal을 선택한다.

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

---

# 12. Subgoal Buffer Update

선택된 subgoal (g^*)에 대해 canonical trajectory를 실제 실행한 뒤, 성공한 경우에만 subgoal buffer를 업데이트한다.

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

여기서 (c_*^{(m)})는 다음 entry다.

```
c_*^(m) = {
  skill_id: m,
  subgoal: g_star,
  terminal_region_key: h_star,
  end_state_keys: {h_tau_star | tau in T_end_exec},
  episode_id: episode_id,
  start_t: start_t,
  end_t: end_t,
  success_flag: true,
  planner_type: "InterpPlan",
  phase: "phase1"
}
```

실패한 trajectory는 subgoal coverage seed로 쓰지 않는다.

---

# 13. Cold-start 처리

Phase1 초반에는 (B_{g,t}^{(m)})가 비어 있거나 매우 작을 수 있다.

이 경우 다음 중 하나를 사용한다.

## Option A. Random valid subgoal warm-start

각 skill마다 최소 (N_{seed})개까지는 valid subgoal 중에서 random 또는 stratified sampling으로 선택한다.

$$
|B_{g,t}^{(m)}|<N_{seed}
\Rightarrow
g^*\sim \mathcal{G}_{valid}
$$

## Option B. Farthest-first initialization

첫 번째 subgoal을 random하게 선택한 뒤, 이후에는 기존 seed와 가장 먼 후보를 선택한다.

$$
g^*
=
\arg\max_{g'_j\in\mathcal{G}_{valid}}
d_g(\hat h^j,B_{g,t}^{(m)})
$$

초기 구현에서는 Option A를 기본으로 두고, 필요하면 Option B를 ablation한다.

---

# 14. Pseudo-code

```python
def select_phase1_subgoal(
    current_state,
    nominal_goal,
    instruction,
    skill_id,
    subgoal_buffer,
    K=32,
    k_nn=5,
    eps=1e-6,
    seed_min=3,
):
    candidates = sample_subgoal_candidates(nominal_goal, K)
    valid_candidates = [
        g for g in candidates
        if is_reachable(g) and is_safe(g)
    ]

    # Cold-start: not enough subgoal seeds
    if len(subgoal_buffer[skill_id]) < seed_min:
        selected_goal = random_choice(valid_candidates)
        selected_traj = interp_plan(current_state, selected_goal)
        return selected_goal, selected_traj

    buffer_keys = subgoal_buffer[skill_id].terminal_region_keys
    buffer_scale = mean_nearest_neighbor_distance(buffer_keys)
    buffer_scale = max(buffer_scale, MIN_GOAL_SCALE)

    best_goal = None
    best_score = -float("inf")

    for g_candidate in valid_candidates:
        preview_traj = interp_plan(current_state, g_candidate)

        # 마지막 20% 구간만 평가
        start_idx = int(0.8 * len(preview_traj.states))
        end_states = preview_traj.states[start_idx:]

        novelty_values = []

        for s_tau in end_states:
            h_tau = goal_descriptor(
                state=s_tau,
                subgoal=g_candidate,
                skill_id=skill_id,
            )

            d_knn = mean_knn_distance(
                query=h_tau,
                keys=buffer_keys,
                k=k_nn,
            )

            novelty = max(
                math.log(d_knn / (buffer_scale + eps)),
                0.0,
            )

            novelty_values.append(novelty)

        score = mean(novelty_values)

        if score > best_score:
            best_score = score
            best_goal = g_candidate

    selected_traj = interp_plan(current_state, best_goal)
    return best_goal, selected_traj
```

---

# 15. 핵심 요약

Skill-wise subgoal buffer는 Phase1에서 각 skill이 이미 seed한 terminal state region을 추적하기 위한 buffer다.

이 buffer는 full skill-wise vector DB와 다르다.

$$
\boxed{
B_{g,t}^{(m)}
=
\text{Phase1 subgoal-side coverage tracking용}
}
$$

$$
\boxed{
B_t^{(m)}
=
\text{Phase2 state-action retrieval / scoring용}
}
$$

Phase1에서는 여러 subgoal 후보를 만들고, 각 후보까지 canonical preview를 생성한 뒤, 마지막 20% 구간의 terminal state descriptor가 기존 skill-wise subgoal buffer와 가장 덜 겹치는 subgoal을 선택한다.

### **[Phase2: Mutual Information-based Consistent Diversity Acquisition]**

Phase1을 통해 어느 정도 state support가 형성되면, 이제 기존 DemInf류 curation의 전제와 비슷해짐.

즉, 이미 다양한 state pool이 어느 정도 존재함.

따라서, $I(S;A)=ΔH(A)−ΔH(A∣S)$를 적용할 수 있음. 이것을 샘플 평가 기준으로 데이터를 취득함.

**목표:** Phase2는 Phase1에서 형성된 state support 안에서, 생성된 후보 trajectory 중 학습에 유용한 trajectory만 선별적으로 수집하는 단계이다.

Phase1에서는 state coverage를 먼저 확보하기 위해 subgoal diversity를 활용하고, trajectory는 canonical하게 유지했다. 따라서 Phase2에서는 이미 어느 정도 다양한 state pool이 존재한다고 보고, 그 안에서 action distribution을 정제한다.

Phase2의 핵심 목표는 다음과 같다.

$$
\text{action coverage는 확장하되, 유사 state에서 action ambiguity는 증가시키지 않는다.}
$$

이를 mutual information 관점에서 보면, 기존 offline curation의 목적은 다음과 같다.

$$
I(S;A)=H(A)-H(A|S)
$$

여기서:

- (H(A)): 전체 action diversity
- (H(A|S)): state가 주어졌을 때 action ambiguity

즉, 좋은 데이터셋은 전체 action은 다양하지만, 같은 state에서는 action이 일관적인 데이터셋이다.

---

## 1. Online Candidate Selection에서의 MI Gain

우리 setting은 offline dataset filtering이 아니라 online acquisition이다. 따라서 전체 dataset의 mutual information을 직접 평가하는 것이 아니라, 후보 trajectory를 현재 buffer에 추가했을 때 dataset quality가 얼마나 좋아지는지를 평가한다.

즉, 우리는 개별 trajectory 자체의 mutual information을 계산하는 것이 아니다. Mutual information은 dataset-level quantity이므로, Phase2에서는 후보 $\xi$가 현재 buffer $D_{t}​$에 추가될 때 dataset-level mutual information을 얼마나 증가시키는지를 marginal gain으로 평가한다.

현재 buffer를 $D_{t}​$, 후보 trajectory를 $\xi$ 라고 하자.

후보 $\xi$의 marginal mutual-information gain을 다음과 같이 정의한다.

$$
Gain_{MI}(D_t,\xi)
=
I(D_t\cup\{\xi\})-I(D_t)
$$

여기서 $I(D)$는 dataset $D$에서의 state-action mutual information이다.

$$
I(D)=H_D(A)-H_D(A|S)
$$

따라서 MI 정의를 대입하면 다음과 같다.

$$
\begin{aligned}
Gain_{MI}(D_t,\xi)
&=
\left[
H_{D_t\cup\{\xi\}}(A)
-
H_{D_t\cup\{\xi\}}(A|S)
\right]
-
\left[
H_{D_t}(A)
-
H_{D_t}(A|S)
\right]
\end{aligned}
$$

이를 항별로 정리하면 다음과 같다.

$$
\begin{aligned}
Gain_{MI}(D_t,\xi)
&=
\left[
H_{D_t\cup\{\xi\}}(A)-H_{D_t}(A)
\right]
-
\left[
H_{D_t\cup\{\xi\}}(A|S)-H_{D_t}(A|S)
\right]
\end{aligned}
$$

각 항을 다음처럼 정의한다.

$$
\Delta H_A(D_t,\xi)
=
H_{D_t\cup\{\xi\}}(A)-H_{D_t}(A)
$$

$$
\Delta H_{A|S}(D_t,\xi)
=
H_{D_t\cup\{\xi\}}(A|S)-H_{D_t}(A|S)
$$

그러면 최종적으로 다음과 같이 쓸 수 있다.

$$
Gain_{MI}(D_t,\xi)
=
\Delta H_A(D_t,\xi)
-
\Delta H_{A|S}(D_t,\xi)
$$

---

## 2. Why Proxy is Needed

실제 robot trajectory는 continuous하고 고차원이다. 특히 action chunk는 다음과 같은 형태를 갖는다.

$$
A\in\mathbb{R}^{50\times 6}
$$

따라서 (H(A)), (H(A|S))를 직접 density estimation으로 안정적으로 추정하기 어렵다.

그래서 Phase2에서는 entropy 자체를 직접 계산하지 않고, 다음과 같이 proxy로 근사한다.

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

즉, Phase2의 실제 구현은 다음 두 항을 계산하는 것이다.

1. 후보 action이 현재 buffer의 action coverage를 얼마나 확장하는가?
2. 후보 action이 유사 state에서 기존 action support를 얼마나 새롭게 넓히는가?
(빼는 방식으로 패널처럼 적용되어, 얼마나 일관되는 가로 적용됨)

---

## 3. Phase2 Reference Buffer as Skill-wise Vector DB

Phase2에서는 skill별 adaptive vector DB를 사용한다.

skill (m)에 대한 현재 reference buffer는 다음과 같다.

$$
B_t^{(m)}
=
P_{phase1}^{(m)}
\cup
D_{phase2,t}^{(m)}
$$

여기서:

- $P_{phase1}^{(m)}$: Phase1에서 형성된 skill-wise state-action prototype
- $D_{phase2,t}^{(m)}$: Phase2에서 현재까지 accepted된 skill segment buffer

각 vector DB entry는 다음과 같이 저장한다.

$$
b_i^{(m)}
=
(e_i, z_i^a, A_i, meta_i)
$$

각 항의 의미는 다음과 같다.

- $e_i$: VLA state / context embedding
- $z_i^a$: action chunk descriptor
- $A_i$: raw action chunk
- $meta_i$: skill id, subgoal, phase 등 metadata

Phase2에서는 VLA embedding을 사용하여 유사 state를 검색한다.

---

## 4. Candidate Representation

후보 skill trajectory segment를 다음과 같이 둔다.

$$
\xi
=
\{(S_\tau, A_{\tau:\tau+H-1})\}_{\tau=1}^{T}
$$

각 window ()에 대해 state embedding과 action descriptor를 만든다.

---

## 4.1 VLA State Embedding as Retrival Key

Phase2에서는 observation과 instruction만을 입력으로 받아 VLA embedding을 생성한다.

$$
e_\tau
=
\phi_{VLA}(o_\tau, I)
$$

여기서:

- ( $o_\tau$):  image / observation
- (I): instruction

Proprioceptive state ( $p_\tau$)는 VLA encoder에 직접 넣지 않고, VLA embedding 뒤에 concat한다.

$$

e_\tau =
[e_\tau^{vla};p_\tau]

$$

따라서 최종 retrieval key는 다음과 같다.

$$
\boxed{
e_\tau
=
[\phi_{\mathrm{VLA}}(o_\tau,I);p_\tau]
}
$$

skill id (m)은 embedding input으로 넣지 않고, skill-wise vector DB partition으로 처리한다.

$$
B_t^{(m)}
$$

즉, candidate window의 유사 state 검색은 같은 skill-wise vector DB 안에서 수행한다.

이 ( $e_\tau$)는 vector DB에서 유사 state를 검색하는 key로 사용된다.

---

## 4.2 DCT Action Descriptor

각 action chunk는 다음과 같다.

$$
A_{\tau:\tau+H-1}\in\mathbb{R}^{50\times 6}
$$

시간축 DCT를 적용한다.

$$
C_\tau
=
DCT_{time}(A_{\tau:\tau+H-1})
$$

앞쪽 (K)개의 저주파 성분만 남긴다.

$$
\tilde C_\tau
=
C_\tau[1:K,:]
$$

이를 flatten하여 action descriptor로 사용한다.

$$
z_\tau^a
=
\psi(A_{\tau:\tau+H-1})
=
vec(\tilde C_\tau)
$$

따라서 (K=3)이면 action descriptor의 차원은 다음과 같다.

$$
z_\tau^a\in\mathbb{R}^{18}
$$

(K)는 action descriptor의 temporal resolution을 결정하는 hyperparameter이다. 너무 작은 (K)는 trajectory의 high-frequency correction이나 contact timing 차이를 잃을 수 있고, 너무 큰 (K)는 noise와 jitter를 action novelty로 과대평가할 수 있다. 따라서 실험에서는 (K{3,5,8})에 대해 ablation하거나, DCT energy preservation ratio를 기준으로 (K)를 선택할 수 있다.

$K
=
\min
\left\{
K':
\frac{
\sum_{k=1}^{K'}\|C_k\|^2
}{
\sum_{k=1}^{50}\|C_k\|^2
}
\ge \eta
\right\}$

---

## 5. Estimating Action Coverage Gain

$H_A(D_t,\xi)$는 후보가 현재 buffer의 action coverage를 얼마나 확장하는지 나타낸다.

단, 여기서 action novelty는 임의의 motion novelty를 의미하지 않는다. 모든 후보 trajectory는 task-conditioned planner 또는 CaP module이 생성한 feasible후보 집합 안에서 평가된다. 따라서 동일 skill 내에서 task execution에 사용 가능한 action coverage의 확장을 의미한다.

Phase2 selector는 다음 후보 집합 안에서만 적용된다

skill $(m)$의 action descriptor buffer를 다음처럼 둔다.

$$
B_A^{(m)}
=
\{z_i^a \mid i\in B_t^{(m)}\}
$$

각 candidate action descriptor $z_\tau^a$에 대해 action-space kNN 거리를 계산한다.

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

candidate action novelty를 다음과 같이 정의한다.

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

해석은 다음과 같다.

- $n_a(z_\tau^a)$: 기존 buffer에 이미 유사한 action pattern이 있음
- $n_a(z_\tau^a)>0$: 후보 action이 기존 action support 바깥에 있음

trajectory-level action coverage gain은 다음처럼 계산한다.

$$
\widehat{\Delta H}_A(D_t,\xi)
=
Agg_{\tau}
\left[
n_a(z_\tau^a)
\right]
$$

추천 aggregation은 top-quantile mean이다.

$$
\widehat{\Delta H}_A(D_t,\xi)
=
\frac{1}{|\mathcal{T}_q|}
\sum_{\tau\in\mathcal{T}_q}
n_a(z_\tau^a)
$$

여기서 $\mathcal{T}_{q}$는 action novelty가 높은 상위 (q%) window index이다.

---

## 6. Estimating Conditional Action Ambiguity Increase

$H_{A|S}(D_t,\xi)$는 후보가 유사 state에서 action ambiguity를 얼마나 증가시키는지 나타낸다.

핵심 아이디어는 다음과 같다.

> VLA embedding으로 유사 state를 찾고, 그 state-neighbor들의 action support 안에 후보 action이 들어가는지 확인한다.
> 

---

## 6.1 Covered State Condition

각 candidate state embedding $e_\tau$에 대해 같은 skill-wise vector DB에서 state-neighbor를 검색한다.

$$
\mathcal{N}_{\rho_m}^{s}(e_\tau)
=
\{i\in B_t^{(m)}
:
d_s(e_\tau,e_i)<\rho_m
\}
$$

여기서 $p_m$은 skill (m)에 대한 state-neighborhood radius이다.

$H(A|S)$ proxy는 충분한 유사 state sample이 있을 때만 계산한다.

$$
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|
\ge
k_{\min}^{(m)}
$$

covered window set은 다음과 같이 정의한다.

$$
\mathcal{T}_{covered}(\xi)
=
\left\{
\tau:
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|
\ge
k_{\min}^{(m)}
\right\}
$$

만약 (_{covered}())가 충분하지 않다면, 해당 후보는 Phase2 MI selector로 신뢰성 있게 평가하지 않는다.

---

## 6.2 Local Action Support

covered state (e_)에 대해, state-neighbor들의 action descriptor를 모은다.

$$
\mathcal{A}_{e_\tau}
=
\{z_i^a
:
i\in\mathcal{N}_{\rho_m}^{s}(e_\tau)
\}
$$

이 set이 candidate state 주변의 local action support이다.

---

## 6.3 Nearest-Support Expansion

후보 action $z_\tau^a$가 local action support 바깥에 있는지 계산한다.

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

local action ambiguity increase는 다음과 같이 정의한다.

$$
\Delta h_{A|S}(e_\tau,z_\tau^a)
=
\left[
\log
\frac{
d_{min}^{a}(z_\tau^a,\mathcal{A}_{e_\tau})
}{
\bar d_{NN}^{a}(\mathcal{A}_{e_\tau})+\epsilon
}
\right]_+
$$

해석은 다음과 같다.

- $h_{A|S}$: 후보 action이 기존 local action support 안에 있음
- $h_{A|S}>0$: 후보 action이 유사 state에서 기존 action support를 새롭게 확장함

즉, 이 값이 크면 후보가 $H(A|S)$를 증가시키는 ambiguous novelty일 가능성이 높다.

---

## 6.4 Trajectory-level Conditional Ambiguity

candidate trajectory 전체의 conditional ambiguity increase는 covered windows에서만 aggregation한다.

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
Agg_{\tau\in\mathcal{T}_{covered}(\xi)}
\left[
\Delta h_{A|S}(e_\tau,z_\tau^a)
\right]
$$

추천 aggregation은 mean 또는 max이다.

Mean aggregation:

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
\frac{1}{|\mathcal{T}_{covered}(\xi)|}
\sum_{\tau\in\mathcal{T}_{covered}(\xi)}
\Delta h_{A|S}(e_\tau,z_\tau^a)
$$

Max aggregation:

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
\max_{\tau\in\mathcal{T}_{covered}(\xi)}
\Delta h_{A|S}(e_\tau,z_\tau^a)
$$

초기 구현에서는 mean을 사용하고, 보수적인 rejection이 필요하면 max를 ablation한다.

---

## 7. Multi-modal Action Support

Nearest-support 방식은 multi-modal action을 자연스럽게 허용한다.

유사 state 안에 여러 valid action mode가 있더라도, 후보 action이 그중 하나와 가까우면 다음 값이 작다.

$$
d_{min}^{a}(z_\tau^a,\mathcal{A}_{e_\tau})
$$

따라서 후보가 기존 local mode 중 하나와 일치하면 penalty가 작다.

반대로 기존 어느 mode와도 멀면 penalty가 커진다.

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

하지만 초기 구현에서는 clustering 없는 nearest-sample 방식을 사용한다.

---

## 8. Final Phase2 Score

최종 Phase2 score는 다음과 같다.

$$
Q_2(\xi)
=
\beta \widehat{\Delta H}_A(D_t,\xi)
-
\lambda \widehat{\Delta H}_{A|S}(D_t,\xi)
$$

따라서 선택은 다음과 같다.

$$
\xi^*
=
\arg\max_{\xi\in\Xi_t^{(m)}}
Q_2(\xi)
$$

여기서 (_t^{(m)})는 현재 skill (m)에서 생성된 후보 trajectory set이다.

---

## 9. Accept / Reject Rule

후보는 다음 조건을 만족할 때 accept한다.

$$
\widehat{\Delta H}_A(D_t,\xi) > \tau_A
$$

$$
\widehat{\Delta H}_{A|S}(D_t,\xi) < \tau_{amb}
$$

또는 최종 score threshold를 사용할 수 있다.

$$
Q_2(\xi)>\tau_Q
$$

해석은 다음과 같다.

- action coverage gain이 충분히 크다.
- 유사 state에서 action ambiguity increase는 작다.

즉, useful diversity에 해당한다.

---

## 10. Buffer Update

accepted candidate는 skill-wise vector DB에 추가한다.

$$
D_{phase2,t+1}^{(m)}
=
D_{phase2,t}^{(m)}
\cup
\{(e_\xi,z_\xi^a,A_\xi,meta_\xi)\}
$$

reference buffer는 다음처럼 업데이트된다.

$$
B_{t+1}^{(m)}
=
P_{phase1}^{(m)}
\cup
D_{phase2,t+1}^{(m)}
$$

이 업데이트는 VLA를 재학습하지 않지만, acquisition state를 adaptive vector DB에 반영한다.

---

## 11. Summary

Phase2에서 mutual-information gain은 직접 entropy를 추정하지 않고, 다음처럼 proxy로 계산한다.

$$
Gain_{MI}(D_t,\xi)
\approx
\widehat{\Delta H}_A(D_t,\xi)
-
\widehat{\Delta H}_{A|S}(D_t,\xi)
$$

여기서:

$$
\widehat{\Delta H}_A(D_t,\xi)
\approx
\text{normalized action-space novelty}
$$

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
\approx
\text{local action support expansion in VLA-neighbor states}
$$

한 줄로 정리하면:

> Phase2는 VLA embedding으로 유사 state를 찾고, DCT action descriptor로 action novelty와 local action support expansion을 계산하여, action diversity는 늘리되 유사 state에서 action ambiguity를 증가시키는 후보는 제거한다.
> 

### Phase-saturation rules

다음의 방식으로 각 phase의 saturation을 판단하여, 다음 phase혹은 종료를 판단함.

Acquisition은 두 단계로 구성된다.

- **Phase1: State Coverage Seeding**
- **Phase2: MI-based Consistent Diversity Acquisition**

논문 Method에서는 전체 episode 수를 고정하지 않고, 각 phase가 자신의 목적을 달성했는지를 기준으로 종료한다.

실험에서는 baseline과 공정하게 비교하기 위해 전체 acquisition budget을 고정하되, Phase1과 Phase2의 경계는 adaptive하게 결정한다.

---

# [1] Method: Saturation-driven Acquisition

## [1-1] Phase1 Saturation: Phase2-readiness

Phase1의 목적은 단순히 state novelty를 많이 만드는 것이 아니라, Phase2에서 local action consistency를 평가할 수 있을 만큼 state support를 만드는 것이다.

따라서 Phase1은 다음 질문을 기준으로 종료한다.

> 현재 buffer가 Phase2 후보들의 (H(A|S)) proxy를 계산할 수 있을 만큼 충분히 covered 되었는가?
> 

---

## Probe Candidate Set

Phase1 중에도 실제 실행 없이 Phase2 후보 generator로 probe trajectory set을 만든다.

$$
\Xi_{\mathrm{probe}}^{(m)}
=
\{\xi_1,\dots,\xi_K\}
$$

여기서 (m)은 skill id이고, (_{}^{(m)})는 skill (m)에 대해 생성된 preview trajectory 후보 집합이다.

각 후보 trajectory는 window 단위로 표현한다.

$$
\xi
=
\{(S_\tau,A_{\tau:\tau+H-1})\}_{\tau=1}^{T}
$$

---

## State Key and VLA Embedding

Phase2에서 사용하는 state key는 action prediction에 필요한 policy input을 포함해야 한다.

현재 VLA가 subgoal (g)를 직접 input으로 받지 못한다고 가정한다. 따라서 VLA embedding은 observation, proprioception, instruction만으로 계산한다.

$$
S_\tau^{key}
=
(o_\tau,p_\tau,I)
$$

$$
e_\tau
=
\phi_{\mathrm{VLA}}(S_\tau^{key})
=
\phi_{\mathrm{VLA}}(o_\tau,p_\tau,I)
$$

여기서:

- (o_): image / observation
- (p_): proprioceptive state
- (I): instruction
- (e_): VLA state embedding

Subgoal (g)는 VLA embedding input으로 넣지 않고, metadata filter로 사용한다.

즉, retrieval은 다음 두 조건을 모두 만족하는 sample 안에서 수행한다.

1. 같은 skill-wise vector DB partition에 있음
2. subgoal metadata가 현재 후보 subgoal과 유사함

skill은 vector DB partition으로 처리한다.

$$
B_t^{(m)}
$$

subgoal filtering은 다음처럼 정의한다.

$$
\mathcal{I}_{g_\tau}^{(m)}
=
\{i\in B_t^{(m)}:
d_g(g_i,g_\tau)<\rho_g^{(m)}
\}
$$

여기서:

- (g_): candidate window의 subgoal
- (g_i): buffer entry (i)의 subgoal metadata
- (d_g): subgoal distance
- (_g^{(m)}): skill (m)에 대한 subgoal metadata radius

---

## Covered Window Definition

candidate window ()에 대해, 먼저 같은 skill (m)과 유사 subgoal metadata를 갖는 buffer entries를 찾는다.

$$
\mathcal{I}_{g_\tau}^{(m)}
=
\{i\in B_t^{(m)}:
d_g(g_i,g_\tau)<\rho_g^{(m)}
\}
$$

그 안에서 VLA embedding 기준 state-neighbor를 찾는다.

$$
\mathcal{N}_{\rho_m}^{s}(e_\tau)
=
\{i\in \mathcal{I}_{g_\tau}^{(m)}
:
d_s(e_\tau,e_i)<\rho_m
\}
$$

여기서:

- (d_s): VLA embedding space에서의 state distance
- (_m): skill (m)에 대한 state-neighborhood radius
- (B_t^{(m)}): skill (m)의 현재 vector DB

데이터가 적은 초기 setting을 고려하여, covered window 판단을 위한 최소 neighbor 수는 다음처럼 둔다.

$$
k_{\min}^{(m)}=3
$$

candidate window ()가 covered되었다고 판단하는 조건은 다음과 같다.

$$
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|
\ge
3
$$

---

## Candidate-level Covered Ratio

후보 trajectory ()의 covered window set을 다음처럼 정의한다.

$$
\mathcal{T}_{\mathrm{covered}}(\xi)
=
\left\{
\tau:
|\mathcal{N}_{\rho_m}^{s}(e_\tau)|
\ge
3
\right\}
$$

후보 trajectory의 covered ratio는 다음과 같다.

$$
R_{\mathrm{cov}}(\xi)
=
\frac{
|\mathcal{T}_{\mathrm{covered}}(\xi)|
}{
|\mathcal{T}(\xi)|
}
$$

여기서:

$$
\mathcal{T}(\xi)=\{1,\dots,T\}
$$

즉, (R_{}())는 후보 trajectory의 window 중 몇 퍼센트가 현재 buffer에서 covered state로 판정되는지를 나타낸다.

---

## Skill-level Phase2-readiness

skill (m)에 대한 probe 후보들의 평균 covered ratio를 다음처럼 정의한다.

$$
\bar R_{\mathrm{cov}}^{(m)}
=
\frac{1}{|\Xi_{\mathrm{probe}}^{(m)}|}
\sum_{\xi\in\Xi_{\mathrm{probe}}^{(m)}}
R_{\mathrm{cov}}(\xi)
$$

전체 skill에 대한 Phase2-readiness score는 skill 평균으로 정의한다.

$$
R_{\mathrm{ready}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\bar R_{\mathrm{cov}}^{(m)}
$$

여기서 (M)은 전체 skill 수다.

---

## Phase1 Saturation Rule

Phase1은 다음 조건을 만족하면 종료한다.

$$
\boxed{
R_{\mathrm{ready}}>\tau_{\mathrm{ready}}
}
$$

추천 기본값은 다음과 같다.

$$
\tau_{\mathrm{ready}}=0.7
$$

즉, Phase2 probe 후보 window의 평균 70% 이상이 covered state로 판정되면 Phase2로 넘어간다.

해석:

> Phase1은 더 이상 state novelty를 많이 얻을 때가 아니라, Phase2에서 local action consistency를 평가할 수 있을 만큼 state support가 준비되었을 때 종료한다.
> 

---

# [1-2] Phase2 Saturation: MI-style Gain Saturation

Phase2의 목적은 action coverage를 늘리면서, 유사 state에서 action ambiguity를 증가시키지 않는 후보를 수집하는 것이다.

후보 trajectory의 Phase2 score는 다음과 같다.

$$
Q_2(\xi)
=
\beta \widehat{\Delta H}_A(D_t,\xi)
-
\lambda \widehat{\Delta H}_{A|S}(D_t,\xi)
$$

여기서:

$$
\widehat{\Delta H}_A(D_t,\xi)
\approx
\text{normalized action-space novelty}
$$

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
\approx
\text{local action support expansion in VLA-neighbor states}
$$

(Q_2)의 scale은 task와 candidate set에 따라 달라질 수 있으므로, 후보 batch 내부에서 normalize한다.

현재 round의 candidate set을 (_t^{(m)})라고 하자.

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

최근 (W)개 accepted trajectory의 평균 normalized score를 계산한다.

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

해석:

> 최근 accepted 후보들이 candidate batch 평균보다 더 이상 좋지 않으면, useful MI-style gain이 saturated되었다고 본다.
> 

---

# [2] Experiment: Fixed Budget + Adaptive Transition

실험에서는 baseline과 공정하게 비교하기 위해 전체 acquisition budget을 고정한다.

$$
B
$$

예시:

$$
B=100
$$

하지만 Phase1과 Phase2의 비율은 고정하지 않는다. Phase1에서 충분한 state support가 형성되면 adaptive하게 Phase2로 전환한다.

---

## [2-1] Total Budget

전체 acquisition episode 수는 다음 조건을 따른다.

$$
t\le B
$$

모든 비교 방법은 같은 총 budget을 사용한다.

$$
B_{\mathrm{ours}}=B_{\mathrm{baseline}}
$$

---

## [2-2] Phase1 Minimum / Maximum Guard

Phase1이 너무 빨리 끝나거나 너무 오래 지속되는 것을 막기 위해 minimum / maximum warm-start budget을 둔다.

$$
B_{1,\min}\le B_1\le B_{1,\max}
$$

예시:

$$
B=100,\quad B_{1,\min}=20,\quad B_{1,\max}=50
$$

---

## [2-3] Experimental Phase1 Transition Rule

Phase1은 다음 조건 중 하나를 만족하면 종료하고 Phase2로 전환한다.

$$
\boxed{
(t\ge B_{1,\min}\land R_{\mathrm{ready}}>\tau_{\mathrm{ready}})
\lor
(t=B_{1,\max})
}
$$

의미:

- 최소 (B_{1,}) episode는 state support seed를 확보한다.
- 이후 (R_{})가 threshold를 넘으면 Phase2로 전환한다.
- readiness가 늦게 만족되더라도 (B_{1,})에서는 강제 전환한다.

---

## [2-4] Experimental Phase2 Stopping Rule

Phase2는 남은 budget을 사용한다.

$$
t=B
\Rightarrow
\text{stop acquisition}
$$

다만, 남은 budget이 있어도 MI-style gain이 saturated되면 early stop할 수 있다.

$$
\bar{\tilde Q}_2^{(W)}<0
\Rightarrow
\text{early stop}
$$

따라서 실험에서 Phase2 stopping rule은 다음과 같다.

$$
\boxed{
t=B
\lor
\bar{\tilde Q}_2^{(W)}<0
}
$$

공정 비교만 강조하려면 early stop을 끄고 (t=B)까지만 진행할 수 있다.

---

# Threshold Setting

## Minimum Neighbor Count

데이터가 적은 초기 setting을 고려하여, covered state 판단을 위한 최소 neighbor 수는 다음처럼 둔다.

$$
\boxed{
k_{\min}^{(m)}=3
}
$$

---

## State-neighborhood Radius

skill (m)의 embedding set을 다음처럼 둔다.

$$
E^{(m)}=\{e_i\}_{i\in B_t^{(m)}}
$$

각 point의 (k)-NN distance를 계산한다.

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

## Subgoal Metadata Radius

VLA가 subgoal (g)를 input으로 받지 못하므로, subgoal은 metadata filtering에 사용한다.

skill (m)의 subgoal metadata set을 다음처럼 둔다.

$$
G^{(m)}=\{g_i\}_{i\in B_t^{(m)}}
$$

각 subgoal의 (k)-NN distance를 계산한다.

$$
r_{g,i}^{(m)}
=
d_k^g(g_i,G^{(m)})
$$

subgoal metadata radius는 다음처럼 정한다.

$$
\boxed{
\rho_g^{(m)}
=
\mathrm{Quantile}_{0.7}
\left(
\{r_{g,i}^{(m)}\}_{i=1}^{|G^{(m)}|}
\right)
}
$$

retrieval 시에는 다음 조건을 먼저 적용한다.

$$
d_g(g_i,g_\tau)<\rho_g^{(m)}
$$

그 다음 VLA embedding 기준 state-neighbor를 찾는다.

---

## Phase2-readiness Threshold

Phase2-readiness threshold는 다음처럼 둔다.

$$
\boxed{
\tau_{\mathrm{ready}}=0.7
}
$$

의미:

> Phase2 probe 후보 window의 평균 70% 이상이 covered state이면 Phase2로 전환한다.
> 

---

## MI-style Gain Saturation Threshold

normalized MI-style score의 saturation threshold는 다음처럼 둔다.

$$
\boxed{
\tau_{\tilde Q}=0
}
$$

즉:

$$
\bar{\tilde Q}_2^{(W)}<0
$$

이면 최근 accepted 후보들이 현재 candidate batch 평균보다 더 이상 좋지 않다고 보고 Phase2를 종료한다.

---

# Final Summary

## Method-level Rules

논문 method에서는 다음 rule을 사용한다.

$$
\boxed{
\text{Phase1 stop: } R_{\mathrm{ready}}>\tau_{\mathrm{ready}}
}
$$

$$
\boxed{
\text{Phase2 stop: } \bar{\tilde Q}_2^{(W)}<0
}
$$

---

## Experiment-level Rules

실험에서는 fixed budget + adaptive transition을 사용한다.

$$
\boxed{
\text{Phase1 stop: }
(t\ge B_{1,\min}\land R_{\mathrm{ready}}>\tau_{\mathrm{ready}})
\lor
(t=B_{1,\max})
}
$$

$$
\boxed{
\text{Phase2 stop: }
t=B
\lor
\bar{\tilde Q}_2^{(W)}<0
}
$$

---

## Core Interpretation

Phase1은 단순히 state novelty가 줄어들어서 종료되는 것이 아니다.

Phase1은 다음 조건이 만족될 때 종료된다.

$$
\text{Phase2에서 local action consistency를 평가할 수 있을 만큼 state support가 준비됨}
$$

Phase2는 다음 조건이 만족될 때 종료된다.

$$
\text{accepted 후보의 MI-style gain이 더 이상 candidate batch 평균보다 크지 않음}
$$

즉, 전체 acquisition은 다음 두 기준으로 진행된다.

1. **Phase1:** Phase2-readiness 확보
2. **Phase2:** useful MI-style gain saturation

[phase_saturation_rules](https://www.notion.so/phase_saturation_rules-3645fe31cfec80deb851fe3c662e3c47?pvs=21)

![image.png](method3%20version2(%EA%B5%AC%ED%98%84%EB%AA%85%EC%84%B8%EC%A0%95%EB%A6%AC)/image%203.png)

![image.png](method3%20version2(%EA%B5%AC%ED%98%84%EB%AA%85%EC%84%B8%EC%A0%95%EB%A6%AC)/image%204.png)

---

그럼 기존처럼 버퍼쪽이랑 같이 쓰는데, 

[phase2_mi_based_consistent_diversity_notion_safe](https://www.notion.so/phase2_mi_based_consistent_diversity_notion_safe-3645fe31cfec80328865d7f969e26874?pvs=21)

[phase2_mi_based_consistent_diversity_revised_notion_safe (copy)](https://www.notion.so/phase2_mi_based_consistent_diversity_revised_notion_safe-copy-3645fe31cfec80518c6dfaf376aceb88?pvs=21)

[민종선배 제공 노션]

[method3_one_page_logic_notion_safe](https://www.notion.so/method3_one_page_logic_notion_safe-3645fe31cfec80e9abb7c18f5dc4a30b?pvs=21)

[vla_uncertainty_mi_consistency](https://www.notion.so/vla_uncertainty_mi_consistency-3645fe31cfec8038a573ee991b1d3bbe?pvs=21)

[method3_implementation_spec_filled](https://www.notion.so/method3_implementation_spec_filled-3645fe31cfec80cc8d44cff8650e9326?pvs=21)

[phase1_subgoal_selection_notion_safe](https://www.notion.so/phase1_subgoal_selection_notion_safe-3645fe31cfec80758d11db6e6de33b48?pvs=21)

[method3_vector_db_construction_notion_safe](https://www.notion.so/method3_vector_db_construction_notion_safe-3645fe31cfec802cb9e7cc1a0b2ad12c?pvs=21)

[final_method3_spec](https://www.notion.so/final_method3_spec-3645fe31cfec80228897e856bcbfed51?pvs=21)