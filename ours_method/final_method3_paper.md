# method파트

마감여부: 시작 전

> 
> 
> 
> 3.1 Problem Formulation
> - 모든 자동 생성 trajectory를 수집할 수 없음
> - 모든 diversity가 useful하지 않음
> - online candidate selection 문제로 정의
> - marginal MI gain 도입
> 
> 3.2 Overall Framework
> - offline curation은 diverse state pool을 가정
> - online acquisition은 state pool도 직접 만들어야 함
> - Phase1이 필요한 이유:
> (1) state coverage 확보
> (2) action quality 추정을 위한 local seed 확보
> - Phase2는 그 support 안에서 MI-style action quality 평가
> 
> 3.3 State Coverage Seeding via Canonical Subgoal Exploration
> - 문제: 초기에는 state coverage도 없고 local action reference도 없음
> - 해결: subgoal diversity로 state region 확장
> - canonical path로 각 region에 representative action seed 저장
> - 즉, Phase2의 local consistency 판단 기준을 만든다
> 
> 3.4 Consistent Diversity Scoring via Marginal MI Gain
> - 문제: 새로운 action이 항상 useful한 것은 아님
> - Phase1 seed를 기준으로 local state neighborhood와 action support 비교
> - ΔH_A: action coverage gain
> - ΔH_A|S: local action ambiguity increase
> - Q2 score 정의
> 
> 3.5 Acquisition Policy: Selecting Useful Diversity
> - Phase2에서 후보를 decision으로 변환
> - under-covered 후보는 state-seeding으로
> - redundant는 reject
> - ambiguous novelty는 reject/down-rank
> - useful diversity만 accept
> 

## 3. Method

### 3.1 Problem Formulation: Pre-selective Real-world Acquisition

**목적: 우리는 online candidate selection문제를 푼다**

Ours의 자동 데이터 취득 파이프라인은 지속적인 real-world data collection을 가능하게 한다.

그러나 real-world에서는 모든 후보를 실행하는 것이 비용적으로 불가능하거나 비효율적이다. 또한 모든 trajectory diversity가 학습에 유용한 것은 아니다.

- **디테일한 내용**
    
    그러나 명확한 기준 없이 모든 trajectory를 수집하는 것은 두 가지 측면에서 비효율적이다.
    첫째, **운영 비용 측면의 비효율**.
    실세계 데이터 수집 과정은 로봇 embodiment를 완전히 점유하고, 하드웨어 마모와 운영 비용을 수반한다.
    둘째, **모델 및 학습 측면의 비효율**.
    데이터의 diversity는 distribution shift robustness에 필수적이다. 그러나 단순히 dataset quantity와 state coverage를 늘리는 것이 항상 유용한 supervision을 의미하지는 않는다. 중복 trajectory는 정보 이득이 낮고 storage, training cost의 비효율을 야기한다. 그리고 유사한 observation/instruction에서 incompatible action mode가 섞인 trajectory는 action ambiguity를 증가시켜 flow-matching / diffusion 기반 VLA의 학습과 closed-loop execution을 불안정하게 만들 수 있다.(citation)
    

그렇다면 다음 연구질문으로 연결된다.
Among generated trajectory candidates, which one should be physically executed and added to the dataset?

제한된 자원 및 효율성을 고려하여 어떤 데이터를 모을 지 “사전 선별하는 문제”로 귀결된다

기존 offline data curation연구는 Mutual Information(MI)을 data qulity의 기준으로 선정했다. 직관적으로 데이터셋의 action 다양성은 늘리되, 유사한 state에서 action은 일관되어야 한다는 기준이다.

$I(S;A)=H(A)−H(A∣S)$

하지만 MI는 완성된 dataset이나 이미 수집된 data pool에서의 적용은 자연스럽다(citation). 
우리의 경우에는 후보 trajectory를 현재 buffer에 추가할 지 말 지를 결정해야 하므로, 초기 상태와 후보를 추가한 상태 사이의 차이인 marginal MI gain으로 바꿔 구한다.

$Gain_{MI}(D_{t},\xi)=I(D_{t}∪{\xi})−I(D_{t})$

$Gain_{MI}(D_t,\xi)
=
\Delta H_A(D_t,\xi)
-
\Delta H_{A|S}(D_t,\xi)$

### 3.2 Overall Framework: Two-phase Acquisition

**목적: (1) offline curation의 online acqusition 차이. (2) 전체 그림**

기존 offline curation에서는 이미 다양한 state에서 취득된 action 데이터셋이 존재한다. 그래서 MI를 바로 적용할 수 있을 뿐 만 아니라 결과적으로 충분한 state coverage를 확보한 고품질 subset 데이터셋을 확보할 수 있다.(**이미 충분히 다양한 state를 포함한 pool이 존재한다면**, MI기반 filtering은 그 state coverage 안에서 action diversity를 유지하면서, 유사한 state에서는 일관된 action을 갖는 demonstration을 선별)

하지만 online acquisition에서는 고정된 pool에서 subset을 선택하는 것이 아니라, real-world interaction을 통해 dataset 자체를 점진적으로 구축하는 문제이다. 따라서 초기에 cold start문제에 빠진다. 이 상태에서는 유사 state 주변의 action sample이 부족하기 때문에 $H(A∣S)$를 안정적으로 판단하기 어렵다. 또한 MI은 state coverage를 제어하지 않는다. 따라서 MI만을 고려하면 초기 state coverage에 종속되는 문제가 있다.

따라서 충분한 초기 state coverage를 제공하고, 그 안에서 $H(A∣S)$를 안정적으로 추정할 수 있도록 전체 online acquistion과정을 두 단계로 분리한다.

**Phase 1: State Coverage Seeding**

**Phase 2: Consistent Diversity Acquisition**

Phase1은 기존 offline curation이 암묵적으로 가정하던 “충분히 다양한 state pool”을 online acquisition 과정에서 만들고, 이후 local action consistency를 평가할 수 있을 만큼 sample seed를 형성하는 것이다.(MI계산 X) 

Phase2의 목적은 Phase1에서 형성된 state support 안에서 mutual information의 직관을 적용하여 전체 action coverage는 확장하되 유사 state에서의 action ambiguity는 증가시키지 않는 trajectory만 선별 수집한다.

**정리:** two-phase acquistion을 통해 기존 offline curation이 암묵적으로 가정하던 “충분히 다양한 state pool”을 online acquisition 과정에서 먼저 직접 만들고, 그 이후에 mutual information의 직관을 적용하는 방식. state coverage 확보와 action distribution 정제를 분리한다

### 3.3 State Coverage Seeding via Canonical Subgoal Exploration

**목적: (1) state coverage 확보, (2) Cold-start problem of H(A|S) 해결**

Online acquisition 초기에는 유사 state 주변의 action sample이 충분하지 않기 때문에 $p(A∣S)$를 안정적으로 추정할 수 없다. 따라서 $H(A∣S)$를 기반으로 어떤 후보가 consistent한지, 혹은 ambiguous한지를 판단하기 어렵다.

이에 따라 Phase1에서는 action consistency를 판단하지 않고, 먼저 충분한 state/subgoal support를 형성한다. 다만 state coverage를 늘리는 과정에 trajectory path 자체를 다양화하면, 유사 state 주변 여러 action mode가 생겨 이후 $H(A∣S)$를 증가시킬 수 있다. 따라서 Phase1에서는 다음 원칙을 따른다.

- Diversify subgoals, but keep paths canonical.

즉, subgoal perturbation을 통해 다양한 terminal state region을 방문하되, 각 subgoal로 이동하는 trajectory는 canonical interpolation planning으로 유지한다. 현재 state를 $S_t$, valid subgoal 후보 집합을 $\mathcal{G}_{valid}$라고 할 때, Phase1은 terminal state region novelty가 가장 큰 subgoal을 선택한다.

$g^*
=
\arg\max_{g'_j\in\mathcal{G}_{valid}}
G_S^{goal}(g'_j)$

선택된 subgoal까지는 canonical trajectory로 실행한다.

$\xi^*
=
\mathrm{InterpPlan}(S_t,g^*)$

여기서 $G_S^{goal}(g'_j)$는 전체 path novelty가 아니라, 후보 subgoal이 기존에 seed되지 않은 terminal state region으로 이어지는 정도를 나타낸다. 이를 위해 Phase1에서는 skill-wise subgoal buffer $B_{g,t}^{(m)}$를 유지한다. 이 buffer는 skill $m$에서 이미 seed한 terminal region을 기록하는 lightweight memory이며, action consistency를 계산하기 위한 것이 아니라 같은 terminal region의 반복 수집을 막기 위한 것이다.

결과적으로 Phase1은 under-covered state region을 확장하는 동시에, 각 region에 대해 canonical action seed를 남긴다. 이 seed는 Phase2에서 새로운 후보 trajectory가 consistent action variation인지, ambiguous action variation인지 판단하는 local reference로 사용된다.

### 3.4 Consistent Diversity Scoring via Marginal MI Gain

Phase1를 통해 state/subgoal support와 canonical seed가 형성된다. phase2의 목적은 Phase1에서 seed된 support를 anchor로 하여 action distribution을 정제하는 것이다. 이를 위하여 각 seed 주변에서 skill-level trajectory perturbation을 통해 여러 action variation 후보를 생성한다.

$g∈G_{seed}^{(m)}​$

$Ξ_t^{(m)}​(g)=\{ξ_{g,1}​,…,ξ_{g,K}​\}$

새로운 traj pattern은 전체 state coverage를 확장할 수 있지만 , 이미 커버된 유사 state 주변에서 phase1 action support와 충돌하면 $H(A∣S)$를 증가시켜 ambiguous supervision을 만들 수 있다

따라서 Phase2에서는 후보 trajectory $\xi$를 현재 buffer $D_t$에 추가했을 때의 marginal MI gain을 기준으로 평가한다.

$Gain_{MI}(D_t,\xi)
=
\Delta H_A(D_t,\xi)
-
\Delta H_{A|S}(D_t,\xi)$

여기서 $\Delta H_A(D_t,\xi)$는 action coverage gain을, $\Delta H_{A|S}(D_t,\xi)$는 유사 state에서의 action ambiguity increase를 의미한다. 따라서 좋은 후보는 action coverage를 확장하되, local ambiguity는 증가시키지 않아야 한다.

$\Delta H_A \uparrow,
\quad
\Delta H_{A|S}\downarrow$

실제 robot trajectory는 continuous하고 고차원이므로 entropy를 직접 추정하는 것은 매우 어렵다.
따라서 descriptor-space proxy로 근사한다.

$\widehat{ΔH}_A​(D_t​,ξ)≈action~descriptor~novelty$

$\widehat{\Delta H}_{A|S}(D_t,\xi)
\approx
\text{local action support expansion around similar states}$

이를 바탕으로 MI-side usefulness score를 다음과 같이 정의한다.

$M_{MI}​(D_{t}​,ξ)=β\widehat{ΔH_{A}}​(Dt​,ξ)−λ\widehat{ΔH_{A∣S}}​(D_{t}​,ξ)$

즉, $M_{MI}$는 candidate가 buffer 관점에서 학습에 유용한지 평가한다. 높은 $M_{MI}$는 후보가 action coverage를 확장하면서도 local action ambiguity를 크게 증가시키지 않는다는 것을 의미한다.

Phase2에서는 Phase1 seed와 accepted Phase2 data로 구성된 skill-wise memory를 사용한다. Candidate state의 retrieval key는 Phase1 data로 학습한 frozen VLA encoder를 사용해 구성한다.

$e_\tau​=[ϕ_{VLA}^{(1)}​(o_{\tau}​,I);p_\tau​]$

같은 skill memory 안에서 유사 state neighbor를 찾고,

$N_{ρm}^{​s}​(e_\tau​)=\{i∈B_t^{(m)}​:d_s​(e_\tau​,e_i​)<ρ_{m}\}$

local action support는 충분한 neighbor가 존재하는 covered state neighborhood에서 계산한다.

$∣N^{s}_{ρ_{m}}​(e_\tau​)∣≥k_{min}^{(m)}$

이 covered-state 조건은 새로운 acquisition category를 만들기 위한 것이 아니라, Phase2 scoring이 유효하게 계산될 수 있는지 확인하는 validity check이다. 후보 action이 기존 local action support 안에 있으면 consistent하다고 보고, support 바깥에 있으면 유사 state에서 새로운 action mode를 추가하는 것으로 간주한다. 

여기에 더해, Phase1-trained VLA를 model-side informativeness estimator로 사용한다. 기존 diffusion/VLA uncertainty 연구에서는 높은 uncertainty를 test-time OOD 또는 failure signal로 보고 reject하거나 intervention을 요청하는 기준으로 사용하였다. 

반면 online acquisition setting에서는 OOD-like 후보를 무조건 제거하지 않는다. online acquisition에서는 OOD-like candidate가 중요한 학습 기회가 될 수 있다. 오히려 VLA 입장에서는 낯설지만, buffer-side MI 기준으로는 유용한 후보는 acquisition-worthy candidate로 본다.

이를 위해 후보 trajectory의 VLA-side uncertainty를 single-step denoising loss로 정의한다.(citation) Diffusion policy의 denoising loss는 sampled noise와 timestep에 따라 달라질 수 있으므로, 각 action chunk에 대해 서로 다른 stochastic denoising evaluation을 R회 수행하고 평균한다.

$U_{VLA}​(ξ)=Agg_{τ}​[{1\over R} \sum_{​r=1}^{R}​L_{denoise}^{(r)}​(o_\tau​,I,p_\tau​​,A_{\tau​:_\tau​+H−1}​;π_{θ}^{(1)}​)]$

이는 Phase1-trained VLA가 후보 action trajectory를 얼마나 낯설어하는지를 나타내는 model-side novelty 또는 informativeness score이다.

따라서 Phase2의 후보 평가는 두 축으로 이루어진다.

$MI-side~usefulness:~M_{MI}​(D_{t}​,ξ)$

$VLA-side~informativeness:~U_{VLA}​(ξ)$

$M_{MI}$는 후보가 buffer-side에서 MI 관점으로 useful한 지를 판단하고, $U_{\mathrm{VLA}}$는 그 후보가 model-prior 관점에서 얼마나 informative한 지를 나타낸다.

### 3.5 Acquisition Policy: Selecting Useful Diversity

3.4에서 정의한 $M_{MI}​$와 $U_{VLA}$는 서로 다른 두 관점의 score이다. $M_{MI}​$는 buffer-side usefulness를 평가하고, $U_{VLA}$는 model-side informativeness로 모델이 얼마나 해당 후보를 어려워하는 지 평가한다. 본 절에서는 이 두 축을 결합하여 실제 acquisition decision으로 변환한다.즉, 기존 test-time policy에서 OOD는 주로 reject해야 할 위험 신호로 다루어졌지만, online acquisition에서는 OOD-like candidate가 중요한 학습 기회가 될 수 있다. 중요한 것은 OOD 여부 자체가 아니라, 그 후보가 buffer 기준으로 useful한지 여부이다.

우리는 Phase2 후보를 다음의 $2\times2$ confusion matrix로 해석한다.

![image.png](method%ED%8C%8C%ED%8A%B8/image.png)

첫째, **Useful OOD**는 VLA 입장에서는 낯설지만, buffer-side MI 기준으로는 유용한 후보이다.

$U_{VLA}​(ξ)↑,M_{MI}​(D_{t}​,ξ)↑$

이 후보는 Phase1-trained VLA가 아직 잘 설명하지 못하는 action trajectory이지만, 동시에 action coverage를 확장하고 local action ambiguity를 크게 증가시키지 않는다. 따라서 이는 본 연구가 우선적으로 수집하고자 하는 핵심 대상이다.

둘째, **Harmful OOD**는 VLA 입장에서는 낯설지만, MI 기준으로는 유용하지 않은 후보이다.

$U_{VLA}​(ξ)↑,M_{MI}​(D_{t}​,ξ)↓$

이 경우 높은 uncertainty는 유용한 novelty가 아니라, ambiguous action mode, IK artifact, 불필요한 detour, 불안정한 trajectory 등에서 비롯될 수 있다. 따라서 단순히 VLA uncertainty만 높다고 accept하면 harmful OOD를 수집할 위험이 있다.

셋째, **Useful ID**는 VLA 입장에서는 익숙하지만, buffer 기준으로는 아직 유용한 후보이다.

$U_{VLA}​(ξ)↓,M_{MI}​(D_{t}​,ξ)↑$

이는 pretrained prior 안에 있는 안정적인 action pattern이지만, 현재 task-specific buffer에서는 아직 충분히 커버되지 않은 후보일 수 있다. 따라서 useful diversity로 볼 수 있지만, model-side informativeness 관점에서는 Useful OOD보다 우선순위가 낮다.

마지막으로, **Redundant ID**는 VLA도 잘 설명하고, buffer 기준으로도 유용하지 않은 후보이다.

$U_{VLA}​(ξ)↓,M_{MI}​(D_{t}​,ξ)↓$

이는 이미 많이 본 trajectory이거나 action coverage gain이 낮은 후보이므로 reject 한다.

이 관점에서 **MI-useful 후보 중 가장 informative한 OOD를 우선 선별 및 수집한다.**

$ξ^∗=argmax_{ξ∈Ξ_{t}^{(m)}​(g)}​U_{VLA}​(ξ)$  $s.t.~ M_{MI}​(D_{t}​,ξ)≥τ_{MI}$

---

![image.png](method%ED%8C%8C%ED%8A%B8/image%201.png)

![image.png](method%ED%8C%8C%ED%8A%B8/image%202.png)