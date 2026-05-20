# Phase1 Subgoal Scoring: 핵심 구현 정리

## 목적

Phase1에서는 path 다양성을 크게 보상하지 않고, **subgoal-side state coverage**를 넓히는 것이 목표다.

즉, 여러 subgoal 후보 중에서:

> “이 subgoal로 가면 최종적으로 도달하는 state region이 기존 skill buffer에 비해 얼마나 새로운가?”

를 평가하여 가장 novel한 subgoal을 선택한다.

---

## 1. Subgoal 후보 생성

현재 nominal subgoal을 \(g\)라고 할 때, 주변에 여러 후보를 만든다.

$$
\mathcal{G}=\{g'_1, g'_2, \dots, g'_K\}
$$

예시:

- Gaussian offset
- shell sampling
- object-relative directional sampling

구현에서는 먼저 여러 후보를 만들고, reachable / safe하지 않은 후보는 제거한다.

$$
\mathcal{G}_{valid}
=
\{g'_j \in \mathcal{G} \mid \mathrm{reachable}(g'_j),\ \mathrm{safe}(g'_j)\}
$$

---

## 2. 각 subgoal에 대해 canonical preview trajectory 생성

각 후보 \(g'_j\)에 대해 현재 state \(S_t\)에서 해당 subgoal까지의 canonical interpolation trajectory를 만든다.

$$
\xi_j
=
\mathrm{InterpPlan}(S_t,g'_j)
=
\{\hat S^j_1, \hat S^j_2, \dots, \hat S^j_T\}
$$

여기서 \(\hat S^j_\tau\)는 실제 실행 후 observation이 아니라, planning 시점에서 얻는 preview state이다.

예시 preview state:

$$
\hat S^j_\tau
=
(q^j_\tau, x^{EE,j}_\tau, x^{EE,j}_\tau-g'_j, m)
$$

---

## 3. 마지막 구간만 평가

Phase1에서는 path 다양성을 많이 보상하지 않기 위해 전체 trajectory를 평가하지 않는다.

대신 subgoal 근처 state만 본다.

$$
\mathcal{T}_{end}
=
\text{last 20\% of preview trajectory}
$$

즉, 질문은 다음과 같다.

> 이 subgoal 후보가 도달하는 terminal state region이 기존 buffer와 겹치는가, 아니면 새로운가?

---

## 4. Preview state embedding 계산

마지막 구간의 각 preview state를 embedding으로 바꾼다.

$$
\hat e^j_\tau
=
\phi_{state}(\hat S^j_\tau, I, g'_j, m)
$$

처음 구현에서는 VLA embedding 대신 geometric descriptor를 사용해도 된다.

예시:

$$
\hat e^j_\tau
=
[q^j_\tau, x^{EE,j}_\tau, x^{EE,j}_\tau-g'_j, m]
$$

중요한 점:

> preview state와 buffer state는 반드시 같은 encoder로 embedding되어야 한다.

---

## 5. 기존 skill buffer와 비교

현재 skill \(m\)의 buffer를 다음처럼 둔다.

$$
B_t^{(m)}
=
\{e_1,e_2,\dots,e_N\}
$$

각 preview embedding \(\hat e^j_\tau\)에 대해 buffer와의 kNN 평균 거리를 계산한다.

$$
d_k^s(\hat e^j_\tau,B_t^{(m)})
=
\frac{1}{k}
\sum_{i\in\mathcal{N}_k^s(\hat e^j_\tau)}
d_s(\hat e^j_\tau,e_i)
$$

---

## 6. Buffer 내부 기준 거리 계산

거리 scale을 정규화하기 위해 buffer 내부의 평균 nearest-neighbor 거리를 사용한다.

$$
\bar d_{NN}^{s}(B_t^{(m)})
=
\frac{1}{|B_t^{(m)}|}
\sum_{e_i\in B_t^{(m)}}
\min_{l\ne i}
d_s(e_i,e_l)
$$

이 값은 현재 buffer에서 state들이 보통 어느 정도 거리로 떨어져 있는지를 나타낸다.

---

## 7. Point-level state novelty 계산

각 preview state의 novelty를 다음처럼 계산한다.

$$
n_s(\hat e^j_\tau)
=
\left[
\log
\frac{
d_k^s(\hat e^j_\tau,B_t^{(m)})
}{
\bar d_{NN}^{s}(B_t^{(m)})+\epsilon
}
\right]_+
$$

해석:

- \(n_s \approx 0\): 기존 buffer에 이미 가까운 state가 있음
- \(n_s > 0\): 기존 buffer보다 바깥쪽의 novel / under-covered state

---

## 8. Subgoal-level state gain 계산

subgoal 후보 \(g'_j\)의 score는 마지막 구간 preview states의 novelty를 aggregation해서 계산한다.

$$
G_S^{goal}(g'_j)
=
\mathrm{Agg}_{\tau\in\mathcal{T}_{end}}
\left[
n_s(\hat e^j_\tau)
\right]
$$

가장 단순한 구현은 평균이다.

$$
G_S^{goal}(g'_j)
=
\frac{1}{|\mathcal{T}_{end}|}
\sum_{\tau\in\mathcal{T}_{end}}
n_s(\hat e^j_\tau)
$$

---

## 9. 최종 subgoal 선택

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

## 10. 핵심 pseudo-code

```python
def select_phase1_subgoal(
    current_state,
    nominal_goal,
    instruction,
    skill_id,
    skill_buffer,
    K=32,
    k_nn=5,
    eps=1e-6,
):
    candidates = sample_subgoal_candidates(nominal_goal, K)

    buffer_embeddings = skill_buffer.state_embeddings
    buffer_scale = mean_nearest_neighbor_distance(buffer_embeddings)

    best_goal = None
    best_score = -float("inf")

    for g_candidate in candidates:
        if not is_reachable(g_candidate):
            continue

        preview_traj = interp_plan(current_state, g_candidate)

        # 마지막 20% 구간만 평가
        start_idx = int(0.8 * len(preview_traj.states))
        end_states = preview_traj.states[start_idx:]

        novelty_values = []

        for s_tau in end_states:
            e_tau = state_encoder(
                state=s_tau,
                instruction=instruction,
                subgoal=g_candidate,
                skill_id=skill_id,
            )

            d_knn = mean_knn_distance(
                query=e_tau,
                keys=buffer_embeddings,
                k=k_nn,
            )

            novelty = max(
                math.log(d_knn / (buffer_scale + eps)),
                0.0,
            )

            novelty_values.append(novelty)

        G_goal = mean(novelty_values)

        if G_goal > best_score:
            best_score = G_goal
            best_goal = g_candidate

    selected_traj = interp_plan(current_state, best_goal)
    return best_goal, selected_traj
```

---

## 한 줄 요약

> 여러 subgoal 후보를 만들고, 각 후보까지 canonical preview를 생성한 뒤, 마지막 20% 구간의 preview state가 기존 skill buffer와 가장 덜 겹치는 subgoal을 선택한다.

즉, Phase1의 subgoal scoring은:

$$
\boxed{
\text{subgoal-side state novelty를 최대화하되, path는 canonical하게 유지하는 선택 과정}
}
$$
