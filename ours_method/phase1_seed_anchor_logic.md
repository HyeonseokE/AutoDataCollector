# Phase1 Seed Anchoring Logic

## 1. 변경된 로직은 무엇인가?

기존에는 Phase2에서도 새로운 subgoal/state region을 계속 탐색할 수 있는 구조였다.  
변경된 로직에서는 **Phase1과 Phase2의 역할을 명확히 분리**한다.

$$
\boxed{
\text{Phase1: state/subgoal support construction}
}
$$

$$
\boxed{
\text{Phase2: action distribution refinement within seeded support}
}
$$

즉, Phase1은 **어디를 방문할지**를 정하는 단계이고, Phase2는 Phase1에서 확보한 seed를 anchor로 삼아 **어떻게 움직일지**를 정제하는 단계다.

Phase1에서는 subgoal perturbation을 통해 다양한 terminal state region을 방문하고, 각 region에 canonical trajectory seed를 남긴다.

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

이를 통해 skill별 seed subgoal set을 구축한다.

$$
\mathcal{G}_{seed}^{(m)}
=
\{g_1^{(m)},g_2^{(m)},\dots,g_N^{(m)}\}
$$

Phase2에서는 새로운 subgoal을 적극적으로 탐색하지 않고, Phase1에서 얻은 seed를 anchor로 사용한다.

$$
g\in\mathcal{G}_{seed}^{(m)}
$$

각 seed 주변에서 skill-level trajectory perturbation을 통해 action variation 후보를 만든다.

$$
\Xi_t^{(m)}(g)
=
\{\xi_{g,1},\xi_{g,2},\dots,\xi_{g,K}\}
$$

따라서 Phase2의 질문은 다음으로 바뀐다.

$$
\boxed{
\text{같은 seeded state/subgoal support 안에서 어떤 action trajectory variation이 useful한가?}
}
$$

---

## 2. 왜 이렇게 바꾸는가?

변경 이유는 **state coverage acquisition과 action distribution refinement를 분리하기 위해서**다.

Phase2에서도 새로운 subgoal/state region을 계속 넓히면, \(Q_2\) score의 해석이 애매해진다.

$$
Q_2(\xi)
=
\beta \widehat{\Delta H}_A(D_t,\xi)
-
\lambda \widehat{\Delta H}_{A|S}(D_t,\xi)
$$

여기서 \(\widehat{\Delta H}_A\)가 커졌을 때, 그 증가가 다음 중 무엇 때문인지 섞일 수 있다.

1. 새로운 action variation 때문
2. 새로운 state/subgoal region을 방문했기 때문

즉, Phase2에서 state novelty와 action novelty가 섞이면, MI-style score가 **action-level useful diversity**를 평가한다는 주장이 약해진다.

따라서 Phase2에서는 Phase1에서 만든 state/subgoal support를 anchor로 두고, 그 안에서 action trajectory variation만 비교한다. 이렇게 하면 \(Q_2\)의 의미가 명확해진다.

$$
\widehat{\Delta H}_A(D_t,\xi)
=
\text{seeded support 안에서 action coverage를 얼마나 확장하는가}
$$

$$
\widehat{\Delta H}_{A|S}(D_t,\xi)
=
\text{유사 state에서 action ambiguity를 얼마나 증가시키는가}
$$

즉, 변경된 설계의 핵심은 다음과 같다.

$$
\boxed{
\text{Phase1은 where to collect를 정하고, Phase2는 how to move를 정제한다.}
}
$$

이렇게 하면 Phase2는 단순한 state exploration이 아니라, Phase1에서 확보한 support 안에서 **learnable and consistent action distribution**을 만드는 단계로 해석된다.

---

## 3. 최종 로직 요약

### Phase1: State/Subgoal Support Construction

Phase1은 subgoal-level state coverage를 확장한다.

1. valid subgoal 후보를 생성한다.
2. 기존에 seed되지 않은 terminal state region으로 이어지는 subgoal을 선택한다.
3. 선택된 subgoal까지 canonical trajectory로 실행한다.
4. 선택된 subgoal과 canonical trajectory를 seed로 저장한다.

수식적으로는 다음과 같다.

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

Phase1 결과는 skill별 seed support다.

$$
\mathcal{G}_{seed}^{(m)}
=
\{g_1^{(m)},\dots,g_N^{(m)}\}
$$

---

### Phase2: Action Distribution Refinement

Phase2는 Phase1에서 얻은 seed를 anchor로 사용한다.

$$
g\in\mathcal{G}_{seed}^{(m)}
$$

각 seed 주변에서 skill-level trajectory perturbation으로 action variation 후보를 생성한다.

$$
\Xi_t^{(m)}(g)
=
\{\xi_{g,1},\ldots,\xi_{g,K}\}
$$

각 후보는 MI-style score로 평가한다.

$$
Q_2(\xi)
=
\beta \widehat{\Delta H}_A(D_t,\xi)
-
\lambda \widehat{\Delta H}_{A|S}(D_t,\xi)
$$

여기서:

$$
\widehat{\Delta H}_A
=
\text{action coverage gain}
$$

$$
\widehat{\Delta H}_{A|S}
=
\text{local action ambiguity increase}
$$

최종적으로 가장 useful한 action variation을 선택한다.

$$
\xi^*
=
\arg\max_{\xi\in\Xi_t^{(m)}(g)}
Q_2(\xi)
$$

accepted trajectory는 dataset과 skill-wise memory에 추가한다.

$$
B_{t+1}^{(m)}
=
P_{\mathrm{phase1}}^{(m)}
\cup
D_{\mathrm{phase2},t+1}^{(m)}
$$

---

## 핵심 한 줄

$$
\boxed{
\text{Phase1은 방문할 subgoal/state support를 만들고, Phase2는 그 seed support 안에서 useful action diversity만 선별한다.}
}
$$

이 변경을 통해 \(H(S)\) 확장과 \(H(A|S)\) 제어의 역할이 분리되고, Phase2의 MI-style score는 state novelty가 아니라 action-level useful diversity를 평가하는 기준으로 해석된다.
