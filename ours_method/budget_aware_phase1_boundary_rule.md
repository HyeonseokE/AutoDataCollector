# Budget-aware Phase1 Boundary Rule

## 목적

총 수집 episode budget이 정해져 있는 실험 setting에서 Phase1과 Phase2를 단순히 고정 비율로 나누지 않고, **Phase1을 도는 동안 현재 state support가 충분한지와 남은 budget으로 Phase2에서 유용한 데이터를 충분히 얻을 수 있는지**를 함께 고려하여 phase boundary를 정한다.

핵심 질문은 다음과 같다.

$$
\boxed{
\text{지금 Phase1을 종료하고 Phase2로 넘어가도, 남은 budget으로 충분한 MI-useful trajectory를 수집할 수 있는가?}
}
$$

---

## 1. 문제 설정

총 수집 budget을 다음처럼 둔다.

$$
B
$$

현재까지 Phase1에서 수집한 episode 수를 \(t\)라고 하면, 남은 budget은 다음과 같다.

$$
B_{\mathrm{rem}}(t)
=
B-t
$$

Phase1을 너무 일찍 종료하면 state/subgoal support가 부족하여 Phase2에서 local action consistency를 안정적으로 평가하기 어렵다.

반대로 Phase1을 너무 오래 수행하면 Phase2에 남은 budget이 부족하여 action distribution refinement를 충분히 수행하지 못한다.

따라서 Phase1 boundary는 다음 두 조건을 함께 만족해야 한다.

$$
\boxed{
\text{State support가 충분히 준비되었는가?}
}
$$

$$
\boxed{
\text{남은 budget으로 Phase2에서 MI-useful 후보를 충분히 얻을 수 있는가?}
}
$$

---

## 2. Phase1 Readiness: State Support 준비도

Phase1 readiness는 Phase2 candidate가 현재 buffer 안에서 얼마나 covered state로 평가될 수 있는지를 본다.

후보 trajectory \(\xi\)의 covered ratio는 다음과 같다.

$$
R_{\mathrm{cov}}(\xi)
=
\frac{
|\mathcal{T}_{covered}(\xi)|
}{
|\mathcal{T}(\xi)|
}
$$

여기서:

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

skill \(m\)에 대해 probe candidate set \(\Xi_{\mathrm{probe}}^{(m)}\)을 만들고, 평균 covered ratio를 계산한다.

$$
\bar R_{\mathrm{cov}}^{(m)}
=
\frac{1}{|\Xi_{\mathrm{probe}}^{(m)}|}
\sum_{\xi\in\Xi_{\mathrm{probe}}^{(m)}}
R_{\mathrm{cov}}(\xi)
$$

전체 readiness score는 skill 평균으로 정의한다.

$$
R_{\mathrm{ready}}(t)
=
\frac{1}{M}
\sum_{m=1}^{M}
\bar R_{\mathrm{cov}}^{(m)}
$$

Phase1 state support가 준비되었다고 판단하는 조건은 다음과 같다.

$$
\boxed{
R_{\mathrm{ready}}(t)
\ge
\tau_{\mathrm{ready}}
}
$$

기본값은 다음처럼 둔다.

$$
\boxed{
\tau_{\mathrm{ready}}=0.7
}
$$

---

## 3. Phase2 Expected MI-usefulness

Phase1 boundary에서는 model-side informativeness \(U_{\mathrm{VLA}}\)를 사용하지 않는다.

그 이유는 Phase1 중에는 아직 Phase1-trained VLA가 없거나 충분히 안정적이지 않기 때문이다. 따라서 Phase1 종료 판단은 VLA uncertainty가 아니라 **MI-side expected usefulness**만 사용한다.

현재 buffer \(D_t\)를 기준으로 Phase2 probe candidate를 생성한다.

$$
\Xi_{\mathrm{probe}}
=
\bigcup_{m=1}^{M}
\Xi_{\mathrm{probe}}^{(m)}
$$

각 probe candidate \(\xi\)에 대해 MI-side usefulness score를 계산한다.

$$
M_{\mathrm{MI}}(D_t,\xi)
=
\beta
\widehat{\Delta H}_A(D_t,\xi)
-
\lambda
\widehat{\Delta H}_{A|S}(D_t,\xi)
$$

현재 probe candidate batch 안에서 normalize한다.

$$
\tilde M_{\mathrm{MI}}(D_t,\xi)
=
\frac{
M_{\mathrm{MI}}(D_t,\xi)-\mu_M
}{
\sigma_M+\epsilon
}
$$

여기서 \(\mu_M\)과 \(\sigma_M\)은 현재 probe candidate batch의 평균과 표준편차이다.

Phase2에서 기대되는 per-episode positive MI gain은 다음과 같이 정의한다.

$$
\bar G_{\mathrm{phase2}}(t)
=
\mathbb{E}_{\xi\in\Xi_{\mathrm{probe}}}
\left[
\max
\left(
\tilde M_{\mathrm{MI}}(D_t,\xi),
0
\right)
\right]
$$

해석하면 다음과 같다.

$$
\boxed{
\bar G_{\mathrm{phase2}}(t)
=
\text{지금 Phase2로 넘어갔을 때, 한 episode당 기대되는 positive MI-useful gain}
}
$$

Phase2에서 수집할 만한 후보가 충분히 있다고 판단하는 조건은 다음과 같다.

$$
\boxed{
\bar G_{\mathrm{phase2}}(t)
\ge
\tau_{\mathrm{gain}}
}
$$

기본값은 다음처럼 둔다.

$$
\boxed{
\tau_{\mathrm{gain}}=0.25
}
$$

즉, probe candidate들의 positive MI-usefulness가 candidate batch 표준편차 기준으로 평균 \(0.25\) 이상이면, Phase2에서 남은 budget을 의미 있게 사용할 수 있다고 판단한다.

---

## 4. Budget Guard

총 budget이 정해진 실험에서는 Phase1이 너무 짧거나 너무 길어지는 것을 막기 위해 minimum / maximum guard를 둔다.

$$
B_{1,\min}
\le
B_1
\le
B_{1,\max}
$$

예시 기본값은 다음과 같다.

$$
\boxed{
B_{1,\min}=20,\quad B_{1,\max}=50
}
$$

여기서:

- \(B_{1,\min}\): Phase1을 최소한 이만큼은 수행해야 함
- \(B_{1,\max}\): 이 시점까지도 readiness가 부족하면 강제로 Phase2로 전환

---

## 5. 최종 Phase1 Transition Rule

최종 Phase1 종료 조건은 다음과 같다.

$$
\boxed{
\left[
t\ge B_{1,\min}
\land
R_{\mathrm{ready}}(t)\ge\tau_{\mathrm{ready}}
\land
\bar G_{\mathrm{phase2}}(t)\ge\tau_{\mathrm{gain}}
\right]
\lor
\left[
t=B_{1,\max}
\right]
}
$$

즉, Phase1은 다음 조건을 모두 만족하면 종료된다.

1. 최소 Phase1 budget을 이미 채웠다.
2. Phase2에서 local consistency를 평가할 수 있을 만큼 state support가 준비되었다.
3. 남은 Phase2에서 얻을 수 있는 expected MI-useful gain이 충분하다.

단, \(B_{1,\max}\)에 도달하면 Phase1을 강제로 종료한다.

---

## 6. 전체 로직 요약

```text
Given total budget B:

for each Phase1 step t:

    1. 현재까지의 Phase1 buffer로 state readiness 계산
       R_ready(t)

    2. Phase2 probe candidates 생성
       Xi_probe

    3. 각 probe candidate의 MI-side usefulness 계산
       M_MI(D_t, xi)

    4. batch-normalized MI score 계산
       M_tilde_MI(D_t, xi)

    5. expected positive Phase2 gain 계산
       G_bar_phase2(t)

    6. 아래 조건을 만족하면 Phase1 종료:
       t >= B_1,min
       R_ready(t) >= tau_ready
       G_bar_phase2(t) >= tau_gain

    7. 또는 t == B_1,max이면 강제 종료
```

---

## 7. 핵심 직관

이 boundary rule은 Phase1을 단순히 고정 episode 수로 자르지 않는다.

대신 다음 두 가지를 동시에 확인한다.

$$
\boxed{
\text{Phase2 scoring이 가능할 만큼 state support가 충분한가?}
}
$$

$$
\boxed{
\text{남은 budget으로 Phase2에서 MI-useful action variation을 얻을 가능성이 충분한가?}
}
$$

따라서 Phase1 boundary는 다음 trade-off를 조절한다.

$$
\boxed{
\text{too early transition}
\Rightarrow
\text{insufficient state support}
}
$$

$$
\boxed{
\text{too late transition}
\Rightarrow
\text{insufficient Phase2 budget}
}
$$

---

## 8. 현재 Useful OOD 버전과의 관계

Phase1 boundary에서는 \(U_{\mathrm{VLA}}\)를 사용하지 않는다.

$$
\boxed{
\text{Phase1 boundary: } R_{\mathrm{ready}} + \bar G_{\mathrm{phase2}}
}
$$

Phase2 내부 selection에서만 \(U_{\mathrm{VLA}}\)를 사용한다.

$$
\boxed{
\text{Phase2 selection: }
\arg\max U_{\mathrm{VLA}}(\xi)
\quad
\mathrm{s.t.}
\quad
\tilde M_{\mathrm{MI}}(D_t,\xi)\ge0
}
$$

즉, Phase1 boundary는 **MI-side expected usefulness**로 결정하고, Phase2 selection은 **MI-useful 후보 중 VLA-informative 후보를 선택**하는 방식이다.
