# Method3: Skill-wise Pre-selective Acquisition

## Skill-wise Candidate Selection

현재 skill $m$에서 candidate segment set은 다음과 같다.

$$
\Xi_m=\{\xi_m^1,\xi_m^2,\dots,\xi_m^K\}
$$

각 candidate $\xi_m^j$에 대해 $IG(\xi_m^j)$와 $AC(\xi_m^j)$를 계산하고, 최종적으로 다음 score가 가장 큰 후보를 선택한다.

$$
Score(\xi_m^j)=IG(\xi_m^j)\cdot AC(\xi_m^j)
$$

$$
\xi_m^*
=
\arg\max_{\xi_m^j\in\Xi_m}
Score(\xi_m^j)
$$

---

# 1. Information Gain, $IG(\xi)$

## Purpose

후보 trajectory가 **기학습 VLA 기준으로 새롭고**, 동시에 **미학습 buffer 기준으로 중복되지 않는지** 측정한다.

---

## 1.1 Model-side Informativeness

기학습 VLA $\pi_0$가 candidate action segment를 얼마나 잘 예측하지 못하는지 flow-matching loss로 측정한다.

$$
U_{\pi_0}(\xi_m^j)
=
\mathcal{L}_{FM}
(O_m,S_m,I,A_{\xi_m^j};\pi_0)
$$

간단히 쓰면:

$$
U_{\pi_0}(\xi_m^j)
=
\left\|
v_{\pi_0}(A_t,t,c_m)-u_t
\right\|_2^2
$$

where

$$
c_m=
[
h^{VL}_{\pi_0}(O_m,I),
h^{prop}_{\pi_0}(S_m)
]
$$

$$
A_t=(1-t)A_0+tA_{\xi_m^j}
$$

$$
u_t=A_{\xi_m^j}-A_0
$$

해석:

$$
U_{\pi_0}\uparrow
\Rightarrow
\text{기학습 VLA 기준 novelty 증가}
$$

단, $U_{\pi_0}$는 candidate action segment가 기학습 VLA 기준으로 얼마나 낯선지를 측정하는 **model-side novelty signal**일 뿐이다. 해당 novelty가 실제로 유용한지, 혹은 action ambiguity를 증가시키는지는 $AC(\xi)$와 함께 판단한다. 즉, $U_{\pi_0}$ 자체로 useful / ambiguous novelty를 구분하지 않는다.

---

## 1.2 Buffer-side Novelty

현재 Phase2에서 수집되었지만 아직 학습에 사용되지 않은 동일 skill buffer를 정의한다.

$$
B_t^{(m)}
=
\{\xi_i\in B_t \mid skill(\xi_i)=m\}
$$

Flow-matching forward 과정에서 얻은 action-head hidden feature를 embedding으로 사용한다.

$$
z(\xi_m^j)
=
h_{\pi_0}^{FM}
(O_m,S_m,I,A_{\xi_m^j})
$$

embedding을 정규화한다.

$$
\bar z(\xi)
=
\frac{z(\xi)}
{\|z(\xi)\|_2+\epsilon}
$$

동일 skill buffer와의 nearest-neighbor distance를 novelty로 둔다.

$$
N_{B_t}^{(m)}(\xi_m^j)
=
\min_{\xi_i\in B_t^{(m)}}
\left\|
\bar z(\xi_m^j)-\bar z(\xi_i)
\right\|_2
$$

해석:

$$
N_{B_t}^{(m)}\uparrow
\Rightarrow
\text{동일 skill buffer 기준 non-redundant}
$$

---

## 1.3 Normalize and Merge IG

현재 candidate set $\Xi_m$ 안에서 min-max normalization한다.

$$
\mathrm{Norm}(Q(\xi_m^j))
=
\frac{
Q(\xi_m^j)-\min_k Q(\xi_m^k)
}{
\max_k Q(\xi_m^k)-\min_k Q(\xi_m^k)+\epsilon
}
$$

따라서:

$$
\tilde U_{\pi_0}(\xi_m^j)
=
\mathrm{Norm}(U_{\pi_0}(\xi_m^j))
$$

$$
\tilde N_{B_t}^{(m)}(\xi_m^j)
=
\mathrm{Norm}(N_{B_t}^{(m)}(\xi_m^j))
$$

최종 IG는 geometric mean으로 결합한다.

$$
IG(\xi_m^j)
=
\left(
\tilde U_{\pi_0}(\xi_m^j)
\right)^\alpha
\left(
\tilde N_{B_t}^{(m)}(\xi_m^j)
\right)^{1-\alpha}
$$

기본값은:

$$
\alpha=0.5
$$

즉:

$$
IG(\xi_m^j)
=
\sqrt{
\tilde U_{\pi_0}(\xi_m^j)
\tilde N_{B_t}^{(m)}(\xi_m^j)
}
$$

---

# 2. Action Consistency, $AC(\xi)$

## Purpose

후보 trajectory가 **유사한 state/context에서 기학습 VLA가 예측한 action mode 및 미학습 buffer에서 관측된 local expert action mode와 가까운지** 측정한다. 이를 통해 $H(A|S)$를 직접 계산하지 않고, state-conditioned action ambiguity를 증가시키지 않는 action-consistent 후보를 근사적으로 선별한다.

---

## 2.1 Model-side AC

Model-side AC는 single-step FM loss가 아니라, **기학습 VLA \(\pi_0\)가 현재 context에서 예측한 action mode와 candidate action이 얼마나 가까운지**로 측정한다. 즉, \(P_{\pi_0}(A^{1:H}\mid O_m,S_m,I)\)에서 높은 확률로 선택될 수 있는 action mode와 candidate가 일치할수록 action-consistent하다고 본다.

먼저 현재 context를 다음처럼 둔다.

$$
x_m=(O_m,S_m,I,m)
$$

기학습 VLA에서 현재 context에 대한 action chunk들을 샘플링한다.

$$
\hat A_{1}^{1:H},\dots,\hat A_{M}^{1:H}
\sim
\pi_0(\cdot\mid O_m,S_m,I)
$$

샘플링된 action chunks를 벡터화한다.

$$
\hat a_r=\mathrm{vec}(\hat A_r^{1:H})
$$

샘플링된 action chunks로 model-predicted action mode set을 만든다.

$$
\mathcal{M}_{\pi_0}(x_m)=\{p_1,\dots,p_R\}
$$

여기서 \(p_r\)는 \(\pi_0\)가 현재 context에서 예측한 action mode prototype이다. 구현상으로는 샘플 action들을 clustering하여 \(R\)개의 prototype을 만들거나, 간단히 \(M\)개의 sampled action 자체를 mode set으로 사용할 수 있다.

candidate action을 벡터화한다.

$$
a_j=\mathrm{vec}(A_{\xi_m^j})
$$

candidate가 가장 가까운 model-predicted action mode와 얼마나 가까운지 계산한다.

$$
D_{\mathrm{model\text{-}AC}}^{j}
=
\min_{p_r\in\mathcal{M}_{\pi_0}(x_m)}
\|a_j-p_r\|_2^2
$$

거리 값이 낮을수록 model-side action consistency가 높으므로, normalization 후 방향을 뒤집는다.

$$
AC_{\mathrm{model}}(\xi_m^j)
=
1-
\mathrm{Norm}(D_{\mathrm{model\text{-}AC}}^{j})
$$

즉:

$$
AC_{\mathrm{model}}(\xi_m^j)
=
1-
\frac{
D_{\mathrm{model\text{-}AC}}^{j}
-
\min_k D_{\mathrm{model\text{-}AC}}^{k}
}{
\max_k D_{\mathrm{model\text{-}AC}}^{k}
-
\min_k D_{\mathrm{model\text{-}AC}}^{k}
+
\epsilon
}
$$

해석:

$$
D_{\mathrm{model\text{-}AC}}^{j}\downarrow
\Rightarrow
AC_{\mathrm{model}}(\xi_m^j)\uparrow
$$

즉, candidate action이 기학습 VLA가 현재 state에서 예측한 action mode와 가까울수록 model-side AC가 높다.

> Note: single-step FM loss는 model-side informativeness \(U_{\pi_0}\)를 계산하는 데 사용하고, model-side AC는 \(\pi_0\)가 실제로 예측/샘플링한 action mode와 candidate action의 거리로 계산한다.

---

## 2.2 Buffer-side AC

현재 context를 다음처럼 둔다.

$$
x_m=(O_m,S_m,I,m)
$$

동일 skill buffer $B_t^{(m)}$에서 $x_m$과 유사한 context를 가진 segment들을 찾는다.

$$
\mathcal{N}_k(x_m;B_t^{(m)})
$$

이웃들의 action chunks로 local expert action mode set을 만든다.

$$
\mathcal{M}_B(x_m)=\{c_1,\dots,c_R\}
$$

candidate action을 벡터화한다.

$$
a_j=\mathrm{vec}(A_{\xi_m^j})
$$

candidate가 가장 가까운 expert mode와 얼마나 가까운지 계산한다.

$$
D_{\mathrm{buffer\text{-}AC}}^{j}
=
\min_{c_r\in\mathcal{M}_B(x_m)}
\|a_j-c_r\|_2^2
$$

거리 값이 낮을수록 action-consistent하므로, normalization 후 방향을 뒤집는다.

$$
AC_{\mathrm{buffer}}(\xi_m^j)
=
1-
\mathrm{Norm}(D_{\mathrm{buffer\text{-}AC}}^{j})
$$

즉:

$$
AC_{\mathrm{buffer}}(\xi_m^j)
=
1-
\frac{
D_{\mathrm{buffer\text{-}AC}}^{j}
-
\min_k D_{\mathrm{buffer\text{-}AC}}^{k}
}{
\max_k D_{\mathrm{buffer\text{-}AC}}^{k}
-
\min_k D_{\mathrm{buffer\text{-}AC}}^{k}
+\epsilon
}
$$

해석:

$$
D_{\mathrm{buffer\text{-}AC}}^{j}\downarrow
\Rightarrow
AC_{\mathrm{buffer}}(\xi_m^j)\uparrow
$$

즉, candidate action이 local expert action mode와 가까울수록 buffer-side AC가 높다.

---

## 2.3 Merge AC

최종 AC는 model-side AC와 buffer-side AC의 geometric mean으로 결합한다.

$$
AC(\xi_m^j)
=
\left(
AC_{\mathrm{model}}(\xi_m^j)
\right)^\lambda
\left(
AC_{\mathrm{buffer}}(\xi_m^j)
\right)^{1-\lambda}
$$

기본값은:

$$
\lambda=0.5
$$

즉:

$$
AC(\xi_m^j)
=
\sqrt{
AC_{\mathrm{model}}(\xi_m^j)
AC_{\mathrm{buffer}}(\xi_m^j)
}
$$

---

# 3. Final Selection

최종 score는 다음과 같다.

$$
Score(\xi_m^j)
=
IG(\xi_m^j)\cdot AC(\xi_m^j)
$$

각 skill step에서는 반드시 하나의 trajectory를 선택해 실행해야 하므로, hard threshold로 candidate를 reject하지 않는다. 대신 모든 candidate를 $IG(\xi)\cdot AC(\xi)$ 기준으로 ranking하고, 가장 높은 score를 갖는 trajectory를 선택한다.

$$
\xi_m^*
=
\arg\max_{\xi_m^j\in\Xi_m}
IG(\xi_m^j)\cdot AC(\xi_m^j)
$$

선택된 trajectory를 실행하고, 실행된 skill segment를 buffer에 추가한다.

$$
\mathrm{Execute}(\xi_m^*)
$$

$$
B_{t+1}=B_t\cup\{\xi_m^*\}
$$

---

# 4. Interpretation

$$
IG \text{ low}
\Rightarrow
\text{Redundant Diversity}
$$

$$
IG \text{ high},\ AC \text{ low}
\Rightarrow
\text{Ambiguous Novelty}
$$

$$
IG \text{ high},\ AC \text{ high}
\Rightarrow
\text{Useful Diversity}
$$

---

# One-line Summary

**IG는 후보가 기학습 VLA와 미학습 buffer 기준으로 새로운지를 측정하고, AC는 그 후보가 기학습 VLA와 미학습 buffer의 action mode 기준으로 일관적인지를 측정한다. 두 점수를 각각 geometric mean으로 통합한 뒤, 최종적으로 $IG \times AC$가 가장 큰 skill trajectory를 선택한다.**
