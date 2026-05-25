# Paper 추가 권고 내용 — SmolVLA Fine-tuning · Candidate DCT · Marginal MI

> Reference: [`final_method3_paper.md`](./final_method3_paper.md)
> Related audit: [`paper_implementation_gaps.md`](./paper_implementation_gaps.md)
> Date: 2026-05-24
> Purpose: paper 에 누락된 핵심 3가지 (VLA 학습 · candidate DCT 변환 · MI 근사 유도) 를 paper-ready 형식으로 정리

---

## 📑 Contents

- [A. §3.4 안에 신설: VLA Fine-tuning on Skill-wise DCT Targets](#a-34-안에-신설-vla-fine-tuning-on-skill-wise-dct-targets)
- [B. §3.4 perturbation 단락에 추가: Candidate Trajectory → DCT Representation](#b-34-perturbation-단락에-추가-candidate-trajectory--dct-representation)
- [C. §3.1 끝부분에 보강: MI → Marginal MI Approximation (유도 명시)](#c-31-끝부분에-보강-mi--marginal-mi-approximation-유도-명시)
- [배치 위치 권고](#배치-위치-권고)
- [추가로 권장되는 그림 1장](#추가로-권장되는-그림-1장)

---

## A. §3.4 안에 신설: **VLA Fine-tuning on Skill-wise DCT Targets**

### (1) Skill-wise trajectory dataset 구성

Phase1 의 성공 episode 로부터 각 skill 호출 단위로 (start state → goal subgoal) trajectory 를 분리하여 skill-wise dataset $\mathcal{D}_{skill}$ 를 구성한다.

- 한 sample = 한 skill segment
- **입력**: segment 시작 frame 의 $(o_\tau,\; I,\; m)$ — observation · instruction · skill type
- **정답**: 그 segment 의 action sequence

### (2) DCT_50 target

가변 길이 action sequence 를 고정 길이 표현으로 만들기 위해 각 segment 의 action trajectory (shape (T, 6)) 를 **DCT_50 coefficient (shape (6, 50))** 로 변환한다 — low-frequency 50 coefficient 만 유지하는 truncated DCT.

$$z^a_\xi = \text{DCT}_{L_0=50}(A_\xi) \;\in\; \mathbb{R}^{6 \times 50}$$

세 가지 효과를 동시 달성:

1. **Skill-unit 단위 sample 화** — H-window sliding chunk 가 아닌 skill segment 단위
2. **가변 길이 → fixed shape** 통일
3. **Low-frequency 보존 → noise 제거**

### (3) Fine-tuning objective

Pretrained SmolVLA 를 base 로 하여, target 을 raw action chunk 대신 위 DCT_50 으로 교체한 single-step denoising loss 로 fine-tune. Fine-tuned policy 를 $\pi_\theta^{(1)}$ 로 표기.

$$\pi_\theta^{(1)}:\; (o_\tau, I, m) \;\longrightarrow\; \hat{z}^a$$

---

## B. §3.4 perturbation 단락에 추가: **Candidate Trajectory → DCT Representation**

### (1) 차원 정합

Curobo 가 생성한 K-batch candidate trajectory $\{\xi_{g,k}\}_{k=1}^K$ 는 joint-space waypoint sequence 이다. U_VLA 계산을 위해 candidate trajectory 도 **VLA 학습 target 과 동일한 (6, 50) DCT 공간** 으로 사영한다.

$$z^a_{\xi_k} = \text{DCT}_{L_0=50}(A_{\xi_k}) \;\in\; \mathbb{R}^{6 \times 50}$$

이로써 VLA 의 출력 공간과 candidate 표현 공간이 **shape · semantic 모두 일치** 하여 single-step denoising loss 계산이 well-defined 해진다.

### (2) Single-step denoising loss

$\pi_\theta^{(1)}$ 가 학습한 DCT space 에서, candidate 의 DCT representation $z^a_{\xi_k}$ 에 대해 single-step denoising loss 를 평가:

$$U_{VLA}(\xi_k) = L_{\text{denoise}}\bigl(o_\tau,\; I,\; m,\; z^a_{\xi_k};\; \pi_\theta^{(1)}\bigr)$$

Paper §3.4 식 (188) 의 R-stochastic 평균은 DCT mode 에서 **R=1 + optional deterministic time** $\sigma$ 로 축약된다 (computational efficiency).

---

## C. §3.1 끝부분에 보강: **MI → Marginal MI Approximation (유도 명시)**

### (1) Direct MI 의 한계

$$I(S; A) = H(A) - H(A|S)$$

는 완성된 dataset 에서 산정되며, online acquisition 의 **후보 채택 결정에는 직접 적용 불가**.

### (2) Marginal gain 으로 reformulation

후보 $\xi$ 가 buffer 에 추가됐을 때의 변화량 (marginal gain) 으로 reformulate:

$$\text{Gain}_{MI}(D_t, \xi) = I(D_t \cup \{\xi\}) - I(D_t)$$

Entropy decomposition 으로 분해하면:

$$\text{Gain}_{MI}(D_t, \xi) = \underbrace{[H(A_{D_t \cup \xi}) - H(A_{D_t})]}_{\Delta H_A} \;-\; \underbrace{[H(A|S)_{D_t \cup \xi} - H(A|S)_{D_t}]}_{\Delta H_{A|S}}$$

즉 **좋은 후보 = action coverage 확장 ($\Delta H_A \uparrow$) ∧ local ambiguity 비증가 ($\Delta H_{A|S} \downarrow$)**.

### (3) Descriptor-space proxy

Continuous 고차원 trajectory 에서 entropy 직접 추정은 불가능하므로, descriptor space 의 두 proxy 로 근사:

| 추정 대상 | Proxy | 구체 정의 |
|----------|-------|----------|
| $\widehat{\Delta H}_A$ | action descriptor novelty | candidate **EE delta DCT** 가 buffer 의 NN 과 멀수록 ↑ |
| $\widehat{\Delta H}_{A\|S}$ | local action support expansion | covered state-neighbor 안에서 $\log(d_{\min}^a / s_a)$, §9.3 |

최종 score:

$$M_{MI}(D_t, \xi) = \beta\, \widehat{\Delta H}_A(D_t, \xi) \;-\; \lambda\, \widehat{\Delta H}_{A|S}(D_t, \xi)$$

---

## 배치 위치 권고

| 추가 내용 | 권고 위치 | 분량 |
|----------|----------|------|
| **A. SmolVLA Fine-tuning** | §3.4 의 candidate scoring 단락 직전 (Phase2 evaluator 정의 단계) | ~ 1/3 page |
| **B. Candidate → DCT** | §3.4 의 perturbation 단락 직후 (U_VLA 식 (188) 직전) | ~ 1/4 page |
| **C. MI → Marginal MI** | §3.1 의 marginal MI gain 식 (74-80) 자리에 유도 보강 | ~ 1/3 page |

---

## 추가로 권장되는 그림 1장

### Figure: Skill-wise DCT pipeline

```text
[Phase1 success episodes]
        ↓ (skill segmentation)
[Skill-wise traj dataset] ──→ SmolVLA fine-tune (DCT_50 target)
                                       ↓
                                  π_θ^(1)
                                       ↑
[Curobo candidate traj] ──(DCT_50)──→ U_VLA = L_denoise(z^a_ξ; π_θ^(1))
```

이 그림 하나로 **§A + §B 의 연결**이 한눈에 보임:

- VLA 가 무엇을 예측하는지 (DCT_50 target)
- Candidate 가 어떻게 동일 공간으로 사영되는지 (DCT_50 변환)
- U_VLA 가 어디서 계산되는지 (π_θ^(1) 안에서 denoise loss)

→ Reviewer 가 method 의 핵심 mechanism 을 식 없이도 직관적으로 파악 가능.

---

## 변경 로그

| 날짜 | 변경 |
|------|------|
| 2026-05-24 | 최초 작성 — §A/§B/§C 추가 권고 + Figure 제안 |
