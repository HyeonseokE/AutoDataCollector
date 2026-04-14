# Action Prediction Debug Log

정책 추론 결과의 EE 지그재그 현상 가설 추적.

- 관측: `chunks_20260412_155741.npz` (15 chunks × 50 steps, RTC on, horizon=20, policy=`skkuprism/pi05_test_pick_place_10K`)
- 현상: 청크 내부 방향 반전 비율 66.1%, 평균 per-step EE Δ 4.52mm, joint Δ 0.513°

| 가설 | 상태 |
|---|---|
| H1. demo 데이터 자체가 떨림 | **기각** |
| H3. 출력 양자화·노이즈 | **부분 확인** (heavy LPF에서만 demo 수준 회복) |
| H6. OOD (CaP 분포 협소) | **기각 (주원인 아님)** — in-dist 에서도 zigzag 지속 |
| H2. Stochastic sampling (flow-matching) | **_다음 대상_** — 두 모델 공통 증상으로 유력 |
| H4. RTC inpainting artifact | 대기 |
| H5. 수렴 부족 / 과적합 | 대기 |

---

## H1. demo 데이터 자체가 떨림

**[1] 가설**: 학습 demo가 teleop 손떨림을 포함하고 있어 모델이 그대로 재현.

**[2] 예측**: demo episode의 action trajectory에서 추론과 유사한 방향 반전율(≳50%)이 관측됨.

**[3] 결과**

| 지표 | DEMO ep1 (`skkuprism/test_pick_red_place_blue_50epi`) | INFERENCE |
|---|---|---|
| 방향 반전 비율 | 2.6% | 66.1% |
| 평균 per-step EE Δ | 1.92 mm | 4.52 mm |
| 평균 per-step joint Δ | 0.198° | 0.513° |

**[4] 결론**: 기각. Demo는 매우 부드러움. 지그재그는 모델 출력 단에서 발생.

---

## H3. 출력 양자화·노이즈

**[1] 가설**: 모델이 올바른 저주파 궤적을 만들지만, 출력 정규화·dequantize 또는 flow sampling 과정에서 고주파 노이즈가 얹혀 지그재그로 보임.

**[2] 예측**: 기존 추론 chunk에 EMA / Savitzky–Golay 저역통과 필터를 걸었을 때 방향 반전율이 demo 수준(~3%)에 근접하면 고주파 노이즈 → H3 유력. α=0.6까지 밀어도 반전율 >30% 유지면 구조적 policy 오류 → H3 기각.

**[3] 결과** (`filter_analysis.py`, 2026-04-13)

| config | reversal | EE Δ (mm) | joint Δ (°) |
|---|---|---|---|
| DEMO baseline | **2.55%** | 1.924 | 0.198 |
| INFER raw | 66.11% | 4.517 | 0.513 |
| INFER EMA α=0.6 | 44.58% | 2.678 | 0.309 |
| INFER EMA α=0.4 | 31.67% | 2.003 | 0.236 |
| INFER EMA α=0.2 | 17.92% | 1.407 | 0.172 |
| INFER SG win=5 order=2 | 28.61% | 2.627 | 0.299 |
| INFER SG win=9 order=2 | 16.39% | 1.817 | 0.217 |
| **INFER SG win=15 order=3** | **3.61%** | 1.474 | 0.180 |

**[4] 결론**: 부분 확인. SG window=15 (≈500ms 평활화) 로만 demo 수준(2.55%)에 도달. 가벼운 필터(EMA α=0.6)로는 반전율이 절반도 안 줄고, EMA α=0.2 는 joint Δ를 demo 이하(0.172°)로 깎아내리는데도 반전율은 여전히 17.92%. 즉 지그재그는 "부드러운 궤적 + 고주파 가우시안 노이즈" 구조가 아니라 **~2–3 Hz 대역의 구조화된 진동**. 순수 dequantize 노이즈는 기각, sampling(H2)/RTC(H4) 유래 진동이 유력. 단기 완화책으로 추론 파이프라인에 SG(win=9, order=2) 이상의 후처리 필터 투입은 정당화됨.

---

## H6. OOD (CaP 분포 협소)

**[1] 가설**: CaP 기반 자동 수집 데이터는 reset/waypoint/속도가 스크립트 결정이라 분포가 매우 좁고, 실제 배포 obs(카메라 노이즈·조명·물체 위치 편차)가 학습 분포 밖이라 velocity field calibration이 깨져 지그재그가 발생.

**[2] 예측**: 학습 데이터셋 episode 의 in-distribution obs 로 오프라인 추론 시 reversal이 demo 수준(2–5%)에 근접하면 확정.

**[3] 결과** (2026-04-13, `offline_policy_eval.py`; pi0.5 대신 GPU 메모리 제약으로 SmolVLA 사용)

| | reversal | EE Δ | joint Δ |
|---|---|---|---|
| DEMO baseline | 2.55% | 1.92mm | 0.198° |
| INFER pi0.5 10K (real deploy, RTC on) | 66.11% | 4.52mm | 0.513° |
| **OFFLINE SmolVLA (in-dist, RTC off)** | **45.97%** | 2.85mm | 0.320° |

**[4] 결론**: 기각 (주원인 아님). in-distribution obs에서도 zigzag가 46% 로 demo(2.55%) 대비 **18배 높음**. OOD는 악화 요인일 수 있으나 근본 원인은 모델/추론 내부에 있음. SmolVLA·Pi0.5 **두 flow-matching 모델이 공통**으로 zigzag 하는 점은 H2(샘플링 노이즈) 쪽을 가리킴.

---

## H2. Stochastic sampling (flow-matching noise integration)

**[1] 가설**: Pi0.5/SmolVLA 모두 flow-matching 계열 → 추론 시 Gaussian noise `x₀`로부터 ODE 적분. noise 시드가 고정이 아니면 step별·chunk별로 다른 샘플을 생성하며, 모델이 덜 수렴됐거나 solver step이 적으면 개별 샘플에 고주파/구조적 진동이 남음.

**[2] 예측**: 같은 obs로 N개 다른 시드로 추론 시 (a) 개별 샘플은 zigzag (b) N개 평균 궤적은 부드러움 → H2 확정. 모든 시드가 동일 결과 = noise 고정 상태면 성립 불가.

**[3] 결과**: _미수행_

**[4] 결론**: _미수행_

---

## H4. RTC inpainting artifact

**[1] 가설**: RTC의 chunk overlap/inpainting이 경계에서 이전 청크 tail과 새 청크 head를 blending하며 smoothness를 깸.

**[2] 예측**: `RTC_ENABLED=false` 시 반전율이 크게 감소. 또는 반전 위치가 청크 경계 근처에 몰려있음.

**[3] 결과**: _미수행_

**[4] 결론**: _미수행_

---

## H5. 수렴 부족 / 과적합

**[1] 가설**: 학습 loss가 plateau에 도달하지 못했거나 train-val gap이 큼.

**[2] 예측**: W&B/TB 로그의 val loss curve가 아직 하강 중이거나 train-val gap이 확연함.

**[3] 결과**: _미수행_

**[4] 결론**: _미수행_
