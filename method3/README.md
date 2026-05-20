# method3 — Pre-selective Real-world Data Acquisition

Method3 의 구현 폴더. 명세: [`ours_method/final_method3_spec.md`](../ours_method/final_method3_spec.md)
(상세: [`ours_method/details/`](../ours_method/details/)).

목표: 현재 buffer 가 충분히 커버하지 못한 state/action region 을 확장하면서도,
유사 state 에서 action ambiguity 를 과도하게 키우지 않는 trajectory 를 선별 수집한다.

## 폴더 구조

```
method3/
├── phase1_state_seeding/     Phase1 — State Coverage Seeding (§4-5)
├── phase2_mi_selection/      Phase2 — MI-based Diversity Acquisition (§7-14)
├── storage/                  §3  — 데이터 저장 레이어 (raw trajectory dataset)
├── reembedding/              §6  — Phase1 raw → P_phase1 seed vector DB
├── phase_control/            §15 — phase saturation & transition
└── acquisition/              §2  — two-phase acquisition 오케스트레이터 (통합)
```

### phase1_state_seeding — Phase1 (§4-5)

subgoal diversity 로 state coverage `H(S)` 를 키우고 path 는 canonical 하게 유지.
skill-wise subgoal buffer `B_{g,t}^{(m)}` 가 이미 seed 한 terminal state region 을
추적하고, 후보 중 가장 덜 커버된 subgoal 을 argmax 로 고른다.

| 파일 | 명세 | 역할 |
|---|---|---|
| `subgoal_candidates.py`   | §4.2     | K개 subgoal 후보 `G` 생성 (uniform_ball/gaussian) |
| `subgoal_validity.py`     | §4.2     | `G_valid` 술어 (reachable/safe/AABB) |
| `canonical_preview.py`    | §5.4     | `InterpPlan` canonical preview + `T_end` 슬라이스 |
| `terminal_descriptor.py`  | §5.3     | `φ_goal` geometric descriptor `ĥ` |
| `subgoal_buffer.py`       | §5       | skill-wise subgoal buffer + §5.2 entry + 거리 |
| `subgoal_selector.py`     | §5.4     | `argmax G_S^goal` 선택기 (+ TRUE-only buffer update) |
| `legacy_blob.py`          | —        | ablation baseline (legacy 3D Gaussian blob) |

선택 흐름: K candidates → canonical preview → 마지막 20% `T_end` state novelty
vs `B_{g,t}^{(m)}` → `argmax G_S^goal`. TRUE episode 만 buffer 에 반영한다.

### phase2_mi_selection — Phase2 (§7-14)

Phase1 이 만든 state support 안에서, 후보 trajectory 를 MI-style score
`Q2 = β·ΔH_A − λ·ΔH_A|S` 로 평가하여 action diversity 는 키우되 유사 state 에서
action ambiguity 는 키우지 않는 trajectory 만 선별 수집한다.

| 파일 | 명세 | 역할 |
|---|---|---|
| `action_descriptor.py`     | §4.2    | DCT action descriptor `ψ → z^a` |
| `vector_db.py`             | §3/§7.2 | skill-wise vector DB `B_t^{(m)}` |
| `neighbor_search.py`       | §8/§9   | L2 kNN/meanNN/min 거리 primitive |
| `radius.py`                | §16     | state-neighborhood radius `ρ_m` |
| `action_coverage.py`       | §8      | action coverage gain `ΔH_A` |
| `conditional_ambiguity.py` | §9      | covered-state + conditional ambiguity `ΔH_A|S` |
| `mi_selector.py`           | §11-12  | `Q2` score + 배치 정규화 + accept + argmax |

선택 흐름: 후보 window 별 `z_τ^a`(DCT) → `ΔH_A`(top-q action novelty) +
`ΔH_A|S`(covered window 의 local action support expansion) → `Q2` → 배치 정규화
`Q̃2` → `Q̃2 > τ_Q̃` accept (§12 Option B).

### storage — 데이터 저장 레이어 (§3)

Method3 는 저장 구조를 두 layer 로 분리한다 (§3, §13 원칙 1): **raw trajectory
dataset** (source of truth) 과 **vector DB** (searchable index `B_t^{(m)}`).

| 파일 | 명세 | 역할 |
|---|---|---|
| `raw_dataset.py` | §3.1 | `RawTrajectoryDataset` — `D_phase{1,2}_raw` writer·reader (JSONL+npz), re-embedding 복원용 원본 정보 + pointer |

raw dataset 은 entry 마다 §3.1 필드(observation_ref / proprioception /
action_chunk / subgoal / metadata 등)를 즉시 영속화한다. observation 은 이미지라
무거우므로 §3 details §4 의 pointer 규약을 따라 `observation_ref` 만 저장한다.
vector DB 영속화는 `phase2_mi_selection/vector_db.py` 의 `SkillVectorDB.save/load`
가 담당한다 (Phase1 `SubgoalBuffer` 와 대칭 — 각 buffer 가 스스로 영속화).

### reembedding — Phase1 → Phase2 vector DB seed (§6)

Phase1 종료 후 Phase1 raw dataset 을 Phase1-trained·frozen VLA encoder 로
re-embedding 하여 Phase2 가 쓸 skill-wise vector DB seed `P_phase1^(m)` 를 만든다.

| 파일 | 명세 | 역할 |
|---|---|---|
| `vla_encoder.py`  | §6 | `VLAStateEncoder` Protocol + `MeanPoolStateEncoder` stub |
| `seed_builder.py` | §6 | `build_phase1_vector_db` — raw dataset → `P_phase1` |

`e_i = [φ_VLA^(1)(o_i,I_i); p_i]`, `z_i^a = ψ(A_i)`(DCT). §6 Step 1(VLA 학습)·
Step 2(freeze)는 외부 ML job — builder 는 학습·freeze 된 encoder 를 받아 Step
3-4(re-embedding + seed DB)만 수행한다. production encoder 는 Phase1-trained
VLA(또는 `preselective_filter/vectorDB/vla_embedding.py` 의 `VLAKeyExtractor`)를
`VLAStateEncoder` 로 감싼다.

### phase_control — phase saturation & transition (§15)

acquisition 을 fixed budget `B` 안에서 Phase1↔Phase2 로 운영하되, 경계는 각
phase 의 목적 달성 여부로 adaptive 하게 정한다.

| 파일 | 명세 | 역할 |
|---|---|---|
| `phase1_readiness.py` | §15.1 | Phase2-readiness `R_cov`/`R̄_cov^(m)`/`R_ready` |
| `phase2_saturation.py`| §15.2 | MI-style gain saturation `Q̄̃2^(W)` window tracker |
| `phase_controller.py` | §15.3 | fixed budget + adaptive transition 상태기계 |

전환 규칙: `(t ≥ B_{1,min} ∧ R_ready > τ_ready) ∨ (t ≥ B_{1,max})` → Phase2.
종료 규칙: `t = B ∨ Q̄̃2^(W) < 0` (early stop 은 공정비교 시 끌 수 있음).

### acquisition — two-phase acquisition 오케스트레이터 (§2)

위 다섯 모듈을 하나의 acquisition 루프로 통합한다. method3 로직(phase 전환·
budget·re-embedding 트리거·MI scoring)은 오케스트레이터가 직접 소유하고, 로봇
실행·VLA 모델·후보 생성은 `AcquisitionEnvironment` 포트로 분리한다.

| 파일 | 명세 | 역할 |
|---|---|---|
| `ports.py`         | §2 | `AcquisitionEnvironment` Protocol (hardware/model 경계) |
| `orchestrator.py`  | §2 | `Method3Acquisition` — Phase1→전환→re-embed→Phase2→종료 |
| `in_memory_env.py` | §2 | `InMemoryAcquisitionEnvironment` — 하드웨어 없는 reference 구현 |
| `demo_offline.py`  | —  | offline dry-run (`python -m method3.acquisition.demo_offline`) |

루프: Phase1 episode → D_phase1_raw 적재 → readiness 측정(§15.1) → 전환 시
re-embedding(§6) → Phase2 episode(후보 생성→MI select→accept→D_phase2_raw +
vector DB §14) → budget/saturation 종료(§15). 실제 시스템(예:
`execution_forward_and_reset`)이 `AcquisitionEnvironment` 를 구현해 주입한다.

**디버깅**: `Method3AcquisitionConfig(verbose=True)` 면 run 전 과정을
`[method3:acq]` 로그로 narration 한다 (phase 전환·readiness·re-embed·accept·
saturation). per-candidate MI 상세는 `phase2_mi.debug_verbose=True` 로 켠다.
하드웨어 없이 전체 흐름을 보려면 `python -m method3.acquisition.demo_offline`.

## 범위 메모

- Phase2 의 VLA state embedding `e_τ = [φ_VLA(o_τ,I); p_τ]` 추출은 이 라이브러리
  밖에서 수행한다 — `Phase2Candidate.state_keys` 에 이미 계산된 key 를 넣어 넘긴다
  (Phase1-trained encoder freeze / re-embedding 절차는 §6, §7 후속 작업).
- Phase 전환·종료 규칙(§15 saturation)과 RPC 파이프라인 연동은 후속 단계.
  `conditional_ambiguity` 는 `R_cov` (covered ratio) 를 리포트하여 §15.1 연결을
  쉽게 해 둔다.
- Phase2 legacy 구현 `preselective_filter/` (IG·AC 방식) 는 그대로 유지된다 —
  본 폴더의 `phase2_mi_selection/` 이 `final_method3_spec` §7-12 정합 구현이다.

## 설정 (config)

Method3 설정은 두 YAML 파일로 분리한다 (`recording_config` 와 같은 방식):

| 파일 | 대상 | 소비 주체 |
|---|---|---|
| `pipeline_config/phase1_config.yaml` | Phase1 subgoal seeding (§4-5) | recording 파이프라인이 자동 로드 |
| `pipeline_config/phase2_config.yaml` | Phase2 MI + 전환 + re-embed (§6-§16) | `method3.config.load_phase2_config` |

- **Phase1 실행**: `phase1_config.yaml` 을 편집하고 `./run_forward_and_reset_ws3.sh`
  를 돌리면 된다. 이 파일이 있으면 `recording_config_ws*.yaml` 의
  `perturbation.subgoal` 보다 우선한다 (실행 로그 `[Perturbation] Phase1 config ←`).
- **Phase2**: `phase2_config.yaml` → `load_phase2_config()` → `Method3Acquisition
  Config`. 하드웨어 배선 전이므로 `python -m method3.acquisition.demo_offline` 로
  offline 확인.

## 테스트

```bash
conda activate lerobot_cap
python -m pytest method3/ -q
```
