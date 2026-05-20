# Stochastic Trajectory Perturbation for CaP Dataset Diversity

---

## 문제

CaP의 스킬들은 사전정의된 deterministic 함수들이라, 동일 태스크에 대해 시드(초기 물체 배치 등)만 달라질 뿐 **로봇이 그리는 경로 자체는 거의 동일**하다. 결과적으로:

- 수집된 데이터셋의 state-action 분포가 매우 좁음
- 학습된 BC/VLA 모델이 test-time에 조금만 벗어나도 OOD
- 한 번 벗어나면 학습 분포로 돌아오는 recovery 데이터가 없어서 compounding error로 실패

핵심은 **다양성(diversity) 부족** + **recovery trajectory 부재**.

---

## 기대 효과

- 동일 태스크/시드에서도 다양한 형상의 trajectory가 생성됨
- 데이터 분포가 넓어져서 모델이 test-time 편차에 robust해짐
- "살짝 벗어난 상태 → 정상 경로로 복귀" 패턴이 자연스럽게 포함되어 recovery 능력 학습 가능

---

## (1) 2-layer hierarchy randomization

전체 문제를 두 층으로 분리해서 다룬다. 각각 다른 granularity의 다양성을 담당.
**핵심 구분 축**은 *어디로* 갈 것인가 vs *어떻게* 갈 것인가:

- **(1-1) Skill-level perturbation** — *"어떻게 갈지"* 에 대한 교란. 개별 스킬이 실행되면서 생성하는 trajectory(waypoint sequence) 자체에 확률적 변형을 주입. 같은 스킬, 같은 시작·목표라도 매번 다른 경로(다른 planner 모드 + 다른 seed)를 탐. 구체 구현체는 (2).
- **(1-2) Subgoal-level perturbation** — *"어디로 갈지"* 에 대한 교란. 에피소드 전체에서 move 계열 함수의 target position(서브골) 자체를 변형. 단, **물체/환경과 직접 interaction이 있는 구간(grasp, place, press 등)은 건드리지 않고**, 이동(approach, retract, transit 등) 구간의 목표 위치만 변형. 구체 구현체는 (3).

두 layer는 직교(orthogonal)하게 동작하므로 동시에 켤 수 있다 — *어디로 갈지* 를 흔든 위에 *어떻게 갈지* 를 또 흔든다.

### 공통 게이팅: `is_transit`

두 layer 모두 같은 단일 진실원천(single source of truth)인 `move_to_position(is_transit=...)` 인자를 통해 활성화된다. `skill_type="move"`는 너무 거칠어서 (pure transit + `execute_pick_object`/`execute_place_object` 내부 하강을 모두 포함) 분리가 안 되므로, 각 `move_to_position` 호출자가 자신의 의도를 명시적으로 선언한다.

- `is_transit=True` (기본값) — 두 perturbation 모두 활성 가능
- `is_transit=False` — `execute_pick_object` / `execute_place_object`의 contact descent 등, 정확한 좌표가 필요한 구간에서 명시적으로 끔

또한 `LeRobotSkills.perturbation_disabled()` 컨텍스트 매니저는 reset 실행 중에 perturbation을 일시 비활성화(forward에는 유지). `execution_forward_and_reset.py`가 사용.

---

## (2) Skill-level perturbation: curobo via-point planner *(DONE)*

GPU-accelerated curobo motion planner로 skill 단위 transit 경로 패턴을
랜덤 생성한다. 직선 baseline + via-point sampling 으로 path topology
다양성을 만든다. 자세한 설계는 [`CUROBO_METHODOLOGY.md`](CUROBO_METHODOLOGY.md)
참고.

### 아키텍처 요약

`curobo` 는 in-process GPU planner 라 별도 daemon / IPC 가 필요 없다.
batched plan_cspace 호출로 N candidate 를 1 회 GPU pass 에서 동시 풀고,
CUDA graph 캡쳐로 두 번째 호출부터 50ms → 5ms 수준으로 떨어진다.

```
┌─ skills_lerobot (lerobot env, numpy 2.x) ────────────────────┐
│   move_to_position(is_transit=True):                         │
│     cands = client.plan_batch(start, goal, n, seed=…)        │
│     chosen = cands[rng.integers(0, len(cands))]              │
│     trajectory.joint_positions = chosen.waypoints            │
│     (constant-velocity time parameterization)                │
└────────────────┬─────────────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
  CuroboBackend       GrpcPlannerClient → grpc_server.server (H100)
  (in-process GPU)     (remote planning + IG·AC selection)
```

두 transport 모두 `plan_batch(start, goal, n, seed)` 인터페이스를 노출하므로
`skills.set_skill_planner_client(client)` 는 transport 와 무관하게 동일.

### Candidate 다양성 (via-point sampling)

| Slot | 설명 |
|------|------|
| `slot 0` | direct — start→goal 한 segment (baseline) |
| `slot 1..n-1` | via — start → v_1 → … → v_K → goal, `K ∈ {1, …, K_max}` random |

via-point 위치는 직선에서 `curobo_via_offset_mag` (기본 0.10m) 거리만큼
랜덤 방향으로 떨어진 점에서 IK 로 풀어낸다.

### Fixed-joint planning

SO-101 5-DoF 중 `wrist_roll`(joint index 4) 을 plan 도중 고정. cartesian-IK
경로의 `maintain_wrist_roll=True` 와 parity 를 맞춰 transit 중 gripper roll
wiggle 을 제거. `curobo_backend.fixed_joint_indices=(4,)` 로 설정.

### Time parameterization

candidate 별 waypoint 개수가 K (via 개수) 에 따라 달라지므로 기본 trajopt
time-parameterize 가 duration ∝ waypoint 수로 편향될 수 있다. 대신 **상수
관절 속도** (`SKILL_PERT_VELOCITY_FACTOR × planner.max_velocity`) + EE
Cartesian cap (≈ 0.15 m/s) 으로 직접 time-parameterize 하여 duration 이
path length 에만 비례하도록 정규화한다.

### Collision world

curobo robot config (`robot_configs/curobo/so101_robot*.yml`) 안에 table
plane / arm self-collision spheres 가 들어 있다. dynamic scene objects 는
`skills.update_planner_obstacles()` 로 detect 결과를 push 하면 backend 가
자체 collision world 에 box 로 반영한다.

### Fallback

다음 중 어떤 조건이든 cartesian-line trajectory 로 폴백:
- `is_transit=False`
- 클라이언트/RNG 미연결
- `plan_batch` 예외
- 빈 후보 배치

### 파라미터 / 설정 (`recording_config_*.yaml`)

```yaml
perturbation:
  skill:
    enabled_forward: true                 # forward execution 에서 활성화
    enabled_reset: false                  # reset 은 직선만 (기본)
    n_candidates: 128
    curobo_robot_cfg_path: robot_configs/curobo/so101_robot0.yml
    curobo_num_trajopt_seeds: 4
    curobo_num_ik_seeds: 16
    curobo_use_cuda_graph: true
    curobo_via_offset_mag: 0.10
    curobo_junction_smooth_k: 5
    curobo_max_vias_per_candidate: 1      # K_max
    fixed_joint_indices: [4]              # SO-101 wrist_roll 고정
    arm_joint_count: 5
```

### 주입 흐름

```
execution_forward_and_reset._setup_skill_perturbation_on_skills()
  ├── recording_config의 perturbation.skill 읽기
  ├── transport=local  → CuroboBackend(urdf, cfg)
  │   transport=grpc   → GrpcPlannerClient(PreselectiveClient(addr), provider)
  └── skills.set_skill_planner_client(client, n_candidates=N)

[per-episode]
_seed_episode_perturbation(batch_index, slot_in_batch)
  └── skills.set_perturbation_rng(seed)  # subgoal/skill 공통 RNG

[per move_to_position(is_transit=True)]
skills_lerobot.move_to_position()
  ├── cartesian-line IK plan → trajectory (goal_joint_rad 추출용)
  ├── if is_transit AND client AND rng AND ik_converged:
  │     seed = rng.integers(0, 2**31-1)
  │     cands = client.plan_batch(current_joints, goal_joint_rad, n=N, seed)
  │     if cands:
  │       chosen = cands[rng.integers(0, len(cands))]
  │       trajectory.joint_positions = chosen.waypoints
  │       (상수-속도 time parameterization 재적용)
  │     else: fallback to cartesian
  └── execute trajectory

[teardown]
_teardown_skill_perturbation()
  └── client.close()  # CUDA graph 해제 + cache empty
```

### 파일 목록

| 경로 | 목적 |
|------|------|
| `perturbation/skill_level/__init__.py` | `TrajectoryCandidate`, `get_curobo_backend()` 노출 |
| `perturbation/skill_level/planner.py` | `TrajectoryCandidate` dataclass (wire shape) |
| `perturbation/skill_level/curobo_backend.py` | `CuroboBackend` + `CuroboBackendConfig` |
| `vla_adaptor/grpc_planner_adapter.py` | `GrpcPlannerClient` — gRPC 모드 클라이언트 |
| `grpc_server/server.py` | H100 측 plan+select gRPC 서버 |
| `skills/skills_lerobot.py` | `set_skill_planner_client` + `move_to_position` candidate 선택 / 시간 재파라미터화 |
| `execution_forward_and_reset.py` | `_setup_skill_perturbation_on_skills` / `_teardown_skill_perturbation` |
| `pipeline_config/recording_config_ws*.yaml` | `perturbation.skill` 섹션 |

---

## (3) Subgoal-level perturbation: 3D Gaussian as subgoal *(DONE)*

Subgoal point를 3D Gaussian blob으로 표현하여 확률적으로 스킬의 목표위치를 변형시킨다. 전체 traj의 위상학적 다양성을 제공.

### 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 분포 | Isotropic 3D Gaussian | `N(0, sigma^2 * I)` |
| sigma | 0.05m (5cm) | 표준편차 |
| clipping | 2 * sigma = 0.1m | Truncated Gaussian (reject & resample, 최대 100회) |
| 적용 확률 | 100% | `is_transit=True`인 모든 호출에 적용 |
| Reachability fallback | 워크스페이스 밖이면 offset drop | 무음 실패 방지 |

### 게이팅: `is_transit` 기반 (skill_type X)

문서 초기 버전은 skill_type 기반 화이트리스트(`move_to_pose`, `move_by_displacement`)를 가정했지만, 실제 구현은 `move_to_position`의 `is_transit` 인자로 게이팅한다. 이유: 같은 `skill_type="move"`라도 pure transit 호출과 `execute_pick_object`/`execute_place_object` 내부 하강 호출이 섞여 있어 type만으로 분리가 안 됨.

| 호출 컨텍스트 | `is_transit` | perturbation |
|---|---|---|
| 일반 `move_to_position` 호출 (approach, retract, transit) | `True` (default) | O |
| `execute_pick_object` 내부 descent | `False` | X |
| `execute_place_object` 내부 descent | `False` | X |
| `open_gripper` / `close_gripper` | — (position target 없음) | X |
| `grasp` / `place` action 자체 | — | X |

### 주입 흐름

```
execution_forward_and_reset._setup_perturbation_on_skills()
  ├── recording_config의 perturbation.subgoal 읽기
  ├── SubgoalPerturbation(SubgoalPerturbationConfig(...)) 생성
  └── skills.set_perturbation(pert)

[per-episode]
_seed_episode_perturbation(batch_index, slot_in_batch)
  └── skills.set_perturbation_rng(seed)

[per move_to_position(is_transit=True)]
skills_lerobot.move_to_position()
  ├── target_position = transform(position to base_link)
  ├── if is_transit AND _perturbation AND _perturbation_rng:
  │     offset = perturbation.sample(rng=_perturbation_rng)  # 3D xyz
  │     candidate = target_position + offset
  │     if kinematics.is_position_reachable(candidate):
  │       target_position = candidate
  │     else: drop offset, log fallback
  └── proceed with IK + trajectory plan (스킬-level (2)가 켜져 있으면 그 후 curobo 후보로 교체)
```

`perturbation_disabled()` 컨텍스트 매니저로 reset 실행 시 일시 해제.

### 설정 (`recording_config.yaml`)

```yaml
perturbation:
  subgoal:
    enabled: false        # true → transit subgoal 위치 변형 활성화
    sigma: 0.05           # metres; isotropic Gaussian 표준편차
    clip_factor: 2.0      # truncate at clip_factor * sigma (최대 offset = 0.1m)
```

### RNG 시딩

에피소드의 `seed` 값으로 `np.random.default_rng(seed)` 생성. 동일 seed + 동일 config → 동일 perturbation offset (재현 가능). 다른 trial에서는 다른 offset이 나오도록 trial별로 별도 RNG를 쓰거나, seed에 trial을 mixing하는 방식으로 확장. 한 RNG가 (3) subgoal과 (2) skill-level curobo via-point seed 추첨 모두에 사용된다.

### 파일 목록

| 경로 | 상태 | 목적 |
|------|------|------|
| `perturbation/__init__.py` | 신규 | top-level export |
| `perturbation/subgoal_level/__init__.py` | 신규 | subgoal 서브패키지 export |
| `perturbation/subgoal_level/subgoal.py` | 신규 | `SubgoalPerturbation`, `SubgoalPerturbationConfig`, truncated Gaussian 샘플러 |
| `skills/skills_lerobot.py` | 변경 | `set_perturbation` / `set_perturbation_rng` / `perturbation_disabled()` + `move_to_position` 안 offset 적용 |
| `execution_forward_and_reset.py` | 변경 | `_setup_perturbation_on_skills` / `_seed_episode_perturbation` |
| `pipeline_config/recording_config_ws*.yaml` | 변경 | `perturbation.subgoal` 섹션 |

---

## (4) Selector *(TODO)*

(2)와 (3)이 만들어낸 candidate trajectory pool에서 **학습에 정말 도움되는 subset만 골라내는 큐레이션 단**.

### 목적

(1)의 perturbation 시스템은 candidate 풀의 폭(diversity)을 키우지만, "perturbation으로 만들어진 모든 trajectory를 그대로 학습 데이터에 넣기"는 두 가지 문제를 낳는다:

1. **Redundancy** — 비슷한 trajectory가 다수 적재되어 effective dataset size 대비 학습 신호가 희석된다.
2. **Quality 분산** — 일부 perturbed trajectory는 task 실패, 불필요한 detour, 학습에 해로운 패턴을 포함한다.

Selector는 candidate pool과 최종 dataset 사이에 끼워, **목표 데이터 정의 → 자동 선별 → rollout** 루프를 만든다. 즉:

- "어떤 데이터가 필요한가"를 score 함수로 정량화
- 후보 풀에서 그 기준에 맞는 부분집합만 채택
- 선택된 (seed, trial)만 LeRobot dataset에 적재

### 파이프라인 위치

```
┌─────────────────────────────────────────────────────────────┐
│ generator side  (1) 2-layer perturbation                    │
│                                                             │
│   (2) curobo via-point planner [skill-level]                │
│   (3) 3D Gaussian subgoal   [subgoal-level]                 │
│                                                             │
│   → candidate pool: logs/seed_*/trial_*/                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ (4) Selector                                                │
│                                                             │
│   scorer    (novelty / recovery / difficulty / coverage …)  │
│   policy    (top-K / Pareto / cluster-uniform)              │
│                                                             │
│   → 선택된 (seed, trial) → run_replay → recorded_data       │
└─────────────────────────────────────────────────────────────┘
```

### 점수(Score) 후보

| 기준 | 측정 방법 | 무엇을 잡아내나 |
|---|---|---|
| **Novelty** | 기존 dataset에 대한 (state, action) k-NN 평균 거리 | 분포의 빈 구역 채움 — OOD 감소 |
| **Recovery** | `subgoal_offset` 크기 × `success=True` | "벗어났다 돌아온" 궤적 — compounding error 학습 |
| **Difficulty** | judger 신호가 발화했지만 최종 ok | 분포 boundary — 정책의 한계 학습 |
| **Skill-coverage** | subtask transition 부근의 cluster diversity | 특정 스킬 전환만 과대표집되는 걸 방지 |
| **BC disagreement** | 작은 BC 모델이 가장 많이 틀리는 후보 | 정보이득 최대 — DAGGER식 active learning |

### 선택 정책 (Selection policy)

- **Top-K** — 단일 score로 K개
- **Pareto front** — 다목적 (예: novelty ↑ ∧ length ↓)
- **Cluster-uniform** — latent space 클러스터별로 균등 샘플링 → 한 모드에 쏠리지 않게

### 예상 위치

```
perturbation/curate/   # (또는 별도 top-level package)
  scorer.py     # NoveltyScorer / RecoveryScorer / UncertaintyScorer / ... (Protocol 기반, 합성 가능)
  selector.py   # TopK / ParetoFront / ClusterUniform
  cli.py        # logs 스캔 → score → select → 선택된 (seed,trial) 리스트 출력
                #  → run_replay --trials 이 리스트로 호출
```

또는 `run_collect_n_success`에 옵션 추가:

```bash
--candidates 50 --keep 10 --score "novelty+recovery" --select topk
```
50개 만들어서 점수 매겨 10개만 LeRobot에 적재.
