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

- **(1-1) Skill-level perturbation** — 개별 스킬이 실행되면서 생성하는 trajectory(waypoint sequence)에 확률적 변형을 주입. 같은 스킬이라도 매번 다른 경로를 탐. 구체 구현체는 (2).
- **(1-2) Subgoal-level perturbation** — 에피소드 전체에서 move 계열 함수의 target position(서브골)을 변형. 단, **물체/환경과 직접 interaction이 있는 구간(grasp, place, press 등)은 건드리지 않고**, 이동(approach, retract, transit 등) 구간의 목표 위치만 변형. 구체 구현체는 (3).

두 layer는 직교(orthogonal)하게 동작하므로 동시에 켤 수 있다.

---

## (2) Skill-level perturbation: Planner ensemble *(이름 미정, TODO)*

다양한 low-level planner의 앙상블로 skill 단위 경로 패턴을 랜덤 생성한다.

mplib은 OMPL 래퍼라 RRT*/RRT-Connect/PRM/BIT*/KPIECE 등이 한 바이너리에 들어있고, 알고리즘별로 *질적으로 다른* 경로를 만들어낸다 (RRT-Connect = greedy 직선형, PRM = roadmap 품질형, BIT* = anytime-optimal). 단순 seed 변경이 한 알고리즘의 분산 안에서만 다양성을 주는 데 비해, 앙상블은 **모드(mode) 자체**를 바꿔 분포의 폭을 넓힌다.

(상세 파라미터/주입 흐름/파일 목록은 구현 시 추가)

---

## (3) Subgoal-level perturbation: 3D Gaussian as subgoal *(이름 미정, DONE)*

Subgoal point를 3D Gaussian blob으로 표현하여 확률적으로 스킬의 목표위치를 변형시킨다. 전체 traj의 위상학적 다양성을 제공.

### 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 분포 | Isotropic 3D Gaussian | `N(0, sigma^2 * I)` |
| sigma | 0.05m (5cm) | 표준편차 |
| clipping | 2 * sigma = 0.1m | Truncated Gaussian (reject & resample) |
| 적용 확률 | 100% | 모든 transit subgoal에 적용 |

### Skill type 분류

| skill_type | 분류 | perturbation |
|------------|------|--------------|
| `move_to_pose` | transit | O |
| `move_by_displacement` | transit | O |
| `grasp` | interaction | X |
| `place` | interaction | X |
| `open_gripper` | gripper | X (position target 없음) |
| `close_gripper` | gripper | X (position target 없음) |

### 주입 흐름

```
run_closed_loop.py (subtask loop)
  ├── subtask.skill_type 확인
  ├── transit이면 → SubgoalPerturbation.sample() → 3D offset
  ├── task_env._subgoal_offset = offset
  ├── execute_snippet() → codegen 코드 실행
  │   └── self.move_by_displacement() → Action(target_pose)
  │       └── self.move() 내부에서 offset 적용:
  │           action.target_pose[:3] += offset
  │           └── left_move_to_pose(perturbed_pose) → IK → trajectory
  └── task_env._subgoal_offset = None (클리어)
```

### 파일 목록

| 경로 | 상태 | 목적 |
|------|------|------|
| `closed_loop_cap/perturbation/__init__.py` | 신규 | |
| `closed_loop_cap/perturbation/subgoal.py` | 신규 | SubgoalPerturbation 클래스 + truncated Gaussian 샘플러 |
| `envs/_base_task.py` | 변경 | `_subgoal_offset` 속성 + `move()` 내 offset 적용 |
| `closed_loop_cap/run_closed_loop.py` | 변경 | subtask별 perturbation 샘플 + 설정/해제 |
| `closed_loop_cap/configs/recording_config.yaml` | 변경 | `perturbation.subgoal` 섹션 추가 |

### 설정 (`configs/recording_config.yaml`)

```yaml
perturbation:
  subgoal:
    enabled: false        # true → transit subgoal 위치 변형 활성화
    sigma: 0.05           # metres; isotropic Gaussian 표준편차
    clip_factor: 2.0      # truncate at clip_factor * sigma (최대 offset = 0.1m)
```

### RNG 시딩

에피소드의 `seed` 값으로 `np.random.default_rng(seed)` 생성. 동일 seed + 동일 config → 동일 perturbation offset (재현 가능). 다른 trial에서는 다른 offset이 나오도록 trial별로 별도 RNG를 쓰거나, seed에 trial을 mixing하는 방식으로 확장 가능.

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
│   (2) Planner ensemble    [skill-level]                     │
│   (3) 3D Gaussian subgoal [subgoal-level]                   │
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
closed_loop_cap/curate/
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
