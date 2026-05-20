# Episodes-per-seed Migration

이 문서는 한 세션 안에서 **same seed × episodes_per_seed 확장** (수직 확장) 을
수행할 때 폴더 ↔ `(seed, slot)` 매핑이 깨지지 않도록 안전하게 마이그레이션하는
절차를 정리합니다. **Phase1 → Phase2 누적 확장** 도 동일 로직으로 다룹니다.

## 언제 필요한가

다음 조건에서 필요:

- 한 세션의 `num_random_seeds` 는 **유지**
- `num_episodes` 가 증가 → `episodes_per_seed = num_episodes // num_random_seeds`
  가 늘어남
- 기존에 수집된 episode 폴더와 LeRobot 데이터셋을 **버리지 않고** 이어 수집

전형적 시나리오:

1. **Phase1 30/10 → Phase1+Phase2 100/10**: phase1 에서 10 seed × 3 episode 모은
   후, phase2 에서 같은 10 seed 에 각 7 episode 씩 더 추가 → 총 10/seed.
2. **Phase1 10/10 → Phase1 40/10**: 같은 phase 안에서 seed 당 1 → 4 로 늘림.
3. 일반적 **same-seed 수직 확장** — seed 종류는 그대로, 한 seed 당 깊이만 증가.

다음 조건에는 적용 **불가**:

- `num_random_seeds` 자체가 바뀜 (새 seed 추가 또는 제거)
- `new_episodes < old_episodes` (축소)
- `new_episodes % num_random_seeds != 0`

이 경우 새 세션을 만들거나 별도 도구를 사용해야 합니다.

## 왜 필요한가 (한 줄 설명)

`execution_forward_and_reset.py` 는 `episode_idx // episodes_per_seed` 로
seed 인덱스를 결정합니다. `episodes_per_seed` 가 3 → 10 으로 바뀌면 같은
`(seed, slot)` 이 다른 `episode_NN` 폴더로 옮겨가야 하므로 **폴더 재명명** 이
필요합니다. 코드 한 줄 바꿔서 해결되지 않습니다.

## 마이그레이션 절차

### 1. dry-run

```bash
python scripts/migrate_session_episodes_per_seed.py \
    --session results/session_YYYYMMDD_HHMMSS \
    --old-episodes 30 \
    --new-episodes 100 \
    --num-seeds 10
```

출력 검토 사항:

- `src ✓` 가 모든 row 에 있어야 함 (없으면 해당 episode 가 디스크에 없음)
- `[ok] batch_info ↔ mapping consistent` 라인이 나와야 함
- `Empty slots after migration` 가 resume 시 채워질 자리 목록

### 2. apply

```bash
python scripts/migrate_session_episodes_per_seed.py \
    --session results/session_YYYYMMDD_HHMMSS \
    --old-episodes 30 \
    --new-episodes 100 \
    --num-seeds 10 \
    --apply
```

- 큰 번호부터 mv 하여 destination 충돌 방지
- `session_config.json` 의 `num_episodes` / `episodes_per_seed` 갱신

### 3. 실행 스크립트 변경

`run_forward_and_reset_wsN.sh` 의 `NUM_EPISODES` 를 새 값으로 갱신:

```bash
NUM_EPISODES=100
RESUME_SESSION="./results/session_YYYYMMDD_HHMMSS"
```

resume 흐름이 빈 슬롯만 골라서 채웁니다.

## 매핑 수식

```
old_eps_per_seed = old_episodes // num_seeds
new_eps_per_seed = new_episodes // num_seeds

seed_idx = (old_ep_num - 1) // old_eps_per_seed
slot     = (old_ep_num - 1) %  old_eps_per_seed
new_ep_num = seed_idx * new_eps_per_seed + slot + 1
```

예: 30/10 → 100/10 일 때

| old_ep | seed | slot | new_ep |
|--------|------|------|--------|
| 01..03 | 1 | 0..2 | 01..03 (no-op) |
| 04..06 | 2 | 0..2 | 11..13 |
| 07..09 | 3 | 0..2 | 21..23 |
| ... | ... | ... | ... |
| 28..30 | 10 | 0..2 | 91..93 |

빈 슬롯 (각 seed 의 slot 3..9) 가 resume 시 새로 채워집니다.

## 동치성 (왜 안전한가)

"30/10 → 마이그레이션 → +70" 결과는 "100/10 fresh 처음부터" 와 **수학적으로 동치**:

- **폴더 매핑**: 마이그레이션 후 sorted `episode_*` 순서가 처음부터 100/10 으로
  모은 sorted 순서와 일치.
- **Perturbation RNG**: `_seed_episode_perturbation` 가 `batch * 10000 + slot`
  으로 결정적이므로, 같은 `(seed, slot)` 이면 두 시나리오에서 같은 RNG.
- **LeRobot dataset 정합**: sorted 폴더 i 번째 ↔ dataset idx `i` 가 보장됨.

## 자동 보호장치

`execution_forward_and_reset.py` 의 `_verify_resume_layout` 가 resume 시
`session_config.json` 의 layout 과 현재 `(NUM_EPISODES, NUM_RANDOM_SEEDS)` 를
비교하여 불일치 시 멈춥니다. 수직 확장 케이스가 감지되면 마이그레이션 명령을
정확히 인쇄하므로 그대로 복사해서 실행하면 됩니다.

## Cleanup 의미 변화 (영향 범위)

`record_dataset/cleanup.py` 의 `ep_states` 가 `range(1, max+1)` 에서
`sorted(glob("episode_*"))` 로 바뀌었습니다. 부수효과:

- **이전**: `episode_NN` 폴더를 직접 삭제하면 `range` 가 그 번호를 `missing`
  으로 잡아 dataset idx 도 자동 삭제.
- **이후**: 폴더가 아예 없으면 ep_states 에서 빠짐. dataset 자동 청소 안 됨.

정상 워크플로우에서는 `batch_info.judge="FALSE"` 로 재취득을 표현하므로
영향이 거의 없습니다. 폴더 직접 삭제 + dataset 청소가 필요한 경우 별도 CLI
로 분리해 사용하세요.

## Phase1 → Phase2 시나리오 적용

세션이 phase1 (예: 30/10) 로 모인 후 phase2 단계에서 같은 세션·같은 10 seed 에
70 episode 를 더 모으려면:

1. **phase1 작업 종료** — 30 episode 모두 수집·검증 완료.
2. **마이그레이션 dry-run** 으로 폴더 재배치 계획 확인.
3. **`--apply`** 로 폴더 이동 + `session_config.json` 갱신.
4. **`run_forward_and_reset_wsN.sh`** 에서
   - `NUM_EPISODES=100`
   - `PHASE="phase2"`
   - `RESUME_SESSION="./results/session_..."`
   설정 후 실행.
5. 코드가 빈 슬롯 (ep04-10, ep14-20, ..., ep94-100) 만 phase2 모드로 채움.

`_verify_resume_layout` 가 마이그레이션을 깜빡한 경우 명령 hint 와 함께
멈추므로 안전합니다.
