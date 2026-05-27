# Method3 Paradigm 일관화 — 마이그레이션 가이드

옛 paradigm 으로 준비된 Phase2 session 을 새 paradigm 에 맞춰 다시 사용하기 위한 절차.

---

## 무엇이 바뀌었나

| 영역 | 옛 paradigm | 새 paradigm |
|---|---|---|
| **Skill ordinal namespace** | `_transit_call_index` — transit move + move_initial 만 카운팅 | `RecordingContext._skill_call_index` — 모든 `set_skill_info` 호출 단위 카운팅 |
| **subgoal_buffer.npz 의 partition key** | transit-only ordinal (skill_0..n) — phase1 의 ~6 skill 만 entry | 모든 호출 단위 (skill_0..10) — phase1 의 11 skill 모두 entry |
| **VDB 의 state_retrieval_key** | `[vla_unit (norm=1) ; proprio_raw (norm 50~150)]` — proprio 가 distance dominate | `[vla_unit ; proprio_unit]` — 둘 다 L2 normalize, scale balance |
| **plan_batch 의 skill_id 매핑** | `f"skill_{_transit_call_index}"` — DB 의 skill_index 와 off-by-N | `f"skill_{RecordingContext._skill_call_index}"` — 매 호출의 *현재* ordinal |
| **Phase2SubgoalReplay cursor** | `self._cursor += 1` per select_subgoal 호출 (transit 만) | `RecordingContext._skill_call_index` 직접 lookup |
| **Grpc mode 의 client-side selector hook** | mode 무관 attach — server 의 chosen 1 cand 에 client 가 다시 selection (double-eval) | grpc mode 면 attach skip — server 가 selection 전체 책임 |
| **Dataset 보호** | `_init_dataset` 의 broken meta 시 destructive `rmtree` | `_has_collected_data` 가 있으면 fail-loud, 사용자 명시 승인 요구 |

→ 모든 paradigm fix 가 master 의 commit `86260c6` 까지 정리됨. 모든 branch 에 propagate.

---

## 데이터 영향 정리

### 영향 받는 데이터 (rebuild 필요)

| 데이터 | 영향 | rebuild 방법 |
|---|---|---|
| `<session>/subgoal_buffer.npz` | namespace 다름 (transit-only → 모든 호출) | `scripts/rebuild_subgoal_buffer_unified.py` |
| `<session>/dct/skill_wise_vector_db.npz` | state_key 의 proprio scale 다름 | cache 삭제 → next phase2 cycle 시 client 가 자동 rebuild |
| `<session>/skill_wise_vector_db.npz` (옛 cache) | 같음 | cache 삭제 |
| `grpc_server/buffer/server_skill_wise_vector_db.npz` (server 측 mirror) | 같음 | client VDB rebuild 시 자동 scp + server restart 로 reload |

### 영향 없는 데이터 (그대로 사용)

- Phase1 dataset (`~/.cache/huggingface/lerobot/<repo>/`)
- Session artifact (`results/<session>/{phase1,phase2}/episode_NN/`)
- `session_config.json`, `seed_NN_setup/`, `readiness_trajectory.jsonl`, `subgoal_phase1_trace.jsonl`
- DCT-tuned VLA ckpt
- Skill DCT parquet (`results/skill_dct/<task>.parquet`)

---

## 마이그레이션 절차 (옛 session 마다 반복)

### Step 0 — 환경 확인

```bash
git checkout master
git pull origin master   # 새 paradigm 가 master 에 있어야
git log --oneline -3
# ✓ master 의 head 가 paradigm fix commit (예: 86260c6 또는 이후)
```

### Step 1 — 옛 buffer/VDB backup (안전)

```bash
SESSION="./results/completed_logs/<your-task>"

cp "$SESSION/subgoal_buffer.npz" \
   "$SESSION/subgoal_buffer.transit_only_legacy.npz"

[ -f "$SESSION/dct/skill_wise_vector_db.npz" ] && \
  cp "$SESSION/dct/skill_wise_vector_db.npz" \
     "$SESSION/dct/skill_wise_vector_db.raw_proprio_legacy.npz"

[ -f "$SESSION/skill_wise_vector_db.npz" ] && \
  cp "$SESSION/skill_wise_vector_db.npz" \
     "$SESSION/skill_wise_vector_db.raw_proprio_legacy.npz"
```

### Step 2 — subgoal_buffer 재build (새 namespace)

Phase1 dataset 의 frame-level data (`observation.ee_pos.robot_xyzrpy`, `skill.goal_position.robot_xyzrpy`, `skill.natural_language` run-length) 로부터 모든 호출 단위 staging.

```bash
python -m scripts.rebuild_subgoal_buffer_unified \
  --dataset "<HF repo_id or 절대 경로>" \
  --out "$SESSION/subgoal_buffer.npz"
```

**확인 출력**:
```
[rebuild] saved <N×11> entries across 11 skills
  skill_0: n=<N>  e.g. nl='Move to initial position' type='move_initial'
  skill_1: n=<N>  e.g. nl='Approach blue block...'  type='move_and_open'
  ...
  skill_10: n=<N> ...
```

### Step 3 — VDB cache 삭제 (next cycle 시 자동 rebuild)

```bash
rm -f "$SESSION/dct/skill_wise_vector_db.npz"
rm -f "$SESSION/skill_wise_vector_db.npz"
```

새 cycle 시작 시 client 가 새 `state_retrieval_key` (proprio L2 normalize) 로 reembedding → `$SESSION/dct/skill_wise_vector_db.npz` 저장.

### Step 4 — Phase2 cycle launch (grpc mode 자동 처리)

```bash
# run_forward_and_reset_ws<N>.sh 의 두 줄 확인:
#   PHASE="phase2"
#   RESUME_SESSION="<your-session>"

bash run_forward_and_reset_ws<N>.sh
```

Cycle 시작 시 client 가:
1. 새 VDB build (proprio L2 normalize 적용)
2. server 에 `grpc_server/buffer/server_skill_wise_vector_db.npz` scp upload
3. **`*server restart 권장*`** 메시지 출력 — server restart 필수.

### Step 5 — Server restart (grpc mode 만)

```bash
bash grpc_server/launch_remote_server.sh stop
bash grpc_server/launch_remote_server.sh
```

Server 가 새 buffer 를 메모리에 load. boot 후 `STATUS: OK` 확인.

### Step 6 — Cycle 재실행 (이번엔 정상 paradigm 으로)

```bash
bash run_forward_and_reset_ws<N>.sh
```

---

## 검증 지표

새 paradigm 이 정상 작동 중인지 cycle log 에서 확인:

| 신호 | 의미 |
|---|---|
| `[Method3 phase2] skill candidate hook SKIPPED — grpc mode (server-side Useful-OOD selection)` | paradigm layering 명시 — grpc 면 client noop |
| `[RecordingContext] Skill: <type> - <label> (call_index=N)` | 매 set_skill_info 호출 단위 ordinal stamp |
| `[Skill Perturbation] plan_batch START (... skill=skill_N [<type>])` | skill_id 가 ordinal 그대로 (off-by-1 X) |
| `[Phase2-Selection] skill=skill_N accepted \| K=64 under_covered=0/64 ... U_VLA=[...,μ=N]` | proprio scale balance 결과 — image part 의 contribution 살아남 |
| `[Skill Perturbation] plan_batch DONE — 1 candidates received` 후 **`[Phase2-Selection] cands=1`** 라인 **없음** | double-eval 자동 차단 |

---

## Random Selection Mode (ablation)

paradigm 의 selection rule 자체의 효과 검증용 baseline.

```yaml
# pipeline_config/phase2_config.yaml
mi_selection:
  selection_mode: random   # Q1 → random
                           # eligible(not under_covered) 중 uniform random pick
                           # eligible 비면 전체 cand 에서 random (fallback)
```

Method3 의 useful-OOD selection 와 비교용. 코드 변경 없이 yaml 토글만.

---

## 데이터 보호 (자동)

새 paradigm 의 `recorder.py:_has_collected_data` 가 dataset 의 `data/` 또는 `videos/` 에 episode 가 있으면 **destructive rmtree 거부** (옛 paradigm 에선 broken meta 만 보고 rmtree 했음 → data loss).

만약 cycle 시작 시 다음 에러:
```
REFUSING auto-rmtree — data/ 또는 videos/ 에 수집된 episode 존재!
```
→ dataset 의 `meta/tasks.parquet` 가 결손된 상태. 수동 복구:
```bash
# (a) 안전 backup
cp -r <dataset_path> <dataset_path>.bak_$(date +%s)
# (b) meta/tasks.parquet 다른 dataset 에서 복사 OR
#     LeRobotDataset.create() 후 data/ videos/ rsync
```

---

## 잠재 paradigm 한계 (향후 작업, 폐기됨)

- **subgoal absolute → detection-relative**: phase1 의 `random_pert` 가 매 episode object 위치 다양화 → phase2 의 fixed `seed_positions.json` (batch 첫 episode 위치만 saved) 과 mismatch. 사용자 폐기 결정 (task #22 deleted).
- **Image embedding 의 narrow distribution**: VLA backbone 의 last feature 가 task-specific image 사이 angular diversity 작음. proprio normalize 가 balance 처리하지만 image 자체의 정보량은 본질적 한계.

---

## 코드 참조

| 영역 | 파일 |
|---|---|
| Skill ordinal | `record_dataset/context.py` (`_skill_call_index` + callback hook) |
| Buffer staging | `method3/phase1_state_seeding/subgoal_selector.py` (`_on_skill_stamp`) |
| Replay cursor | `method3/phase2_mi_selection/subgoal_replay.py` (`_skill_call_index` lookup) |
| State key scale | `method3/reembedding/seed_builder.py:state_retrieval_key` + `method3/phase2_mi_selection/curobo_candidate_gen.py:_l2_unit` |
| Grpc layering | `execution_forward_and_reset.py:_attach_phase2_skill_hook` |
| Selection mode | `method3/phase2_mi_selection/mi_selector.py` (Q1~Q4 + random) |
| Buffer rebuild | `scripts/rebuild_subgoal_buffer_unified.py` |
| Dataset 보호 | `record_dataset/recorder.py:_has_collected_data` |
