# Multi-Task 병렬 운영 가이드 (task별 git branch 전략)

이 문서는 한 codebase에서 **서로 다른 task**(예: sort, pnp)를 **두 개 이상의 머신에서 동시에** 데이터 수집·학습·서버 launch 하기 위해 채택한 git branch 분리 전략을 설명한다.

---

## 1. 왜 branch 분리인가

`phase2_config.yaml`은 task별 단일 source-of-truth다. ckpt 경로, dataset 이름, gRPC port, tmux session, skill_dct_parquet 등이 전부 여기서 결정된다.

**한 파일을 두 task가 동시에 쓸 수 없다**:
- 양쪽 task가 같은 `phase2_config.yaml`을 push 하면 서로 덮어씀
- 서버는 yaml 한 개당 인스턴스 하나만 띄울 수 있음

해결 가능한 대안 3가지를 비교 검토했다:

| 대안 | 코드 변경 | 회귀 위험 | 운영 복잡도 |
|---|---|---|---|
| A) 코드 env-override (`PHASE2_CONFIG_PATH`) | 30+ 위치 수정 | 매우 높음 | 중 |
| B) **task별 git branch** (채택) | **0** | **0** | 낮음 |
| C) fork 분리 | 0 | 0 | 높음 (cherry-pick 양방향) |

**B를 택한 이유**: 코드 한 줄도 안 바꾸기 때문에 회귀 위험이 0이다. 각 branch가 자기 yaml만 들고 있고, 서버 launch 시 그 branch에서 `git pull`로 yaml이 그대로 도달한다.

---

## 2. 현재 구조

### Branch layout

```
master  ──● 공통 코드 + 기본 yaml (pnp 값)
         │
         └──● task/sort   (master + 1 commit: phase2_config.yaml = sort 값)
```

- **`master`** — pnp 머신이 checkout. `phase2_config.yaml`은 pnp용 값.
- **`task/sort`** — sort 머신(현 ws4)이 checkout. master 위에 sort 전용 1 commit이 얹혀있음.

### 두 branch의 phase2_config.yaml 차이 (현재)

| 키 | master (pnp) | task/sort |
|---|---|---|
| `phase1_trained_vla_path` | `lerobot/outputs/train/smolvla_dct_20260523_031844/checkpoints/001400/pretrained_model` | `lerobot/outputs/train/smolvla_dct_20260523_225127/checkpoints/003650/pretrained_model` |
| `phase1_dataset_path` | `CoRL2026-CSI/pnp_ours_100_table1` | `CoRL2026-CSI/table2/sort_ours_100_table1` |
| `phase1_min` | 30 | 40 |
| `transport.address` | `127.0.0.1:50061` | `127.0.0.1:50062` |
| `selector.skill_dct_parquet` | `results/skill_dct/pnp_ours_100_table1.parquet` | `results/skill_dct/sort_ours_100_table1.parquet` |
| `remote.tmux_session` | `phase2_server_pnp` ⚠️ (legacy `phase2_server` 는 §6 prefix 충돌) | `phase2_server_sort` |
| `remote.gpu_id` | 2 | 2 (H100 80GB에서 두 SmolVLA 공유) |
| `remote.project_path` | `/home/csiwoo/CORL2026/AutoDataCollector` (master worktree) | `/home/csiwoo/CORL2026/AutoDataCollector-sort` (task/sort worktree) |

**중요 (4가지 모두 필수)**:
1. `transport.address` port — task별 unique. 같으면 SSH tunnel 충돌.
2. `remote.tmux_session` — task별 unique **+ prefix 도 겹치지 않게**. tmux 의 `has-session -t` 는 prefix match 라 `phase2_server` 가 `phase2_server_sort` 와 match → 잘못된 reuse (§6 참고).
3. `remote.project_path` — task별 별도 디렉토리(원격에 `git worktree` 로). 같은 디렉토리면 두 launch 가 서로의 branch 를 checkout 해서 working tree 가 oscillate + `sync_artifacts` 가 같은 `grpc_server/buffer/server_skill_wise_vector_db.npz` 를 덮어쓴다.
4. `remote.gpu_id` — 같아도 OK (H100 80GB 면 fit). GPU memory 부족하면 다르게.

### Sort branch에만 들어있는 추가 변경
- `pipeline_config/phase1_config.yaml` — `readiness.auto_stop_on_ready: false`
- `pipeline_config/recording_config_ws4.yaml` — sort dataset_repo_id + ws4 wrist 카메라 `/dev/v4l/by-path/...`
- `run_forward_and_reset_ws4.sh` — ws4 머신 sort 전용 정리본

---

## 3. 운영 흐름

### 머신별 1회 setup (각 머신에서 한 번만)

**(a) git branch checkout**:
```bash
# pnp 머신
cd AutoDataCollector
git fetch hyeonseok
git checkout master

# sort 머신 (현재 ws4)
cd AutoDataCollector
git fetch hyeonseok
git checkout task/sort
```

**(b) 원격 server-side worktree** — 양쪽 머신이 같은 H100 server 공유할 때 *반드시* 필요:
```bash
# 원격 H100 에서 (한 번만):
ssh csiwoo@61.107.202.230
cd ~/CORL2026/AutoDataCollector       # 기본 = master (pnp)
git fetch origin
git worktree add ../AutoDataCollector-<task> task/<task>
# 예: git worktree add ../AutoDataCollector-sort task/sort
ls ~/CORL2026/                         # 두 디렉토리 다 있는지
```

각 task 의 yaml `remote.project_path` 가 자기 worktree 디렉토리를 가리키도록 (§5 Step 2 표 참고).

**(c) 클라이언트 env 의존성 (fresh 머신 setup)** — ADC 처음 clone 한 머신은 다음 순서:
```bash
# 1) base env (Python 3.10, pinocchio, opencv 등 — environment.yaml 의 conda 패키지)
conda env create -f environment.yaml
conda activate lerobot_cap

# 2) 전체 pip deps (200+ 패키지 — grpcio, gymnasium, google-genai, accelerate, av,
#    datasets, diffusers, transformers 등 phase2 client 가 필요로 하는 거 다 포함)
pip install -r requirements.txt

# 3) requirements.txt 누락 보강 (2026-05-24 기준)
pip install num2words                  # SmolVLM processor 가 import — 없으면 chain Step 3 에서 fail

# 4) lerobot 패키지: src layout (lerobot/src/lerobot/) 이고 Python 3.12+ 요구라
#    lerobot_cap=3.10 에 pip install -e 거부됨. 모든 ws 스크립트 / chain 가
#    PYTHONPATH 로 잡아야 함:
grep "PYTHONPATH" run_forward_and_reset_ws4.sh
# → export PYTHONPATH="$SCRIPT_DIR/lerobot/src:${PYTHONPATH:-}"
# 새 ws<N>.sh 만들 때 같은 줄 *반드시* 추가.
```

이미 setup 된 머신이 의존성 빠뜨려서 fail 한 경우는 누락된 패키지만 단발 install:
```bash
pip install grpcio grpcio-tools google-genai num2words gymnasium
```
(sequential ImportError 함정은 §6 참고.)

### 데이터 수집 cycle (각 머신에서 평소대로)

```bash
# phase1 (raw episode 수집)
bash run_forward_and_reset_wsN.sh    # PHASE="phase1"
```

phase1 종료 시 chain script가 자동 호출되어 Step 0–4를 돈다:
```bash
bash scripts/phase2_prep_chain.sh --dataset <repo_id> --session-dir <path>
```

**chain Step 4는 자기 branch의 `phase2_config.yaml`을 in-place 수정**한다. 즉 sort 머신에서 chain을 돌리면 `task/sort` branch의 yaml만 갱신된다.

### Phase2 launch

```bash
# 1) yaml 변경 commit + push (자기 branch)
git add pipeline_config/phase2_config.yaml
git commit -m "chore(<task>): yaml updated by chain"
git push hyeonseok <branch>

# 2) 원격 서버 launch (yaml의 remote/transport 자동 사용)
bash grpc_server/launch_remote_server.sh
```

`launch_remote_server.sh`는 원격 H100에서:
1. 자기 branch로 `git pull`
2. `sync_artifacts`로 ckpt/DB/parquet rsync
3. tmux 세션(`phase2_server_sort` 등)에서 server.py 부팅 (yaml의 gpu_id 사용)
4. SSH port forwarding tunnel(`127.0.0.1:50062` ↔ remote `50062`)

여기서 두 머신이 동시에 launch해도:
- tmux session 이름이 다름 → 원격에서 두 서버 인스턴스 공존
- port가 다름 → SSH tunnel 충돌 없음
- GPU(같은 H100 GPU 2)는 VRAM 공유 — SmolVLA 둘 다 (~15GB×2) H100 80GB에 fit

### Phase2 client 실행

```bash
# PHASE 토글 후 실행
# run_forward_and_reset_wsN.sh:
#   PHASE="phase1"  →  PHASE="phase2"
bash run_forward_and_reset_wsN.sh
```

각 머신의 client는 자기 yaml의 `transport.address` 그대로 호출. sort 머신은 `127.0.0.1:50062`, pnp 머신은 `127.0.0.1:50061`로 자동 라우팅.

---

## 4. 공통 코드 변경 시 sync 규칙

**원칙**: 공통 코드는 `master`가 SoT다. task branch는 master + (자기 task 전용 commits)이다.

### Case A — 공통 코드 변경 (skills, codegen prompt, chain script 등)

```bash
# 어느 머신에서든 (보통 master checkout 한 머신)
git checkout master
# ... 변경 ...
git add <files>
git commit -m "feat(common): ..."
git push hyeonseok master
```

그 후 **task branch sync**:

```bash
# sort 머신
git checkout task/sort
git fetch hyeonseok
git rebase hyeonseok/master            # task/sort 의 commits 를 master 위에 다시 얹음
git push --force-with-lease hyeonseok task/sort   # rebase 후엔 force 필요
```

`--force-with-lease`는 그동안 다른 사람이 task/sort에 push한 게 있으면 거부 → 안전.

### Case B — task 전용 변경 (해당 task의 ckpt/dataset 갱신, yaml 튜닝)

```bash
# sort 머신
git checkout task/sort
# ... 변경 ...
git add <files>
git commit -m "chore(task/sort): ..."
git push hyeonseok task/sort
```

pnp 머신은 영향 없음 (master만 pull).

### Case C — 한쪽 task의 commit을 공통으로 promote

```bash
# 예: task/sort 에서 만든 튜닝을 pnp 에도 적용
git checkout master
git cherry-pick <sha-from-task/sort>
git push hyeonseok master
# task/sort 쪽도 rebase 로 정합화 (위 Case A 와 동일)
```

---

## 5. 새 task 추가 방법

예: `task/wash` (washing dishes) 추가.

### Step 0 — (해당 머신이 처음이면) ADC repo + env setup
ADC 가 이미 clone 돼있고 `lerobot_cap` env 활성 가능한 머신이면 Step 1 로 바로. fresh 머신이면 §3 (c) 의 4단계 (conda env create → pip install -r requirements.txt → num2words → PYTHONPATH 확인) 먼저.

### Step 1 — branch 만들기

```bash
git fetch hyeonseok
git checkout -b task/wash hyeonseok/master   # master 베이스로 분기
```

### Step 1.5 — 원격 H100 에 worktree 추가 (한 번만)

```bash
ssh csiwoo@61.107.202.230
cd ~/CORL2026/AutoDataCollector
git fetch origin
git worktree add ../AutoDataCollector-wash task/wash
```

이걸 빠뜨리면 wash launch 가 master 디렉토리에서 git checkout task/wash 시도 → 기존 pnp/sort working tree 와 충돌 + `sync_artifacts` 가 다른 task 의 `server_skill_wise_vector_db.npz` 덮어씀.

### Step 2 — yaml 7 키 수정 (task별로 반드시 달라야 함)

```bash
$EDITOR pipeline_config/phase2_config.yaml
```

| 키 | 값 |
|---|---|
| `phase1_trained_vla_path` | wash 작업의 DCT-tuned ckpt 경로 |
| `phase1_dataset_path` | wash dataset repo id |
| `phase1_min` | wash phase1 최소 episode |
| `transport.address` | `127.0.0.1:50063` (다른 task와 다른 port — 50063 / 50064 / … 증가) |
| `selector.skill_dct_parquet` | `results/skill_dct/<wash_dataset>.parquet` |
| `remote.tmux_session` | `phase2_server_wash` (task별 unique **+ 다른 task tmux 이름의 prefix 가 되면 안 됨** — §6 prefix 충돌 참고) |
| `remote.project_path` | `/home/csiwoo/CORL2026/AutoDataCollector-wash` (Step 1.5 의 worktree 디렉토리) |
| `remote.gpu_id` | 사용할 원격 GPU (현재 GPU 2만 idle — 같은 GPU에 SmolVLA 셋이 fit하는지 확인 필요) |

### Step 3 — task별 추가 설정 (필요 시)

머신마다 다른 hardware path가 있으면:
- `pipeline_config/recording_config_wsN.yaml` 의 카메라 path
- `run_forward_and_reset_wsN.sh` 의 RESUME_SESSION 등

### Step 4 — 데이터 수집 & 학습 & DB build

평소 cycle: phase1 → chain → phase2_config.yaml 갱신.

### Step 5 — branch push

```bash
git add pipeline_config/phase2_config.yaml \
        pipeline_config/recording_config_wsN.yaml \
        run_forward_and_reset_wsN.sh
git commit -m "chore(task/wash): initial wash task config — port 50063, tmux phase2_server_wash"
git push -u hyeonseok task/wash
```

### Step 6 — wash 머신에서 launch

```bash
git checkout task/wash
bash grpc_server/launch_remote_server.sh
# (PHASE2 진행) run_forward_and_reset_wsN.sh 의 PHASE="phase2"
bash run_forward_and_reset_wsN.sh
```

---

## 6. 주의사항 & 알려진 함정

### GPU 공유 한계
현재 모든 task가 원격 H100의 **GPU 2 한 장**을 공유한다 (다른 GPU는 외부 사용자 점유). SmolVLA 한 인스턴스 ≈ 15GB VRAM이므로:
- 2개 task: OK (30GB / 80GB)
- 3개 task: 45GB — 여유 충분하지만 throughput 분산
- 4개 이상: throughput과 latency 영향 확인 필요

GPU 추가 확보 시 각 task yaml의 `remote.gpu_id`를 다르게 (0, 1, 2, 3) 설정하면 isolation 가능.

### Port 충돌
`transport.address`의 port는 **모든 task와 다른 머신을 통틀어 unique** 해야 한다. 현재 사용:
- 50061 — master (pnp)
- 50062 — task/sort

새 task는 50063부터.

### tmux session 이름 충돌 (단순 동일)
`remote.tmux_session`도 task별 unique. 같은 이름이면 두 번째 launch가 첫 번째 세션을 reuse(=오염)한다.

### ⚠️ tmux session 이름 *prefix* 충돌 (가장 함정스러움)
**`tmux has-session -t NAME` 은 prefix match**. `phase2_server` 가 `phase2_server_sort` 와도 match → pnp launch 가 sort 의 tmux 를 자기 것으로 reuse 하고 *새 server.py 를 안 부팅*. 결과: pnp client 가 Ready RPC Connection refused.

**규칙**:
- 모든 task tmux 이름을 `phase2_server_<task>` 패턴으로 통일. **단독 `phase2_server` 금지** — future task 와 항상 prefix conflict.
- `launch_remote_server.sh` 의 `has-session -t '=$NAME'` (= prefix 로 exact match 강제). 이미 적용됨.

검증 명령:
```bash
ssh csiwoo@61.107.202.230 'tmux ls | grep phase2'
# 두 task 띄운 후: phase2_server_pnp + phase2_server_sort 둘 다 보여야 정상.
# 한 개만 보이면 한쪽 launch 가 prefix match 로 reuse 한 것.
```

### `launch_remote_server.sh stop` 의 cross-task 영향 (수정됨)
원래 stop 은 `pkill -9 -f 'grpc_server.server'` 로 *모든* task 의 server 를 죽였다. 이제 `--port $REMOTE_PORT` 매칭으로 자기 task server 만 죽인다. tmux kill-session 도 `'=$NAME'` exact 적용. **2026-05-24 이후 launch_remote_server.sh 만 안전** — 원격에 옛 버전이 캐시돼있는 environment 라면 양쪽 server 가 같이 죽는 회귀 발생 가능. 의심되면 `git log -p grpc_server/launch_remote_server.sh | head -50` 으로 fix 들어왔는지 확인.

### 원격 server-side worktree 누락 (Step 1.5)
새 task 만들 때 yaml 만 분리하고 원격 worktree 안 만들면:
- 두 launch 가 같은 디렉토리에서 자기 branch checkout → working tree oscillate
- `sync_artifacts.sh` 가 같은 `grpc_server/buffer/server_skill_wise_vector_db.npz` 를 덮어씀 → 나중 launch 가 먼저 launch 의 DB 를 망가뜨림
- server.py 가 잘못된 ckpt path 로 부팅 (이전 launch yaml 로)

증상: 한쪽 server 가 *다른 task* 의 client 요청을 처리하는 것처럼 보임 (yaml 이 다른 task 로 mutate 된 상태).

### lerobot_cap env 의존성 누락 (sequential ImportError)
phase2 client 부팅 시 다음 import 순서로 fail 한다 (없는 패키지 만나면 한 번에 하나씩 멈춤):
```
grpcio          → ModuleNotFoundError: No module named 'grpc'
lerobot.utils   → No module named 'lerobot.utils'         (lerobot src layout, PYTHONPATH 미설정)
gymnasium       → No module named 'gymnasium'              (lerobot.envs)
num2words       → SmolVLM processor 가 num2words 요구
google.genai    → codegen llm_utils.gemini
```

각 머신 lerobot_cap setup 시 한 번에:
```bash
pip install grpcio grpcio-tools google-genai num2words gymnasium
```

또 `export PYTHONPATH="$SCRIPT_DIR/lerobot/src:${PYTHONPATH:-}"` 를 `run_forward_and_reset_ws*.sh` 상단에 추가 (lerobot 자체는 Python 3.12+ 요구라 `pip install -e lerobot/` 는 lerobot_cap=3.10 에서 거부됨).

### NVIDIA driver kernel/userspace mismatch (apt 자동 업데이트 함정)
시스템이 `apt unattended-upgrades` 로 `libnvidia-*` user-space lib 을 새 버전(예: 580.159) 으로 올렸는데 재부팅 안 하면, 메모리에 로드된 옛 커널 모듈(예: 580.126) 과 mismatch. 증상: `nvidia-smi: Failed to initialize NVML: Driver/library version mismatch`. PyTorch 는 CUDA fallback to CPU → 학습 10배 느려짐.

해결: 재부팅. 재부팅 후 dkms 빌드 실패하면 `gcc-12` 설치 후 `sudo CC=/usr/bin/gcc-12 dkms install nvidia/<ver> -k $(uname -r)`. 자세한 진단은 이 prompt 의 first-setup 작업 기록 참고.

### chain script가 yaml을 in-place 수정
`scripts/phase2_prep_chain.sh` Step 4는 자기 checkout된 branch의 `phase2_config.yaml`을 수정한다. 즉:
- task/sort에 checkout된 머신에서 chain 돌리면 → task/sort branch yaml만 갱신
- master에 checkout된 머신에서 chain 돌리면 → master branch yaml 갱신

**잘못된 branch에서 chain을 돌리면 다른 task의 yaml을 덮어쓴다**. chain 실행 전에 `git branch --show-current`로 확인 습관 권장.

### Sync 깜빡 방지
공통 코드 변경 후 task branch rebase를 잊으면 그 task는 outdated 코드로 계속 동작한다. PR 머지 후 sync 체크리스트:
```bash
git fetch hyeonseok
for br in task/sort task/wash; do
  echo "=== $br ==="
  git log --oneline hyeonseok/$br..hyeonseok/master
done
```
출력이 비어있지 않은 branch는 rebase 필요.

### 머신 hardware 의존 파일
`pipeline_config/recording_config_wsN.yaml`의 카메라 path(`/dev/v4l/by-path/...`)나 `run_forward_and_reset_wsN.sh`의 conda 경로는 머신마다 다를 수 있다. 다른 머신에 push 시 그 머신엔 영향 없으나(다른 hardware path), 혼동 방지를 위해 가능하면 머신별 ws 스크립트(`ws1.sh`, `ws4.sh` 등)를 분리해 사용.

---

## 7. 참고 — 첫 setup 시 적용한 git 작업 기록

처음 sort branch 분리 시(2026-05-24)의 작업 순서:

```bash
# 1) 다른 머신과 sync 위해 master rebase
git fetch hyeonseok master
git reset @{u}                          # 잘못 만든 local commit 폐기
git checkout -- pipeline_config/phase2_config.yaml
git stash push -m "auto-stash" -- <12 modified files>
git stash pop                           # auto 3-way merge
# (자동 merge 후 cleanup.py 회귀 확인 → 수동 패치)

# 2) 공통 변경 master 에 commit
git add <9 common files>
git commit -m "feat(common): cross-task improvements"
git rebase hyeonseok/master             # 그동안 들어온 새 commit 위에 얹기
git push hyeonseok master

# 3) sort branch 분기 + sort 전용 변경
git checkout -b task/sort
# phase2_config.yaml 6 키 수정 (Section 2 표 참고)
git add pipeline_config/phase2_config.yaml \
        pipeline_config/phase1_config.yaml \
        pipeline_config/recording_config_ws4.yaml \
        run_forward_and_reset_ws4.sh
git commit -m "chore(task/sort): ..."
git push -u hyeonseok task/sort
```
