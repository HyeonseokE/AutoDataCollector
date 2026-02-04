# CaP → LeRobot 학습 파이프라인 가이드

이 가이드는 Code-as-Policies(CaP)로 생성된 로봇 조작 데이터를 수집하고,
LeRobot v3.0 데이터셋으로 저장하여 Diffusion Policy를 학습하는 전체 과정을 설명합니다.

---

## 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│                    CaP → LeRobot 학습 파이프라인                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Step 1] 데이터 수집 (Forward + Reset 반복)                      │
│      ./run_forward_and_reset.sh                                │
│      │                                                          │
│      ▼                                                          │
│  [Step 2] 데이터셋 검증                                           │
│      python scripts/verify_dataset.py --dataset <name>         │
│      │                                                          │
│      ▼                                                          │
│  [Step 3] LeRobot 공식 학습 파이프라인 실행                        │
│      ./scripts/run_lerobot_train.sh <dataset_name>             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: 데이터 수집

### 1.1 설정 파일 수정

`run_forward_and_reset.sh`에서 핵심 설정 수정:

```bash
# 에피소드 수 (많을수록 좋은 학습 데이터)
NUM_EPISODES=50  # 권장: 50-100 에피소드

# 데이터셋 레코딩 활성화 (필수!)
RECORD_DATASET=true

# 태스크 명령어
INSTRUCTION="pick up the yellow dice and place it on the blue dish"
```

### 1.2 데이터 수집 실행

```bash
./run_forward_and_reset.sh
```

### 1.3 데이터 저장 위치

```
~/.cache/huggingface/lerobot/local/cap_dataset_YYYYMMDD_HHMMSS/
├── meta/
│   ├── info.json           # 데이터셋 메타정보
│   ├── stats.json          # 통계 (정규화용)
│   └── episodes/           # 에피소드 정보
├── data/                   # 상태/액션 데이터 (parquet)
└── videos/                 # 카메라 영상 (mp4)
```

---

## Step 2: 데이터셋 검증

### 2.1 수집된 데이터셋 확인

```bash
# 사용 가능한 데이터셋 목록
ls ~/.cache/huggingface/lerobot/local/ | grep cap_

# 데이터셋 검증 스크립트 실행
python scripts/verify_dataset.py --dataset cap_dataset_YYYYMMDD_HHMMSS -v
```

예상 출력:
```
============================================================
LeRobot Dataset Verification
============================================================

Dataset: cap_dataset_YYYYMMDD_HHMMSS
Path: /home/csi/.cache/huggingface/lerobot/local/cap_dataset_YYYYMMDD_HHMMSS

[1/5] Checking required files...
  [OK] info.json
  [OK] stats.json
  [OK] data/
  [OK] videos/

[2/5] Verifying metadata (info.json)...
  Codebase version: v3.0
  Total episodes: 50
  Total frames: 26400
  FPS: 30

[3/5] Verifying features...
  [OK] observation.state: dtype=float32, shape=[6]
  [OK] action: dtype=float32, shape=[6]
  [OK] observation.images.front: dtype=video, shape=[480, 640, 3]

[5/5] Testing LeRobot compatibility...
  [OK] LeRobotDataset loaded: 26400 samples

============================================================
Verification Complete!
  Status: VALID
============================================================
```

---

## Step 3: LeRobot 공식 학습 파이프라인 실행

### 3.1 래퍼 스크립트 사용 (권장)

```bash
# 기본 학습 실행
./scripts/run_lerobot_train.sh cap_dataset_YYYYMMDD_HHMMSS

# 학습 파라미터 변경
./scripts/run_lerobot_train.sh cap_dataset_YYYYMMDD_HHMMSS \
    --steps=100000 \
    --batch_size=16
```

### 3.2 직접 lerobot_train.py 실행

```bash
# Python path 설정
export PYTHONPATH="lerobot/src:$PYTHONPATH"

# 학습 실행
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/cap_dataset_YYYYMMDD_HHMMSS" \
    --dataset.root="~/.cache/huggingface/lerobot/local/cap_dataset_YYYYMMDD_HHMMSS" \
    --policy.type="diffusion" \
    --policy.push_to_hub=false \
    --steps=50000 \
    --batch_size=32 \
    --log_freq=100 \
    --save_freq=5000 \
    --eval_freq=0 \
    --num_workers=4 \
    --output_dir="outputs/train_cap"
```

### 3.3 주요 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--policy.type` | diffusion | 정책 타입 (diffusion, act, vqbet 등) |
| `--steps` | 50000 | 총 학습 스텝 |
| `--batch_size` | 32 | 배치 크기 |
| `--log_freq` | 100 | 로그 출력 주기 |
| `--save_freq` | 5000 | 체크포인트 저장 주기 |
| `--eval_freq` | 0 | 평가 주기 (0=비활성화) |
| `--num_workers` | 4 | DataLoader 워커 수 |

### 3.4 학습 로그 예시

```
INFO step:100 smpl:3200 ep:6 epch:0.12 loss:1.054 grdn:7.2 lr:1.0e-05
INFO step:200 smpl:6400 ep:12 epch:0.24 loss:0.892 grdn:5.8 lr:2.0e-05
INFO step:300 smpl:9600 ep:18 epch:0.36 loss:0.756 grdn:4.5 lr:3.0e-05
...
INFO step:50000 smpl:1600000 ep:3030 epch:60.6 loss:0.234 grdn:2.1 lr:1.0e-04
INFO Checkpoint policy after step 50000
INFO End of training
```

### 3.5 학습 결과물

```
outputs/train_cap/
├── checkpoints/
│   ├── 005000/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   ├── 010000/
│   └── ...
└── last/  # 최종 체크포인트 심링크
```

---

## 빠른 시작 요약

```bash
# 1. 데이터 수집 (50 에피소드)
vim run_forward_and_reset.sh  # NUM_EPISODES=50 설정
./run_forward_and_reset.sh

# 2. 데이터셋 확인
ls ~/.cache/huggingface/lerobot/local/ | grep cap_
python scripts/verify_dataset.py --dataset cap_dataset_YYYYMMDD_HHMMSS

# 3. 학습 실행
./scripts/run_lerobot_train.sh cap_dataset_YYYYMMDD_HHMMSS

# 4. 결과 확인
ls outputs/train_*/last/
```

---

## 추가 옵션

### 다른 정책 타입 사용

```bash
# ACT (Action Chunking with Transformers)
./scripts/run_lerobot_train.sh cap_dataset_... --policy.type=act

# VQ-BeT
./scripts/run_lerobot_train.sh cap_dataset_... --policy.type=vqbet
```

### WandB 로깅 활성화

```bash
./scripts/run_lerobot_train.sh cap_dataset_... \
    --wandb.enable=true \
    --wandb.project="cap_training"
```

### 이미지 augmentation 활성화

```bash
./scripts/run_lerobot_train.sh cap_dataset_... \
    --dataset.image_transforms.enable=true
```

---

## 권장 설정

| 항목 | 권장값 | 비고 |
|------|-------|------|
| 에피소드 수 | 50-100 | 더 많을수록 좋음 |
| 학습 스텝 | 50,000-100,000 | 데이터 양에 따라 조정 |
| 배치 크기 | 32 | GPU 메모리에 따라 조정 |
| FPS | 30 | 로봇 제어 주기와 일치 |

---

## 문제 해결

### GPU 메모리 부족

```bash
# 배치 크기 줄이기
./scripts/run_lerobot_train.sh cap_dataset_... --batch_size=8
```

### 데이터 로딩 느림

```bash
# 워커 수 늘리기
./scripts/run_lerobot_train.sh cap_dataset_... --num_workers=8
```

### 체크포인트에서 재시작

```bash
python -m lerobot.scripts.lerobot_train \
    --config_path="outputs/train_cap/last/train_config.json" \
    --resume=true
```
