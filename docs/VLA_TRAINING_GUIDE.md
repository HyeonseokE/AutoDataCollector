# 데이터 취득 → VLA 학습 전체 가이드

이 문서는 CaP(Code-as-Policies)로 로봇 조작 데이터를 수집하고, LeRobot의 SmolVLA 모델을 학습하는 전체 과정을 설명합니다.

---

## 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│           CaP 데이터 수집 → SmolVLA 학습 파이프라인            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Step 1] 설정 파일 수정                                     │
│      ↓                                                      │
│  [Step 2] 데이터 수집 실행                                   │
│      ↓                                                      │
│  [Step 3] 데이터셋 검증                                      │
│      ↓                                                      │
│  [Step 4] SmolVLA 학습                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 설정 파일 수정

### 1-1. 태스크 및 에피소드 설정 (`run_forward_and_reset.sh`)

```bash
vim run_forward_and_reset.sh
```

```bash
# 수정할 부분:
ROBOT_ID=3                    # 로봇 번호
INSTRUCTION="pick up the yellow dice and place it on the blue dish"  # 태스크
NUM_EPISODES=50               # 에피소드 수 (권장: 50-100)
RECORD_DATASET=true           # 레코딩 활성화 (필수)
```

### 1-2. 객체 설정 (`pipeline_config/detection_config.yaml`)

```yaml
objects:
  - "yellow dice"    # INSTRUCTION에 맞게 수정
  - "blue dish"
```

---

## Step 2: 데이터 수집 실행

```bash
./run_forward_and_reset.sh
```

**실행 중 출력:**
```
========================================
Forward + Reset Pipeline
========================================
Instruction: pick up the yellow dice and place it on the blue dish
Num Episodes: 50
Record Dataset: true
========================================

[Episode 1/50] Forward: SUCCESS, Reset: SUCCESS
[Episode 2/50] Forward: SUCCESS, Reset: SUCCESS
...

[DatasetRecorder] Dataset finalized at:
  ~/.cache/huggingface/lerobot/local/cap_dataset_YYYYMMDD_HHMMSS
```

---

## Step 3: 데이터셋 검증

```bash
# 데이터셋 목록 확인
ls ~/.cache/huggingface/lerobot/local/ | grep cap_

# 데이터셋 검증
python scripts/verify_dataset.py --dataset cap_dataset_YYYYMMDD_HHMMSS
```

**정상 출력:**
```
============================================================
LeRobot Dataset Verification
============================================================
  Codebase version: v3.0
  Total episodes: 50
  Total frames: 26000

  [OK] observation.state: (6,) float32
  [OK] action: (6,) float32
  [OK] observation.images.front: (480, 640, 3) video

  Status: VALID
============================================================
```

---

## Step 4: SmolVLA 학습

```bash
# Python path 설정
export PYTHONPATH="lerobot/src:$PYTHONPATH"

# SmolVLA 학습 실행
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/cap_dataset_YYYYMMDD_HHMMSS" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/local/cap_dataset_YYYYMMDD_HHMMSS" \
    --policy.type="smolvla" \
    --policy.push_to_hub=false \
    --steps=10000 \
    --batch_size=4 \
    --log_freq=100 \
    --save_freq=1000 \
    --eval_freq=0 \
    --num_workers=4 \
    --output_dir="outputs/smolvla_training"
```

**학습 로그:**
```
INFO Creating policy
  num_learnable_params=100M
  num_total_params=450M

INFO Start offline training
  step:100  loss:1.426  lr:9.2e-05
  step:200  loss:1.194  lr:8.5e-05
  step:500  loss:1.053  lr:5.8e-05
  ...
  step:10000 loss:0.856  lr:2.5e-06

INFO Checkpoint policy after step 10000
INFO End of training
```

---

## 빠른 실행 요약

```bash
# 1. 설정 (에피소드 수 변경)
sed -i 's/NUM_EPISODES=.*/NUM_EPISODES=50/' run_forward_and_reset.sh

# 2. 데이터 수집
./run_forward_and_reset.sh

# 3. 데이터셋 이름 확인
DATASET=$(ls -t ~/.cache/huggingface/lerobot/local/ | grep cap_ | head -1)
echo "Dataset: $DATASET"

# 4. 검증
python scripts/verify_dataset.py --dataset $DATASET

# 5. SmolVLA 학습
PYTHONPATH=lerobot/src:$PYTHONPATH python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/$DATASET" \
    --dataset.root="$HOME/.cache/huggingface/lerobot/local/$DATASET" \
    --policy.type="smolvla" \
    --policy.push_to_hub=false \
    --steps=10000 \
    --batch_size=4 \
    --output_dir="outputs/smolvla_$DATASET"
```

---

## 주요 파라미터 설명

| 파라미터 | 설명 | 권장값 |
|---------|------|-------|
| `--policy.type` | 모델 타입 | `smolvla` |
| `--steps` | 학습 스텝 | 10,000-50,000 |
| `--batch_size` | 배치 크기 | 2-8 (GPU 메모리에 따라) |
| `--log_freq` | 로그 주기 | 100 |
| `--save_freq` | 체크포인트 저장 주기 | 1000-5000 |

---

## 학습 결과물

```
outputs/smolvla_training/
├── checkpoints/
│   ├── 001000/
│   ├── 002000/
│   └── ...
└── last/  → 최종 모델
    ├── config.json
    ├── model.safetensors
    └── ...
```

---

## 다른 정책 타입 사용

```bash
# Diffusion Policy (263M params)
--policy.type="diffusion"

# ACT (Action Chunking Transformer)
--policy.type="act"

# VQ-BeT
--policy.type="vqbet"
```

---

## SmolVLA 모델 정보

| 항목 | 값 |
|------|-----|
| VLM Backbone | SmolVLM2-500M-Video-Instruct |
| 총 파라미터 | 450M |
| 학습 파라미터 | 100M (Expert만 학습) |
| 입력 | Image + State + Language |
| 출력 | Action sequence (chunk_size=50) |

---

## 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
--batch_size=2
```

### 패키지 누락
```bash
pip install num2words termcolor
```

### 데이터셋 경로 오류
```bash
# 전체 경로 확인
ls -la ~/.cache/huggingface/lerobot/local/cap_dataset_*/meta/info.json
```

### 체크포인트에서 재시작
```bash
python -m lerobot.scripts.lerobot_train \
    --config_path="outputs/smolvla_training/last/train_config.json" \
    --resume=true
```
