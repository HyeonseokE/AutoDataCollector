#!/usr/bin/env bash
# SmolVLA 학습 실행 스크립트
#
#   ./train_smolvla.sh                                    # 기본값으로 학습
#   NUM_GPUS=4 ./train_smolvla.sh                         # 4 GPU DDP
#   NUM_GPUS=2 MIXED_PRECISION=bf16 ./train_smolvla.sh    # bf16
#   EPOCHS=30 ./train_smolvla.sh                          # epoch 수 변경 (STEPS 자동계산)
#   STEPS=1000 ./train_smolvla.sh                         # STEPS 직접 지정 시 epoch 계산 skip
#   ./train_smolvla.sh --optimizer.lr=2e-4                # draccus 인자 직접
#
# effective batch = BATCH_SIZE × NUM_GPUS.
# STEPS 미지정 시: dataset.total_frames 를 조회해 EPOCHS 만큼 환산.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1   # ~/.local (user-site) 차단
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH: AutoDataCollector lerobot + project root --------
# project root (= REPO_DIR/..) 도 PYTHONPATH 에 추가해야 lerobot trainer 에서
# method3.dct.skill_dct_dataset (SkillDCTDataset wrapper) 등을 import 가능.
PROJECT_ROOT="$(cd "$REPO_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# -------- 학습 설정 --------
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"   # "" 이면 from scratch
POLICY_REPO_ID="${POLICY_REPO_ID:-}"                 # 값 있으면 push_to_hub=true
DATASET_REPO_ID="${DATASET_REPO_ID:-CoRL2026-CSI/distribute_phase1_20_table1}"
DATASET_REVISION="${DATASET_REVISION:-v3.0}"         # HF tag/branch/commit
JOB_NAME="${JOB_NAME:-smolvla_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/train/$JOB_NAME}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"                     # STEPS 미지정 시 epoch → step 환산용
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"

NUM_GPUS="${NUM_GPUS:-1}"                  # 단일 GPU default (DDP off)
MIXED_PRECISION="${MIXED_PRECISION:-no}"   # no | fp16 | bf16
MASTER_PORT="${MASTER_PORT:-0}"            # 0 = auto

EFFECTIVE_BS=$(( BATCH_SIZE * NUM_GPUS ))

# -------- STEPS 자동계산 (EPOCHS → STEPS) --------
# STEPS 가 직접 주어지면 그대로 사용. 아니면 dataset.total_frames 기반 환산.
if [ -z "${STEPS:-}" ]; then
    echo "[INFO] Querying $DATASET_REPO_ID (revision=$DATASET_REVISION) for total_frames..."
    SAMPLE_COUNT=$(python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata("$DATASET_REPO_ID", revision="$DATASET_REVISION")
print(meta.total_frames)
PY
)
    if ! [[ "$SAMPLE_COUNT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to fetch total_frames" >&2
        echo "       got: $SAMPLE_COUNT" >&2
        exit 1
    fi
    STEPS_PER_EPOCH=$(( (SAMPLE_COUNT + EFFECTIVE_BS - 1) / EFFECTIVE_BS ))  # ceil
    STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
    echo "[INFO] total_frames=$SAMPLE_COUNT  global_batch=$EFFECTIVE_BS  steps/epoch=$STEPS_PER_EPOCH  epochs=$EPOCHS  -> STEPS=$STEPS"
fi

SAVE_FREQ="${SAVE_FREQ:-50000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-0}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-smolvla}"

# -------- 이미지 augmentation (각 transform 개별 on/off) --------
# IMG_AUG=true 일 때만 활성. 각 transform 은 1=on / 0=off.
# MAX_NUM 은 매 프레임당 샘플링할 개수 (자동으로 켜진 개수 이하로 clip).
# 파라미터 범위는 lerobot 기본값 사용.
IMG_AUG="${IMG_AUG:-true}"
IMG_AUG_MAX_NUM="${IMG_AUG_MAX_NUM:-3}"
IMG_AUG_RANDOM_ORDER="${IMG_AUG_RANDOM_ORDER:-true}"
IMG_AUG_BRIGHTNESS="${IMG_AUG_BRIGHTNESS:-1}"
IMG_AUG_CONTRAST="${IMG_AUG_CONTRAST:-1}"
IMG_AUG_SATURATION="${IMG_AUG_SATURATION:-1}"
IMG_AUG_HUE="${IMG_AUG_HUE:-1}"
IMG_AUG_SHARPNESS="${IMG_AUG_SHARPNESS:-1}"
IMG_AUG_AFFINE="${IMG_AUG_AFFINE:-1}"

# -------- 카메라 rename (dataset → policy) --------
# "dataset_name:policy_name" 형태, `observation.images.` prefix 자동.
# 비우려면 CAMERA_RENAME_PAIRS=()
# 완전한 JSON override 는 POLICY_RENAME_MAP 환경변수로.
CAMERA_RENAME_PAIRS=(
    "left_wrist:camera1"
    "top:camera2"
)

if [ -z "${POLICY_RENAME_MAP+x}" ]; then
    POLICY_RENAME_MAP=""
    if (( ${#CAMERA_RENAME_PAIRS[@]} )); then
        for _p in "${CAMERA_RENAME_PAIRS[@]}"; do
            _src="${_p%%:*}"; _dst="${_p##*:}"
            POLICY_RENAME_MAP+="${POLICY_RENAME_MAP:+, }\"observation.images.${_src}\": \"observation.images.${_dst}\""
        done
        POLICY_RENAME_MAP="{${POLICY_RENAME_MAP}}"
    fi
fi

# -------- policy CLI 분기 --------
# --policy.path 와 --policy.type 은 배타적 (lerobot parser.py).
POLICY_CLI_ARGS=()
if [ -n "$POLICY_PATH" ]; then
    POLICY_CLI_ARGS+=(--policy.path="$POLICY_PATH")
else
    POLICY_CLI_ARGS+=(--policy.type="$POLICY_TYPE")
fi
# smolvla_base 의 기본 config 는 push_to_hub=true 라서 명시적으로 꺼야 함.
if [ -n "$POLICY_REPO_ID" ]; then
    POLICY_CLI_ARGS+=(--policy.push_to_hub=true --policy.repo_id="$POLICY_REPO_ID")
else
    POLICY_CLI_ARGS+=(--policy.push_to_hub=false)
fi

# -------- 요약 출력 --------
if [ "$IMG_AUG" = "true" ]; then
    _aug_list=""
    [ "$IMG_AUG_BRIGHTNESS" = "1" ] && _aug_list+="brightness "
    [ "$IMG_AUG_CONTRAST"   = "1" ] && _aug_list+="contrast "
    [ "$IMG_AUG_SATURATION" = "1" ] && _aug_list+="saturation "
    [ "$IMG_AUG_HUE"        = "1" ] && _aug_list+="hue "
    [ "$IMG_AUG_SHARPNESS"  = "1" ] && _aug_list+="sharpness "
    [ "$IMG_AUG_AFFINE"     = "1" ] && _aug_list+="affine "
    IMG_AUG_SUMMARY="on [${_aug_list% }] (max_num=$IMG_AUG_MAX_NUM, random_order=$IMG_AUG_RANDOM_ORDER)"
    unset _aug_list
else
    IMG_AUG_SUMMARY="off"
fi
cat <<EOF
========= SmolVLA Training =========
 env     : $CONDA_ENV ($(python --version 2>&1))
 lerobot : $REPO_DIR/src (PYTHONPATH)
 policy  : type=$POLICY_TYPE  path=${POLICY_PATH:-<scratch>}  hub=${POLICY_REPO_ID:-off}
 dataset : $DATASET_REPO_ID @ $DATASET_REVISION
 output  : $OUTPUT_DIR
 train   : steps=$STEPS  epochs=$EPOCHS  batch=$BATCH_SIZE (global=$EFFECTIVE_BS)  workers=$NUM_WORKERS
 device  : $DEVICE  gpus=$NUM_GPUS  precision=$MIXED_PRECISION
 wandb   : $WANDB_ENABLE (project=$WANDB_PROJECT)
 rename  : ${POLICY_RENAME_MAP:-<none>}
 img aug : $IMG_AUG_SUMMARY
====================================
EOF

# -------- 실행 --------
# OUTPUT_DIR 은 mkdir 하지 않음 (train.py:122 가 존재 시 에러)
TRAIN_ARGS=(
    "${POLICY_CLI_ARGS[@]}"
    --policy.device="$DEVICE"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.revision="$DATASET_REVISION"
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --num_workers="$NUM_WORKERS"
    --seed="$SEED"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --eval_freq="$EVAL_FREQ"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
)
[ -n "$POLICY_RENAME_MAP" ] && TRAIN_ARGS+=(--rename_map="$POLICY_RENAME_MAP")

# 이미지 augmentation: 켜진 transform 만 JSON dict 로 조립해서 한 번에 전달.
if [ "$IMG_AUG" = "true" ]; then
    _tfs="" ; _n=0
    _add() {
        [ -n "$_tfs" ] && _tfs+=", "
        _tfs+="\"$1\": {\"weight\": 1.0, \"type\": \"$2\", \"kwargs\": $3}"
        _n=$((_n+1))
    }
    [ "$IMG_AUG_BRIGHTNESS" = "1" ] && _add brightness ColorJitter     '{"brightness": [0.8, 1.2]}'
    [ "$IMG_AUG_CONTRAST"   = "1" ] && _add contrast   ColorJitter     '{"contrast": [0.8, 1.2]}'
    [ "$IMG_AUG_SATURATION" = "1" ] && _add saturation ColorJitter     '{"saturation": [0.5, 1.5]}'
    [ "$IMG_AUG_HUE"        = "1" ] && _add hue        ColorJitter     '{"hue": [-0.05, 0.05]}'
    [ "$IMG_AUG_SHARPNESS"  = "1" ] && _add sharpness  SharpnessJitter '{"sharpness": [0.5, 1.5]}'
    [ "$IMG_AUG_AFFINE"     = "1" ] && _add affine      RandomAffine    '{"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}'
    unset -f _add
    if (( _n > 0 )); then
        _max_n="$IMG_AUG_MAX_NUM"
        (( _max_n > _n )) && _max_n="$_n"
        TRAIN_ARGS+=(
            --dataset.image_transforms.enable=true
            --dataset.image_transforms.max_num_transforms="$_max_n"
            --dataset.image_transforms.random_order="$IMG_AUG_RANDOM_ORDER"
            --dataset.image_transforms.tfs="{${_tfs}}"
        )
    else
        echo "WARN: IMG_AUG=true but all individual transforms are off; augmentation disabled." >&2
    fi
    unset _tfs _n _max_n
fi

TRAIN_ARGS+=("$@")

if [ "$NUM_GPUS" -gt 1 ]; then
    # lerobot 공식 multi-GPU 방식: docs/source/multi_gpu_training.mdx
    # PYTHONPATH 로 AutoDataCollector lerobot 을 쓰므로 python -m 으로 통일.
    python -m accelerate.commands.launch \
        --multi_gpu \
        --num_processes="$NUM_GPUS" \
        --num_machines=1 \
        --mixed_precision="$MIXED_PRECISION" \
        --main_process_port="$MASTER_PORT" \
        -m lerobot.scripts.lerobot_train \
        "${TRAIN_ARGS[@]}"
else
    python -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
fi

# -------- post-train: D_phase1 (ID) U_VLA distribution sidecar JSON --------
# paper §3.4 식 (18) 의 U_VLA 를 D_phase1 random sample 에 대해 계산하여
# {ckpt}/uvla_id_stats.json sidecar 로 저장. server boot 시 이 JSON 을 자동
# lookup 하여 Phase2 selector 의 OOD/ID gate (tau_U_ID) 로 사용한다.
#
# 비활성화: SKIP_UVLA_ID=1 ./train_smolvla.sh
if [ "${SKIP_UVLA_ID:-0}" = "1" ]; then
    echo ""
    echo "[uvla_id] SKIP_UVLA_ID=1 — sidecar generation skipped."
else
    # 가장 최근 checkpoint 디렉터리 (예: outputs/.../checkpoints/001400/).
    LAST_CKPT_DIR=$(ls -dt "$OUTPUT_DIR/checkpoints/"*/ 2>/dev/null | head -1)
    LAST_CKPT="${LAST_CKPT_DIR%/}/pretrained_model"
    DATASET_NAME="${DATASET_REPO_ID##*/}"
    PARQUET="$PROJECT_ROOT/results/skill_dct/${DATASET_NAME}.parquet"

    echo ""
    echo "========= [uvla_id] post-train sidecar JSON ========="
    echo " ckpt    : $LAST_CKPT"
    echo " dataset : $DATASET_REPO_ID"
    echo " parquet : $PARQUET"
    echo "====================================================="
    if [ -d "$LAST_CKPT" ] && [ -f "$PARQUET" ]; then
        python "$PROJECT_ROOT/tools/eval_phase1_uvla_distribution.py" \
            --vla "$LAST_CKPT" \
            --dataset "$DATASET_REPO_ID" \
            --parquet "$PARQUET" \
            --n_samples "${UVLA_N_SAMPLES:-100}" \
            --R "${UVLA_R:-4}" \
            --sigma "${UVLA_SIGMA:-0.5}" \
            --agg "${UVLA_AGG:-mean}" \
            --device "$DEVICE" \
            || echo "[uvla_id][WARN] sidecar generation failed (non-fatal)."
    else
        echo "[uvla_id][WARN] ckpt or parquet missing — skip."
        echo "                ckpt    exists=$([ -d "$LAST_CKPT" ] && echo y || echo n)"
        echo "                parquet exists=$([ -f "$PARQUET" ] && echo y || echo n)"
    fi
fi
