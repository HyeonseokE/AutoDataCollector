#!/usr/bin/env bash
# SmolVLA *DCT paradigm* 학습 스크립트 (method3 §6 / paradigm step [1]).
#
# 일반 SmolVLA 학습은 ``train_smolvla.sh`` 를 사용. 본 스크립트는 *DCT-tuned*
# VLA 학습 전용으로, 다음을 강제한다:
#   - SKILL_DCT_PARQUET (build_skill_dct.py 의 출력) 필수
#   - dataset 의 skill segment 수를 기준으로 STEPS 계산 (frame-level X)
#   - JOB_NAME default = smolvla_dct_<timestamp>
#   - lerobot dataset factory 가 SkillDCTDataset wrap branch 진입 (action 자리에
#     skill 단위 DCT_50 target, language 에 skill_type prefix)
#
# 사용:
#   SKILL_DCT_PARQUET=results/skill_dct/<ds>.parquet \
#   DATASET_REPO_ID=CoRL2026-CSI/<ds> \
#   bash lerobot/scripts/train_DCT_smolvla.sh
#
# 옵션 env (train_smolvla.sh 와 동일):
#   NUM_GPUS / MIXED_PRECISION / EPOCHS / STEPS / BATCH_SIZE / ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# Auto-detect conda profile (miniconda3 / anaconda3 / CONDA_PREFIX 기반)
_CONDA_SH=""
for _p in "$HOME/miniconda3/etc/profile.d/conda.sh" \
          "$HOME/anaconda3/etc/profile.d/conda.sh" \
          "${CONDA_PREFIX:+$CONDA_PREFIX/etc/profile.d/conda.sh}" \
          "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -n "$_p" ] && [ -f "$_p" ]; then _CONDA_SH="$_p"; break; fi
done
if [ -z "$_CONDA_SH" ]; then
    echo "ERROR: conda.sh not found. Set CONDA_PREFIX or install conda." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$_CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH --------
PROJECT_ROOT="$(cd "$REPO_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# -------- DCT paradigm 강제 --------
SKILL_DCT_PARQUET="${SKILL_DCT_PARQUET:-}"
if [ -z "$SKILL_DCT_PARQUET" ]; then
    echo "ERROR: train_DCT_smolvla.sh requires SKILL_DCT_PARQUET env." >&2
    echo "       e.g., SKILL_DCT_PARQUET=results/skill_dct/<ds>.parquet bash $0" >&2
    exit 1
fi
if [ ! -f "$SKILL_DCT_PARQUET" ]; then
    echo "ERROR: skill DCT parquet not found: $SKILL_DCT_PARQUET" >&2
    exit 1
fi

# -------- 학습 설정 --------
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-CoRL2026-CSI/distribute_phase1_20_table1}"
DATASET_REVISION="${DATASET_REVISION:-v3.0}"
JOB_NAME="${JOB_NAME:-smolvla_dct_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/train/$JOB_NAME}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"

NUM_GPUS="${NUM_GPUS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
MASTER_PORT="${MASTER_PORT:-0}"

EFFECTIVE_BS=$(( BATCH_SIZE * NUM_GPUS ))

# -------- STEPS 자동계산 (skill segments 기준) --------
# paradigm: 한 sample = 한 skill segment. epoch 당 segments/batch steps.
if [ -z "${STEPS:-}" ]; then
    echo "[INFO] DCT paradigm — counting skill segments in $SKILL_DCT_PARQUET..."
    SAMPLE_COUNT=$(python - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("$SKILL_DCT_PARQUET").num_rows)
PY
)
    if ! [[ "$SAMPLE_COUNT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to fetch segment count: $SAMPLE_COUNT" >&2
        exit 1
    fi
    STEPS_PER_EPOCH=$(( (SAMPLE_COUNT + EFFECTIVE_BS - 1) / EFFECTIVE_BS ))
    STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
    echo "[INFO] segments=$SAMPLE_COUNT  global_batch=$EFFECTIVE_BS  steps/epoch=$STEPS_PER_EPOCH  epochs=$EPOCHS  -> STEPS=$STEPS"
fi

SAVE_FREQ="${SAVE_FREQ:-50000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-0}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-smolvla-dct}"

# -------- 이미지 augmentation --------
IMG_AUG="${IMG_AUG:-true}"
IMG_AUG_MAX_NUM="${IMG_AUG_MAX_NUM:-3}"
IMG_AUG_RANDOM_ORDER="${IMG_AUG_RANDOM_ORDER:-true}"
IMG_AUG_BRIGHTNESS="${IMG_AUG_BRIGHTNESS:-1}"
IMG_AUG_CONTRAST="${IMG_AUG_CONTRAST:-1}"
IMG_AUG_SATURATION="${IMG_AUG_SATURATION:-1}"
IMG_AUG_HUE="${IMG_AUG_HUE:-1}"
IMG_AUG_SHARPNESS="${IMG_AUG_SHARPNESS:-1}"
IMG_AUG_AFFINE="${IMG_AUG_AFFINE:-1}"

# -------- 카메라 rename --------
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

# -------- policy CLI --------
POLICY_CLI_ARGS=()
if [ -n "$POLICY_PATH" ]; then
    POLICY_CLI_ARGS+=(--policy.path="$POLICY_PATH")
else
    POLICY_CLI_ARGS+=(--policy.type="$POLICY_TYPE")
fi
if [ -n "$POLICY_REPO_ID" ]; then
    POLICY_CLI_ARGS+=(--policy.push_to_hub=true --policy.repo_id="$POLICY_REPO_ID")
else
    POLICY_CLI_ARGS+=(--policy.push_to_hub=false)
fi

# -------- 요약 --------
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
========= SmolVLA DCT-paradigm Training =========
 env            : $CONDA_ENV ($(python --version 2>&1))
 lerobot        : $REPO_DIR/src (PYTHONPATH)
 policy         : type=$POLICY_TYPE  path=${POLICY_PATH:-<scratch>}  hub=${POLICY_REPO_ID:-off}
 dataset        : $DATASET_REPO_ID @ $DATASET_REVISION
 skill DCT prq  : $SKILL_DCT_PARQUET
 output         : $OUTPUT_DIR
 train          : steps=$STEPS  epochs=$EPOCHS  batch=$BATCH_SIZE (global=$EFFECTIVE_BS)  workers=$NUM_WORKERS
 device         : $DEVICE  gpus=$NUM_GPUS  precision=$MIXED_PRECISION
 wandb          : $WANDB_ENABLE (project=$WANDB_PROJECT)
 rename         : ${POLICY_RENAME_MAP:-<none>}
 img aug        : $IMG_AUG_SUMMARY
=================================================
EOF

# -------- 실행 --------
TRAIN_ARGS=(
    "${POLICY_CLI_ARGS[@]}"
    --policy.device="$DEVICE"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.revision="$DATASET_REVISION"
    --dataset.video_backend="${VIDEO_BACKEND:-pyav}"
    --dataset.skill_dct_parquet="$SKILL_DCT_PARQUET"
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
