#!/usr/bin/env bash
# ============================================================
# phase2_prep_chain.sh — Phase1 종료 후 Phase2 사전 준비 3-step chain
#
# Phase1 early termination 이 발생하면 execution_forward_and_reset.py 의
# main() 끝에서 prompt 후 본 스크립트를 호출한다. 수동 호출도 가능:
#
#   bash scripts/phase2_prep_chain.sh \
#       --dataset CoRL2026-CSI/pnp_ours_100_table1 \
#       --session-dir results/session_20260522_044832
#
# 옵션:
#   --skip-train      : Step 2 (VLA 학습) 건너뛰기. --vla-ckpt 명시 필요.
#   --vla-ckpt PATH   : Step 2 skip 시 Step 3 의 vla checkpoint 지정.
#   --train-job NAME  : Step 2 JOB_NAME override (default smolvla_dct_<ts>).
#   --steps N         : Step 2 STEPS override (default = train script 의 epoch 환산).
#
# 각 step exit code 검증 — fail 시 chain 즉시 중단.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

# ---------- color helpers ----------
bold()  { printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
info()  { printf "\033[36m  %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m!! %s\033[0m\n" "$*"; }
err()   { printf "\033[1;31mXX %s\033[0m\n" "$*" >&2; }
green() { printf "\033[1;32m++ %s\033[0m\n" "$*"; }

# ---------- args ----------
DATASET=""
SESSION_DIR=""
SKIP_TRAIN=0
VLA_CKPT=""
TRAIN_JOB=""
STEPS_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)         DATASET="$2"; shift 2 ;;
        --session-dir)     SESSION_DIR="$2"; shift 2 ;;
        --skip-train)      SKIP_TRAIN=1; shift ;;
        --vla-ckpt)        VLA_CKPT="$2"; shift 2 ;;
        --train-job)       TRAIN_JOB="$2"; shift 2 ;;
        --steps)           STEPS_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) err "unknown arg: $1"; exit 2 ;;
    esac
done

if [ -z "$DATASET" ]; then err "--dataset required"; exit 2; fi
if [ -z "$SESSION_DIR" ]; then err "--session-dir required"; exit 2; fi
if [ ! -d "$SESSION_DIR" ]; then err "session dir not found: $SESSION_DIR"; exit 2; fi

# dataset basename (e.g., CoRL2026-CSI/pnp_ours_100_table1 → pnp_ours_100_table1)
DATASET_BASENAME="$(echo "$DATASET" | awk -F/ '{print $NF}')"
# absolute path — train_smolvla.sh 가 cwd 를 lerobot/ 로 변경하므로
# relative path 는 resolve 실패. session-dir 도 동일.
SKILL_DCT_PARQUET="$PROJ_ROOT/results/skill_dct/${DATASET_BASENAME}.parquet"
SESSION_DIR_ABS="$(cd "$SESSION_DIR" && pwd)"
SESSION_DIR="$SESSION_DIR_ABS"

bold "Phase2 prep chain"
info "dataset      : $DATASET"
info "session dir  : $SESSION_DIR"
info "skill DCT parquet : $SKILL_DCT_PARQUET"

# ============================================================
# Step 0 — session 폴더 reorg (phase1/ phase2/ 구조)
# ============================================================
# chain 진입 = Phase1 종료 → Phase2 준비. 기존 flat 한 episode_*/ 를
# phase1/ 하위로 옮기고 phase2/ 빈 폴더 생성. 이후 Phase2 cycle 의
# episode 는 phase2/ 하위에 쌓인다. idempotent (이미 reorg 됐으면 0 moved).
bold "Step 0/3 — session 폴더 reorg (episode_* → phase1/)"
python - "$SESSION_DIR" <<'PY'
import sys, shutil
from pathlib import Path
sd = Path(sys.argv[1])
phase1 = sd / "phase1"; phase1.mkdir(exist_ok=True)
moved = 0
for ep in sorted(sd.glob("episode_*")):
    if not ep.is_dir():
        continue
    dst = phase1 / ep.name
    if dst.exists():
        print(f"  skip (dest exists): {dst}")
        continue
    shutil.move(str(ep), str(dst))
    moved += 1
(sd / "phase2").mkdir(exist_ok=True)
print(f"  moved {moved} episode_* → {phase1}/  (phase2/ ready)")
PY

# ============================================================
# Step 1 — skill segment DCT parquet (VLA 학습용 preprocessed dataset)
# ============================================================
bold "Step 1/3 — build skill segment DCT parquet"

mkdir -p "$(dirname "$SKILL_DCT_PARQUET")"

# conda env (lerobot_cap) — train script 와 동일.
CONDA_ENV="${CONDA_ENV:-lerobot_cap}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if python -m method3.dct.build_skill_dct \
        --dataset "$DATASET" \
        --out "$SKILL_DCT_PARQUET"; then
    green "Step 1 done — skill DCT parquet at $SKILL_DCT_PARQUET"
else
    err "Step 1 failed (build_skill_dct)"
    exit 1
fi

# ============================================================
# Step 2 — DCT-tuned VLA training (optional)
# ============================================================
if [ "$SKIP_TRAIN" -eq 1 ]; then
    bold "Step 2/3 — VLA training SKIPPED (--skip-train)"
    if [ -z "$VLA_CKPT" ]; then
        err "--skip-train requires --vla-ckpt PATH"; exit 2
    fi
    if [ ! -d "$VLA_CKPT" ]; then
        err "vla ckpt not found: $VLA_CKPT"; exit 2
    fi
    info "using existing ckpt: $VLA_CKPT"
else
    bold "Step 2/3 — DCT-tuned VLA training"
    if [ -z "$TRAIN_JOB" ]; then
        TRAIN_JOB="smolvla_dct_$(date +%Y%m%d_%H%M%S)"
    fi
    info "job name     : $TRAIN_JOB"
    info "output       : lerobot/outputs/train/$TRAIN_JOB"

    _train_envs=(
        "DATASET_REPO_ID=$DATASET"
        "SKILL_DCT_PARQUET=$SKILL_DCT_PARQUET"
        "JOB_NAME=$TRAIN_JOB"
        "CONDA_ENV=lerobot_cap"
    )
    if [ -n "$STEPS_OVERRIDE" ]; then
        _train_envs+=("STEPS=$STEPS_OVERRIDE")
    fi

    # train_DCT_smolvla.sh — DCT paradigm 전용 (일반 train_smolvla.sh 와 분리됨).
    # 자체 conda activate 라 PYTHONPATH 등 다시 잡힘 — 안전.
    if env "${_train_envs[@]}" bash lerobot/scripts/train_DCT_smolvla.sh; then
        green "Step 2 done — training complete"
    else
        err "Step 2 failed (train_DCT_smolvla)"
        exit 1
    fi

    # latest checkpoint 자동 검색 — mtime desc 의 첫 번째
    VLA_CKPT="$(find "lerobot/outputs/train/$TRAIN_JOB/checkpoints" \
        -type d -name pretrained_model 2>/dev/null \
        | xargs -I {} stat -c '%Y {}' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')"
    if [ -z "$VLA_CKPT" ] || [ ! -d "$VLA_CKPT" ]; then
        err "Step 2 ok 이지만 latest checkpoint 못 찾음 (탐색: lerobot/outputs/train/$TRAIN_JOB/checkpoints/*/pretrained_model)"
        exit 1
    fi
    info "latest ckpt  : $VLA_CKPT"
fi

# ============================================================
# Step 3 — P_phase1 vector DB rebuild
# ============================================================
bold "Step 3/3 — P_phase1 DB rebuild"

if python -m method3.dct.rebuild_p_phase1 \
        --session "$SESSION_DIR" \
        --subdir dct \
        --vla-ckpt "$VLA_CKPT" \
        --dataset "$DATASET" \
        --skill-dct-parquet "$SKILL_DCT_PARQUET"; then
    green "Step 3 done — DB at $SESSION_DIR/dct/skill_wise_vector_db.npz"
else
    err "Step 3 failed (rebuild_p_phase1)"
    exit 1
fi

bold "Phase2 prep chain DONE"
green "next steps (manual):"
green "  1. cp $SESSION_DIR/dct/skill_wise_vector_db.npz grpc_server/buffer/server_skill_wise_vector_db.npz"
green "  2. edit pipeline_config/phase2_config.yaml:"
green "       phase1_trained_vla_path: $VLA_CKPT"
green "       phase1_dataset_path:     $DATASET"
green "       selector.skill_dct_parquet: $SKILL_DCT_PARQUET"
green "  3. restart server: bash grpc_server/launch_remote_server.sh"
green "  4. run_forward_and_reset_ws3.sh 의 PHASE 변경: PHASE=\"phase1\" → PHASE=\"phase2\""
green "     그 후 bash run_forward_and_reset_ws3.sh 실행 → Phase2 cycle 의 새 episode 는"
green "     session/phase2/ 하위에 저장됨."
