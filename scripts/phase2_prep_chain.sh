#!/usr/bin/env bash
# ============================================================
# phase2_prep_chain.sh — Phase1 종료 후 Phase2 사전 준비 chain (Step 0-4)
#
# Phase1 early termination 이 발생하면 execution_forward_and_reset.py 의
# main() 끝에서 prompt 후 본 스크립트를 호출한다. 수동 호출도 가능:
#
#   bash scripts/phase2_prep_chain.sh \
#       --dataset CoRL2026-CSI/pnp_ours_100_table1 \
#       --session-dir results/session_20260522_044832
#
# Step 0  session 폴더 reorg (episode_* → phase1/, phase2/ mkdir)
# Step 1  skill segment DCT parquet build
# Step 2  DCT-tuned VLA 학습 (--skip-train 으로 생략 가능)
# Step 3  P_phase1 vector DB rebuild
# Step 4  artifact 배치 — DB → grpc_server/buffer/ cp + phase2_config.yaml
#         3-key (ckpt/dataset/skill_dct_parquet) 자동 edit
#
# 옵션:
#   --skip-train      : Step 2 (VLA 학습) 건너뛰기. --vla-ckpt 명시 필요.
#   --vla-ckpt PATH   : Step 2 skip 시 Step 3 의 vla checkpoint 지정.
#   --train-job NAME  : Step 2 JOB_NAME override (default smolvla_dct_<ts>).
#   --steps N         : Step 2 STEPS override (default = train script 의 epoch 환산).
#
# 각 step exit code 검증 — fail 시 chain 즉시 중단. chain 종료 후 manual
# next-steps (git commit+push → launch_remote_server.sh → PHASE 변경+run) 출력.
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
            sed -n '3,27p' "$0" | sed 's/^# \?//'
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
bold "Step 0/4 — session 폴더 reorg (episode_* → phase1/)"
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
bold "Step 1/4 — build skill segment DCT parquet"

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
    bold "Step 2/4 — VLA training SKIPPED (--skip-train)"
    if [ -z "$VLA_CKPT" ]; then
        err "--skip-train requires --vla-ckpt PATH"; exit 2
    fi
    if [ ! -d "$VLA_CKPT" ]; then
        err "vla ckpt not found: $VLA_CKPT"; exit 2
    fi
    info "using existing ckpt: $VLA_CKPT"
else
    bold "Step 2/4 — DCT-tuned VLA training"
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
bold "Step 3/4 — P_phase1 DB rebuild"

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

# ============================================================
# Step 4 — artifact 자동 배치 (DB cp + phase2_config.yaml 3-key edit)
# ============================================================
# 수동 cp / edit 의 누락·오타를 없애고, "yaml 을 서버로 보내는" 경로를
# 명확히 한다. yaml 은 git-tracked → git commit+push 후 launch_remote_server.sh
# 의 git pull 로 서버 도착. (sync_artifacts.sh 는 ckpt/DB/parquet 만 rsync —
# yaml 은 sync 대상 아님. 그래서 git push 가 *반드시* server restart 전.)
bold "Step 4/4 — artifact 배치 (DB cp + yaml edit)"

PHASE2_YAML="$PROJ_ROOT/pipeline_config/phase2_config.yaml"
DB_SRC="$SESSION_DIR/dct/skill_wise_vector_db.npz"
DB_DST="$PROJ_ROOT/grpc_server/buffer/server_skill_wise_vector_db.npz"

# (a) DB → grpc_server/buffer/ (sync_artifacts.sh 의 rsync source 위치)
mkdir -p "$(dirname "$DB_DST")"
cp "$DB_SRC" "$DB_DST"
info "DB copied  : $DB_SRC → $DB_DST"

# (b) phase2_config.yaml 3-key value-line 교체. repo-relative path 로 저장
#     (local/remote portable — server.py 가 repo root 기준 resolve).
_ckpt_rel="${VLA_CKPT#"$PROJ_ROOT/"}"
_parq_rel="${SKILL_DCT_PARQUET#"$PROJ_ROOT/"}"
python - "$PHASE2_YAML" "$_ckpt_rel" "$DATASET" "$_parq_rel" <<'PY'
import re, sys
yaml_path, ckpt, dataset, parquet = sys.argv[1:5]
lines = open(yaml_path, encoding="utf-8").read().splitlines(keepends=True)
def _patch(lines, key, value, indent=""):
    pat = re.compile(rf'^{re.escape(indent)}{re.escape(key)}:\s*"[^"]*"(.*)$')
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if m:
            lines[i] = f'{indent}{key}: "{value}"{m.group(1)}\n'
            return True
    return False
ok1 = _patch(lines, "phase1_trained_vla_path", ckpt)
ok2 = _patch(lines, "phase1_dataset_path", dataset)
ok3 = _patch(lines, "skill_dct_parquet", parquet, indent="  ")
open(yaml_path, "w", encoding="utf-8").writelines(lines)
print(f"  yaml patched: vla={ok1} dataset={ok2} skill_dct_parquet={ok3}")
PY
green "Step 4 done — DB 배치 + yaml 3-key 갱신 완료"

bold "Phase2 prep chain DONE"
green "=== 산출물 (절대경로) ==="
green "  skill DCT parquet : $SKILL_DCT_PARQUET"
green "  VLA checkpoint    : $VLA_CKPT"
green "  P_phase1 DB       : $DB_SRC"
green "  → server buffer   : $DB_DST"
green "  session/phase1/   : $SESSION_DIR/phase1/"
green "  session/phase2/   : $SESSION_DIR/phase2/  (Phase2 cycle episode 저장 위치)"
green ""
green "=== manual next-steps (순서 고정 — yaml 은 git 으로만 서버 도달) ==="
green "  (1) git add pipeline_config/phase2_config.yaml && git commit -m '...' && git push"
green "      → origin 갱신. *이게 없으면* 다음 step 의 git pull 이 예전 yaml 받음."
green "  (2) bash grpc_server/launch_remote_server.sh"
green "      → 서버 git pull (origin→server, 최신 yaml) + sync_artifacts (ckpt/DB/parquet"
green "        rsync) + server restart. (1) 을 건너뛰면 서버가 예전 ckpt load."
green "  (3) run_forward_and_reset_ws3.sh 의 PHASE: \"phase1\" → \"phase2\" 후 실행."
green "      → Phase2 cycle 의 새 episode 는 session/phase2/ 하위에 저장."
