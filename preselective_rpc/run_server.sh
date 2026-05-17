#!/bin/bash
# Preselective RPC Server Runner (전용 conda 환경)
#
# 전용 conda 환경(기본 gpu_server, Python 3.12)으로 preselective_rpc gRPC
# 서버를 띄운다. 실제 셋업/활성화/실행은 공식 run_h100_server.sh 에 위임하며,
# 아래 핵심 설정만 인자로 넘긴다. 해당 환경이 없으면 setup 단계에서 새로
# 생성된다 (기존 lerobot 환경은 건드리지 않음).
#
# 실행:  bash preselective_rpc/run_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# 핵심 설정 (Essential Configuration)
# ============================================================

# 아래 값들은 모두 호출 시 환경변수로 덮어쓸 수 있다.
#   예) GPU_ID=3 ENV_NAME=my_env bash preselective_rpc/run_server.sh

# [필수] 사용할 conda 환경 — 없으면 Python 3.12로 새로 생성됨
ENV_NAME="${ENV_NAME:-gpu_server}"

# [필수] 사용할 GPU → CUDA_VISIBLE_DEVICES 로 전달.
#        여러 장은 "0,1" 처럼 콤마로 지정.
GPU_ID="${GPU_ID:-3}"

# [필수] gRPC 바인드 주소 / 포트
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50061}"

# [필수] recording config — server 가 preselective_filter +
#        perturbation.skill 섹션을 읽어들이는 yaml
RECORDING_CONFIG="${RECORDING_CONFIG:-pipeline_config/recording_config_ws3.yaml}"

# [필수] curobo backend 가 사용하는 URDF
URDF="${URDF:-assets/urdf/so101_robot4.urdf}"

# [선택] 1 이면 환경 셋업 단계를 건너뜀 (이미 구성된 환경 빠른 재시작)
SKIP_SETUP="${SKIP_SETUP:-0}"

# ============================================================
# 실행 — run_h100_server.sh 에 위임
# ============================================================

export ENV_NAME HOST PORT RECORDING_CONFIG URDF SKIP_SETUP
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec bash "$SCRIPT_DIR/run_h100_server.sh"
