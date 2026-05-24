#!/usr/bin/env bash
# server tmux 의 EE-delta 관련 핵심 log capture
REMOTE_HOST="${REMOTE_HOST:-csiwoo@61.107.202.230}"
env -u LD_LIBRARY_PATH -u LD_PRELOAD ssh -o ConnectTimeout=10 "$REMOTE_HOST" "
    echo '=== startup log (URDF / EE delta) ==='
    tmux capture-pane -p -t phase2_server_pnp -S -2000 | grep -iE 'EE delta DCT|urdf|listening on|Phase2MISelector ready' | tail -10
    echo
    echo '=== fail mode (FK / exception / traceback) ==='
    tmux capture-pane -p -t phase2_server_pnp -S -1000 | grep -iE 'FK failed|exception|traceback|broadcast' | tail -20
    echo
    echo '=== 가장 최근 plan_batch ==='
    tmux capture-pane -p -t phase2_server_pnp -S -300 | grep -iE 'plan_batch req|curobo returned|candidate.*select failed' | tail -10
"
