#!/usr/bin/env bash
# =============================================================================
#  Record the LIVE rerun viewer window → mp4, exactly at the angle you see.
#
#  rerun 의 인터랙티브 카메라 각도는 SDK 로 읽어올 수 없으므로, "지금 보고 있는
#  화면 그대로" 를 남기려면 화면 자체를 녹화하는 것이 정답이다. 이 스크립트는
#  rerun 뷰어 창을 자동으로 찾아 그 영역만 x11grab 으로 녹화한다 (번들 ffmpeg).
#
#  사용법:
#    1) run_forward_and_reset_ws3.sh 를 실행해 rerun 뷰어가 떠 있는 상태에서,
#    2) rerun 뷰어를 원하는 각도로 회전/줌 해 두고 (패널은 숨기면 더 깔끔),
#    3) 다른 터미널에서:
#         ./scripts/record_rerun_view.sh out.mp4
#       녹화가 시작된다. 그 각도 그대로 누적되는 EE 궤적이 녹화됨.
#    4) Ctrl+C 로 종료 → ffmpeg 가 파일을 정상 마무리.
#
#  옵션(env):
#    DURATION=30     초 단위 자동 종료 (미지정 → Ctrl+C 까지)
#    REC_FPS=30      녹화 fps
#    REC_WINDOW='[Rr]erun'   창 제목 매칭 정규식
#    REC_REGION='WxH+X+Y'    창 자동탐색 대신 영역 직접 지정
#    REC_DISPLAY=:1  대상 X 디스플레이
# =============================================================================
set -uo pipefail

OUTPUT="${1:-rerun_view_$(date +%Y%m%d_%H%M%S).mp4}"
FPS="${REC_FPS:-30}"
DISPLAY_ID="${REC_DISPLAY:-${DISPLAY:-:1}}"
WIN_MATCH="${REC_WINDOW:-[Rr]erun}"
REGION="${REC_REGION:-}"
DURATION="${DURATION:-}"

# -------- resolve an ffmpeg with x11grab (bundled imageio_ffmpeg is known-good) --
FFMPEG="$(python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
if [ -z "$FFMPEG" ]; then FFMPEG="$(command -v ffmpeg || true)"; fi
if [ -z "$FFMPEG" ]; then
    echo "[record_rerun_view] ffmpeg 없음 (imageio_ffmpeg 또는 system ffmpeg 필요)" >&2
    exit 1
fi

# -------- determine capture geometry --------
if [ -n "$REGION" ]; then
    # parse WxH+X+Y
    W=${REGION%%x*}; rest=${REGION#*x}
    H=${rest%%+*}; rest=${rest#*+}
    X=${rest%%+*}; Y=${rest#*+}
else
    WID="$(xwininfo -root -tree 2>/dev/null | grep -iE "$WIN_MATCH" | grep -oE '0x[0-9a-f]+' | head -1)"
    if [ -n "${WID:-}" ]; then
        eval "$(xwininfo -id "$WID" | awk '
            /Absolute upper-left X/{x=$NF}
            /Absolute upper-left Y/{y=$NF}
            /^  Width/{w=$NF}
            /^  Height/{h=$NF}
            END{print "X="x" Y="y" W="w" H="h}')"
        echo "[record_rerun_view] rerun 창 발견 (id=$WID): ${W}x${H}+${X}+${Y}"
    else
        eval "$(xwininfo -root | awk '/^  Width/{w=$NF}/^  Height/{h=$NF}END{print "W="w" H="h}')"
        X=0; Y=0
        echo "[record_rerun_view] rerun 창 못 찾음 → 전체 화면 캡처 (${W}x${H}). REC_WINDOW/REC_REGION 로 지정 가능." >&2
    fi
fi

# ffmpeg 는 짝수 해상도 요구
W=$((W - W % 2)); H=$((H - H % 2))

TLIM=()
[ -n "$DURATION" ] && TLIM=(-t "$DURATION")

echo "[record_rerun_view] ${W}x${H}+${X},${Y} @ ${FPS}fps  display=${DISPLAY_ID}  → ${OUTPUT}"
echo "[record_rerun_view] Ctrl+C 로 종료하면 파일이 정상 마무리됩니다."

# x11grab. Ctrl+C(SIGINT) 시 ffmpeg 가 moov atom 을 써서 mp4 를 valid 하게 닫음.
exec "$FFMPEG" -y -hide_banner -loglevel warning \
    -f x11grab -framerate "$FPS" -video_size "${W}x${H}" -i "${DISPLAY_ID}+${X},${Y}" \
    "${TLIM[@]}" -pix_fmt yuv420p -c:v libx264 -preset veryfast -crf 18 \
    "$OUTPUT"
