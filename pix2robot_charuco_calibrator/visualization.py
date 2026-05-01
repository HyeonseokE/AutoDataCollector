"""
시각화 유틸 — 페이즈 배너 (터미널 + OpenCV 오버레이)

각 페이즈마다 색상이 다르게 표시되어 사용자가 현재 단계를 항상 인지할 수 있도록 함.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ANSI 색상 코드 (터미널)
ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "gray":   "\033[90m",
    "red":    "\033[31m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "blue":   "\033[34m",
    "magenta":"\033[35m",
    "cyan":   "\033[36m",
}

# 페이즈별 메타데이터: (이름, ANSI 색, OpenCV BGR 색)
PHASE_INFO = {
    1: ("BOARD SETUP",        "cyan",    (200, 200, 200)),
    2: ("CAMERA INTRINSICS",  "blue",    (255, 100, 50)),
    3: ("EXTRINSICS",         "green",   (50, 200, 50)),
    4: ("VERIFICATION",       "magenta", (200, 50, 200)),
}

# 페이즈 헤더 / 푸터 바 높이 — 이미지 외부에 패딩으로 추가됨.
# 마우스 콜백은 click y에서 PHASE_HEADER_HEIGHT 만큼 빼야 원본 이미지 좌표.
PHASE_HEADER_HEIGHT = 40
PHASE_FOOTER_HEIGHT = 28


def print_phase_banner(phase: int, total: int = 4, subtitle: str = ""):
    """터미널에 페이즈 시작 배너 출력."""
    name, ansi_color, _ = PHASE_INFO[phase]
    color = ANSI[ansi_color]
    bold = ANSI["bold"]
    reset = ANSI["reset"]

    title = f"[Phase {phase}/{total}] {name}"
    border = "═" * (len(title) + 4)

    print()
    print(f"{color}{bold}╔{border}╗{reset}")
    print(f"{color}{bold}║  {title}  ║{reset}")
    print(f"{color}{bold}╚{border}╝{reset}")
    if subtitle:
        print(f"{color}  {subtitle}{reset}")
    print()


def print_phase_summary(phase: int, success: bool, details: str = ""):
    """페이즈 종료 요약."""
    _, ansi_color, _ = PHASE_INFO[phase]
    color = ANSI[ansi_color]
    reset = ANSI["reset"]
    status = (
        f"{ANSI['green']}{ANSI['bold']}PASS{reset}" if success
        else f"{ANSI['red']}{ANSI['bold']}FAIL{reset}"
    )
    print(f"{color}  [Phase {phase}] {status}{reset} — {details}")
    print()


def print_session_summary(results: dict):
    """전체 4페이즈 끝난 후 요약."""
    bold = ANSI["bold"]
    reset = ANSI["reset"]

    print()
    print(f"{bold}╔══════════════════════════════════════════════╗{reset}")
    print(f"{bold}║   Charuco Calibration — Session Complete     ║{reset}")
    print(f"{bold}╚══════════════════════════════════════════════╝{reset}")
    for phase_num, info in results.items():
        name, ansi_color, _ = PHASE_INFO[phase_num]
        color = ANSI[ansi_color]
        status_color = ANSI["green"] if info.get("success") else ANSI["red"]
        status = "PASS" if info.get("success") else "FAIL"
        details = info.get("details", "")
        print(f"  {color}[Phase {phase_num}] {name:<22}{reset} "
              f"{status_color}{status}{reset}  {details}")
    print()


def draw_phase_overlay(
    image: np.ndarray,
    phase: int,
    progress_text: str = "",
    keys_text: str = "",
) -> np.ndarray:
    """
    이미지 위·아래에 페이즈 배너를 **추가** (overlay 아님 — 캔버스 확장).

    상단 패딩: [PHASE X/4] PHASE_NAME  |  progress_text  (페이즈 색상)
    하단 패딩: Keys: keys_text

    중요: 마우스 콜백은 click y 좌표에서 PHASE_HEADER_HEIGHT 만큼 빼야
    원본 이미지 좌표로 복원됨.
    """
    name, _, bgr_color = PHASE_INFO[phase]
    h, w = image.shape[:2]
    has_footer = bool(keys_text)
    top = PHASE_HEADER_HEIGHT
    bot = PHASE_FOOTER_HEIGHT if has_footer else 0

    canvas = np.zeros((h + top + bot, w, image.shape[2] if image.ndim == 3 else 1),
                      dtype=image.dtype)
    canvas[top:top + h, :] = image

    # 상단 헤더 바 (이미지 영역과 겹치지 않도록 top-1까지)
    cv2.rectangle(canvas, (0, 0), (w, top - 1), bgr_color, -1)
    header_text = f"[Phase {phase}/4]  {name}"
    if progress_text:
        header_text += f"   |   {progress_text}"
    cv2.putText(
        canvas, header_text, (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # 하단 키 가이드 바
    if has_footer:
        y0 = top + h
        cv2.rectangle(canvas, (0, y0), (w, y0 + bot), (40, 40, 40), -1)
        cv2.putText(
            canvas, f"Keys: {keys_text}", (10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
        )

    return canvas


def draw_status_text(
    image: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (10, 70),
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: bool = True,
) -> np.ndarray:
    """상태 메시지를 (반투명 배경 위에) 여러 줄로 표시."""
    out = image.copy()
    x, y = origin
    line_h = 22

    if bg and lines:
        max_w = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            for line in lines
        )
        overlay = out.copy()
        cv2.rectangle(
            overlay,
            (x - 6, y - 18),
            (x + max_w + 10, y + line_h * len(lines) - 16),
            (0, 0, 0), -1,
        )
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            out, line, (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out


def confirm_yn(prompt: str, default: bool = True) -> bool:
    """터미널 Y/N 프롬프트."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def prompt_float(prompt: str, default: Optional[float] = None) -> float:
    """터미널 float 입력. Enter면 default."""
    suffix = f" [default {default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  숫자를 입력해주세요.")


def prompt_int(prompt: str, default: Optional[int] = None) -> int:
    """터미널 int 입력."""
    suffix = f" [default {default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  정수를 입력해주세요.")
