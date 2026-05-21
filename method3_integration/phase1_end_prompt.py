"""Phase1 종료 후 사용자 대화형 prompt (3-option).

execution_forward_and_reset.py main() 끝에서 호출. Phase1 early termination
이 발생했거나 attempted episode 수에 도달한 경우 사용자에게:

  (1) Phase2 prep chain 실행 → 폴더 reorg + chain + 절대경로 출력 + exit
  (2) Resume mode 재실행 (실패 episode 만 채움; 실패 있을 때만 노출)
  (3) 추가 Phase1 data 수집 (사용자 입력 N) → resume + num_episodes+=N 재실행

(2), (3) 은 subprocess 로 self 재호출 후 본 prompt loop *다시 진입* — chain
또는 사용자 skip 까지 반복.

폴더 reorg: chain 진입 직전 ``session/episode_*`` 를 ``session/phase1/`` 로
mv. chain 산출물 절대경로 출력 — skill DCT parquet / VLA ckpt / P_phase1 DB
/ session/phase1/ / session/phase2/ (앞으로 생성).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _read_int(prompt: str, *, min_value: int = 1) -> int | None:
    """정수 입력. EOF/Ctrl+C → None."""
    try:
        raw = input(prompt).strip()
        if not raw:
            return None
        n = int(raw)
        return n if n >= min_value else None
    except (EOFError, KeyboardInterrupt, ValueError):
        return None


def _episode_counts(session_dir: Path) -> tuple[int, int]:
    """Return (captured, attempted) by inspecting session metadata.

    captured  = success episode 수 (folder existence + skill_summary.json 의 success).
    attempted = 시도 episode 수 (== max episode_num seen).
    fallback: episode_* 폴더 수 = captured 로 가정. attempted 는 same 면 100% success
    가정 (구버전 호환).
    """
    import json
    # episode 폴더는 session_dir/episode_* 또는 session_dir/phase1/episode_*.
    ep_dirs: list[Path] = []
    ep_dirs.extend(sorted(session_dir.glob("episode_*")))
    ep_dirs.extend(sorted(session_dir.glob("phase1/episode_*")))
    if not ep_dirs:
        return 0, 0
    nums = []
    captured = 0
    for ep in ep_dirs:
        if not ep.is_dir():
            continue
        try:
            n = int(ep.name.split("_")[-1])
            nums.append(n)
        except (ValueError, IndexError):
            continue
        # 성공 판정: session_summary 의 forward_judge_true 카운트 또는 episode_*/eval.json
        # 간단하게는 episode_*/skill_summary.json 의 forward.judge_true 검사.
        ok = False
        for cand in ("skill_summary.json", "eval.json", "forward_summary.json"):
            p = ep / cand
            if p.exists():
                try:
                    j = json.loads(p.read_text())
                    # 여러 schema 호환
                    v = j.get("forward_judge_true") if isinstance(j, dict) else None
                    if v is None and isinstance(j, dict):
                        v = j.get("forward", {}).get("judge_true")
                    if v is True or v == 1:
                        ok = True
                        break
                except Exception:
                    continue
        if ok:
            captured += 1
    attempted = max(nums) if nums else 0
    # captured 가 0 일 때: schema 없으면 *folder 수* 를 captured 로 fallback (모두 성공
    # 가정 — boundary check 가 ready 였으므로 high success 일 확률 큼).
    if captured == 0 and ep_dirs:
        captured = len(ep_dirs)
    return captured, attempted


def run_prompt_loop(
    *,
    session_dir: Path,
    dataset_repo_id: str,
    stop_reason: str,
    project_root: Path,
    self_argv: list[str],
) -> None:
    """Phase1 종료 후 대화형 prompt loop.

    Args:
        session_dir: 현재 session 경로 (절대 또는 상대).
        dataset_repo_id: lerobot HF repo id (e.g., CoRL2026-CSI/...).
        stop_reason: phase1_readiness_hook.stop_reason (e.g., phase1_ready_and_gain).
        project_root: AutoDataCollector repo root.
        self_argv: argv (sys.argv 그대로) — (2)/(3) subprocess re-launch 시 base.
    """
    session_dir = Path(session_dir).resolve()
    chain_script = project_root / "scripts" / "phase2_prep_chain.sh"
    if not chain_script.exists():
        print(f"  [prompt] chain script not found: {chain_script}")
        return

    while True:
        captured, attempted = _episode_counts(session_dir)
        has_failures = attempted > captured and attempted > 0
        rate = (captured / attempted * 100.0) if attempted > 0 else 0.0

        print()
        print(CYAN + "=" * 70 + RESET)
        print(CYAN + BOLD + f"  Phase1 early termination (reason={stop_reason})" + RESET)
        print(f"  session : {session_dir}")
        print(f"  dataset : {dataset_repo_id}")
        if attempted > 0:
            color = RED if has_failures else GREEN
            failed = attempted - captured
            print(
                f"  {color}{captured}/{attempted} episodes captured "
                f"({failed} failed) — {rate:.1f}% success{RESET}"
            )
        print(CYAN + "=" * 70 + RESET)
        print(f"  (1) Phase2 prep chain")
        if has_failures:
            print(f"  (2) Resume — fill {attempted - captured} failed episodes")
        print(f"  (3) Collect more Phase1 episodes (enter N)")
        try:
            ans = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  → exit")
            return

        if ans == "1":
            # (1) chain: chain script 가 Step 0 에서 폴더 reorg 까지 담당
            # (phase2_prep_chain.sh Step 0). 여기선 chain 호출만.
            print(f"  → launching {chain_script.name} (Step 0 reorg + Step 1-3) ...")
            rc = subprocess.call([
                "bash", str(chain_script),
                "--dataset", dataset_repo_id,
                "--session-dir", str(session_dir),
            ])
            if rc != 0:
                print(f"  {RED}[prep chain] exit code {rc} — Phase2 prep INCOMPLETE{RESET}")
                return
            # chain script 의 마지막 출력이 산출물 절대경로 + manual next-steps
            # (DB upload / yaml edit / server restart / PHASE 변경) 를 모두 안내.
            print(f"  {GREEN}[prep chain] DONE — 위 chain 출력의 artifact 경로 / next-steps 참고{RESET}")
            return

        if ans == "2":
            if not has_failures:
                print(f"  {YELLOW}(2) unavailable — no failed episodes{RESET}")
                continue
            # (2) resume — subprocess self re-launch with --resume
            print(f"  → relaunching with --resume {session_dir} ...")
            new_argv = _patch_resume_argv(self_argv, str(session_dir))
            rc = subprocess.call(new_argv)
            if rc != 0:
                print(f"  {RED}[resume] exit code {rc}{RESET}")
                return
            print(f"  {GREEN}[resume] done — re-checking termination state{RESET}")
            # loop 진입 — re-check readiness/counts

        elif ans == "3":
            # (3) more — 사용자 입력 N → done + N
            n = _read_int(
                f"  {YELLOW}How many additional Phase1 episodes to collect? (enter integer ≥ 1){RESET}\n  > ",
                min_value=1,
            )
            if n is None:
                print(f"  {YELLOW}invalid input — back to prompt{RESET}")
                continue
            target = captured + n
            print(f"  → relaunching with --resume {session_dir} --num-episodes {target} (done={captured} + {n}) ...")
            new_argv = _patch_resume_argv(self_argv, str(session_dir), num_episodes=target)
            rc = subprocess.call(new_argv)
            if rc != 0:
                print(f"  {RED}[more] exit code {rc}{RESET}")
                return
            print(f"  {GREEN}[more] done — re-checking termination state{RESET}")
            # loop 진입

        else:
            print(f"  {YELLOW}invalid choice — type 1, 2, or 3{RESET}")


def _patch_resume_argv(
    argv: list[str],
    session_dir: str,
    *,
    num_episodes: int | None = None,
) -> list[str]:
    """argv 를 --resume <session_dir> 로 변경한 새 argv 반환.

    원본 argv 에 --resume 있으면 그 값을 교체, 없으면 추가.
    num_episodes 지정 시 --num-episodes 도 교체/추가.
    sys.executable 을 [0] 으로 prepend — `python execution_forward_and_reset.py ...` 호출 패턴.
    """
    out = [sys.executable]
    skip = False
    has_resume = False
    has_num = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--resume":
            out.append(tok)
            out.append(session_dir)
            has_resume = True
            skip = True
            continue
        if tok.startswith("--resume="):
            out.append(f"--resume={session_dir}")
            has_resume = True
            continue
        if num_episodes is not None:
            if tok == "--num-episodes":
                out.append(tok)
                out.append(str(num_episodes))
                has_num = True
                skip = True
                continue
            if tok.startswith("--num-episodes="):
                out.append(f"--num-episodes={num_episodes}")
                has_num = True
                continue
        out.append(tok)
    if not has_resume:
        out.extend(["--resume", session_dir])
    if num_episodes is not None and not has_num:
        out.extend(["--num-episodes", str(num_episodes)])
    return out
