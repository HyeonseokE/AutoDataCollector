"""
Dataset Cleanup for Resume Mode

resume 시 재취득 대상 에피소드를 데이터셋에서 제거하여
파이프라인 결과와 데이터셋의 1:1 정합성을 유지합니다.

데이터셋에는 judge=="TRUE"인 에피소드만 저장되므로,
성공 에피소드의 파이프라인 폴더가 삭제되어 재취득 대상이 되면
해당 데이터셋 에피소드도 함께 제거해야 합니다.
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Optional

LEROBOT_PATH = Path(__file__).parent.parent / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

# Module-level ANSI color — dataset add/remove events 는 BLUE.
# 모듈 top 에 정의해 함수 어디서든 NameError 없이 참조. 충돌 회피용 `_C_` prefix.
_C_DS = "\033[94m"     # BLUE — dataset count change
_C_END = "\033[0m"


def _dprint(*args, sep: str = " ", end: str = "\n") -> None:
    """Dataset-event 전용 print — 본문 전체를 BLUE 로 감싼다.

    cleanup.py 의 모든 print 가 dataset add/remove/trim 이벤트라 일괄 BLUE.
    multi-arg / multi-line concat 도 sep 으로 join 후 한 번에 색 입힘 — 색 코드가
    중간에 끊겨 회색 공백이 보이는 일이 없다.
    """
    msg = sep.join(str(a) for a in args)
    print(f"{_C_DS}{msg}{_C_END}", end=end)


def cleanup_dataset_for_resume(
    session_dir: str,
    repo_id: str,
    root: Optional[str] = None,
) -> dict:
    """resume 전 데이터셋에서 재취득 대상 에피소드를 제거.

    매핑 로직:
    - 데이터셋에는 judge=="TRUE"인 에피소드만 순서대로 저장됨
    - 폴더가 삭제된 에피소드는 원래 TRUE/FALSE를 알 수 없으므로,
      데이터셋의 실제 에피소드 수를 기준으로 역추적

    순회 방식:
    1. ep1부터 순서대로 스캔하며 각 에피소드의 상태 파악
       - TRUE (유지) / FALSE (재취득, dataset 미기록) / 삭제됨 (재취득, 불명)
    2. dataset에 기록된 에피소드 수만큼 "TRUE 또는 삭제됨"에 dataset idx를 순서대로 할당
    3. 삭제된 에피소드에 할당된 dataset idx = 삭제 대상
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    session_path = Path(session_dir)
    dataset_path = Path(root) if root else HF_LEROBOT_HOME / repo_id

    stats = {
        "pipeline_episodes_total": 0,
        "pipeline_episodes_success": 0,
        "pipeline_episodes_to_rerun": 0,
        "dataset_episodes_before": 0,
        "dataset_episodes_after": 0,
        "deleted_indices": [],
        # episode lifecycle — 생존(폴더 존재 + judge=TRUE) 에피소드 번호 목록.
        # None = 미계산(early-return). resume 시 subgoal buffer reconcile 에 쓴다.
        "kept_true_episodes": None,
    }

    # 1. 파이프라인 에피소드 스캔 (dataset 검사보다 먼저 — buffer reconcile 은
    # dataset 상태와 무관하게 kept_true_episodes 정보를 요구하므로 broken/missing
    # dataset 으로 early-exit 하기 전에 반드시 채워둔다).
    #   "kept_true"  = TRUE + 폴더 존재 → dataset에 기록됨, 유지
    #   "false"      = FALSE + 폴더 존재 → dataset에 미기록, 재취득
    #   "missing"    = 폴더는 존재하지만 batch_info.json 없음 → 불완전 episode, 재취득
    #
    # NOTE: ep_states 는 sorted(glob("episode_*")) 의 *실제 존재하는* 폴더만
    # 순회한다. 과거 코드는 range(1, max+1) 로 빈 번호도 missing 으로 잡아
    # 그 자리 dataset idx 를 삭제했으나, episodes_per_seed 마이그레이션
    # (예: 30/10 → 100/10, phase1→phase2 확장) 후엔 빈 번호(ep04~10 등)가
    # 정상 상태이므로 잘못 삭제되는 문제가 있었다. sorted 폴더 순서가 dataset
    # idx 0,1,2,… 와 1:1 정렬되므로 폴더 기반 스캔이 옳다.
    #
    # 부수효과: 사용자가 episode_NN 폴더를 직접 삭제했을 때 그 자리 dataset
    # idx 가 자동 정리되던 동작은 사라진다. 정상 워크플로우에선 batch_info.
    # judge="FALSE" 마킹으로 재취득을 표현하므로 영향 거의 없음. 그래도 폴더
    # 직접 삭제로 dataset 청소가 필요하면 별도 CLI 로 분리할 것.
    # 폴더명 = execution / save 순서 (seed_major / round_robin 둘 다).
    # sorted-by-name 순회 → dataset save 순서와 1:1 매칭.
    # chain reorg 이후 episode 는 phase1/ / phase2/ 하위로 이동되므로 session
    # 직속 glob 만으로는 *0개로 잡혀* dataset trim 이 skip 되던 버그가 있었다.
    # execution_forward_and_reset._iter_episode_dirs 와 동일한 식으로 legacy
    # (직속) + phase1/ + phase2/ 를 모두 수집한다. phase1 ep_01..40 다음
    # phase2 ep_41.. 순서가 sorted-by-name 과 일치하므로 dataset idx 매핑이
    # 깨지지 않는다.
    _ep_legacy = sorted(session_path.glob("episode_*"))
    _ep_ph1 = sorted(session_path.glob("phase1/episode_*"))
    _ep_ph2 = sorted(session_path.glob("phase2/episode_*"))
    episode_dirs = [p for p in (_ep_legacy + _ep_ph1 + _ep_ph2) if p.is_dir()]
    ep_states = []  # [(ep_num, state)]
    for ep_dir in episode_dirs:
        try:
            ep_num = int(ep_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        batch_info_path = ep_dir / "batch_info.json"
        if not batch_info_path.exists():
            ep_states.append((ep_num, "missing"))
            stats["pipeline_episodes_to_rerun"] += 1
        else:
            with open(batch_info_path) as f:
                bi = json.load(f)
            judge = bi.get("judge", "")
            if judge == "TRUE":
                ep_states.append((ep_num, "kept_true"))
                stats["pipeline_episodes_success"] += 1
            else:
                ep_states.append((ep_num, "false"))
                stats["pipeline_episodes_to_rerun"] += 1
    stats["pipeline_episodes_total"] = len(ep_states)
    # episode lifecycle — 생존 에피소드(폴더 존재 + judge=TRUE). resume 시
    # subgoal buffer reconcile 에 사용 (삭제된 에피소드의 stale entry 제거).
    # 모든 early-return 경로 직전에 채워져 있어야 buffer 가 dataset 상태와
    # 정합 유지. 사용자가 episode 폴더를 다 삭제하면 [] 가 돼서 reconcile 이
    # 옛 tagged entries 를 전부 정리한다.
    stats["kept_true_episodes"] = [n for n, st in ep_states if st == "kept_true"]

    # 2. 데이터셋 존재 확인 및 열기
    if not dataset_path.exists():
        _dprint("[Cleanup] No dataset found, skipping")
        return stats

    # 불완전 데이터셋 감지: meta/episodes/ parquet이 없으면 finalize() 미호출 상태
    # 전체 삭제 대신 finalize를 시도하여 기존 데이터 보존
    episodes_meta_dir = dataset_path / "meta" / "episodes"
    if dataset_path.exists() and not episodes_meta_dir.exists():
        _dprint(f"[Cleanup] Incomplete dataset detected (no meta/episodes/) — attempting recovery...")
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
            dataset.meta._close_writer()  # flush metadata buffer → create meta/episodes/
            dataset._close_writer()
            _dprint(f"[Cleanup] Recovery successful: {dataset.meta.total_episodes} episodes recovered")
        except Exception as e:
            _dprint(f"[Cleanup] Recovery failed ({e}) — removing dataset")
            shutil.rmtree(dataset_path)
            return stats

    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        actual_dataset_episodes = dataset.meta.total_episodes
        stats["dataset_episodes_before"] = actual_dataset_episodes
    except Exception as e:
        _dprint(f"[Cleanup] Warning: Cannot open dataset, removing: {e}")
        shutil.rmtree(dataset_path)
        return stats

    # early-return 경로에서도 dataset 변경 없음을 정확히 표시 — 디폴트 0 으로
    # 남아 "N → 0" 처럼 데이터셋 손실로 오해되지 않도록 dataset_episodes_after 를
    # dataset_episodes_before 와 같게 유지.
    stats["dataset_episodes_after"] = actual_dataset_episodes

    if actual_dataset_episodes == 0:
        _dprint("[Cleanup] Dataset is empty, skipping")
        return stats

    if not episode_dirs:
        _dprint("[Cleanup] No episodes found in session, skipping dataset trim")
        return stats

    # 3. dataset index 매핑
    #    dataset에는 TRUE 에피소드만 순서대로 저장됨.
    #    "kept_true"와 "missing" 중 원래 TRUE였던 것들이 dataset idx를 차지함.
    #    "false"는 discard되어 dataset에 없음.
    #
    #    방법: "false"가 아닌 에피소드(=TRUE였거나 삭제됨)에 순서대로
    #    dataset idx를 할당. 총 할당 수가 actual_dataset_episodes와 일치해야 함.
    could_be_in_dataset = [(ep_num, state) for ep_num, state in ep_states
                           if state != "false"]

    # 할당 가능한 수가 실제 dataset 에피소드 수보다 적으면 → 데이터 불일치
    if len(could_be_in_dataset) < actual_dataset_episodes:
        _dprint(f"[Cleanup] Warning: dataset has {actual_dataset_episodes} episodes but "
              f"only {len(could_be_in_dataset)} non-FALSE pipeline episodes found")
        # 초과분은 뒤에서 자르기 (이전 불완전 resume 잔여물)
        excess_start = len(could_be_in_dataset)
        excess_indices = list(range(excess_start, actual_dataset_episodes))
        if excess_indices:
            _dprint(f"[Cleanup] Trimming {len(excess_indices)} excess episodes from dataset tail")

    # 앞에서부터 actual_dataset_episodes개만 할당
    dataset_indices_to_delete = []
    assigned = 0
    for ep_num, state in could_be_in_dataset:
        if assigned >= actual_dataset_episodes:
            break
        if state == "missing":
            # 삭제된 에피소드 → 원래 TRUE였고 dataset에 기록됨 → 삭제 대상
            dataset_indices_to_delete.append(assigned)
            _dprint(f"  ep{ep_num:02d} (deleted folder) → dataset idx {assigned} → DELETE")
        else:
            # kept_true → 유지
            _dprint(f"  ep{ep_num:02d} (TRUE, kept)     → dataset idx {assigned} → keep")
        assigned += 1

    # 초과분 처리 (dataset에 pipeline보다 많은 에피소드가 있는 경우)
    if actual_dataset_episodes > assigned:
        excess_indices = list(range(assigned, actual_dataset_episodes))
        dataset_indices_to_delete.extend(excess_indices)
        _dprint(f"[Cleanup] {len(excess_indices)} excess episodes at tail → DELETE (indices {excess_indices})")

    # Corrupted-episode guard — video 프레임 수 ≠ metadata length 인 episode 감지.
    # delete_episodes() 는 *살아남는* episode 를 video reindex 할 때
    # `src_ep["length"] == to_frame - from_frame` 를 assert 한다. corrupted
    # episode 가 survivor 에 있으면 그 한 개 때문에 cleanup 전체가 crash.
    # corrupted episode 는 어차피 reindex 불가 → 삭제 목록에 강제 편입한다
    # (delete 대상은 reindex 를 안 거치므로 crash 회피).
    try:
        import pandas as _pd
        import glob as _glob
        _ep_files = sorted(_glob.glob(
            str(dataset_path / "meta" / "episodes" / "chunk-*" / "*.parquet")
        ))
        if _ep_files:
            _edf = _pd.concat([_pd.read_parquet(f) for f in _ep_files],
                              ignore_index=True)
            _fps = dataset.meta.fps
            _corrupt = []
            for _, _r in _edf.iterrows():
                _idx = int(_r["episode_index"])
                _length = int(_r["length"])
                for _vk in dataset.meta.video_keys:
                    _t0 = _r.get(f"videos/{_vk}/from_timestamp")
                    _t1 = _r.get(f"videos/{_vk}/to_timestamp")
                    if _t0 is not None and _t1 is not None:
                        _vf = round((float(_t1) - float(_t0)) * _fps)
                        if _vf != _length:
                            _corrupt.append(_idx)
                            break
            _corrupt_new = [i for i in _corrupt
                            if i not in dataset_indices_to_delete]
            if _corrupt_new:
                print(f"[Cleanup] {len(_corrupt_new)} corrupted episode(s) "
                      f"(video frames ≠ metadata length) → force-deleting: "
                      f"{_corrupt_new}")
                dataset_indices_to_delete.extend(_corrupt_new)
    except Exception as _e:
        print(f"[Cleanup] corrupted-episode scan skipped: {_e}")

    dataset_indices_to_delete = sorted(set(dataset_indices_to_delete))
    stats["deleted_indices"] = sorted(dataset_indices_to_delete)

    if not dataset_indices_to_delete:
        _dprint(f"[Cleanup] Dataset is clean ({actual_dataset_episodes} episodes), no cleanup needed")
        stats["dataset_episodes_after"] = actual_dataset_episodes
        return stats

    _dprint(f"[Cleanup] Deleting {len(dataset_indices_to_delete)} episodes from dataset: {dataset_indices_to_delete}")

    # 4. 전부 삭제해야 하는 경우 → 데이터셋 디렉토리 자체 제거
    if len(dataset_indices_to_delete) >= actual_dataset_episodes:
        _dprint(f"[Cleanup] All episodes to be deleted, removing dataset entirely")
        del dataset
        shutil.rmtree(dataset_path)
        stats["dataset_episodes_after"] = 0
        return stats

    # 5. delete_episodes()로 정리된 데이터셋 생성
    from lerobot.datasets.dataset_tools import delete_episodes

    temp_dir = dataset_path.parent / f"{dataset_path.name}_cleanup_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=dataset_indices_to_delete,
            output_dir=temp_dir,
            repo_id=repo_id,
        )
        stats["dataset_episodes_after"] = new_dataset.meta.total_episodes
        _dprint(f"[Cleanup] New dataset: {new_dataset.meta.total_episodes} episodes")

        del dataset
        del new_dataset

        # 6. 안전하게 교체: old → backup, temp → original, backup 삭제
        backup_dir = dataset_path.parent / f"{dataset_path.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        dataset_path.rename(backup_dir)
        temp_dir.rename(dataset_path)
        shutil.rmtree(backup_dir)

        _dprint(f"[Cleanup] Dataset replaced successfully")

    except Exception as e:
        _dprint(f"[Cleanup] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return stats
