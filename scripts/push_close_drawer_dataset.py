"""Push close_drawer dataset (Phase1+Phase2 combined) to HF Hub.

Source : ~/.cache/huggingface/lerobot/CoRL2026/table1/ours_close_drawer  (100 epi)
Target : https://huggingface.co/datasets/CoRL2026-CSI/SO101-CloseDrawer_Ours_100epi

Auto-generated LeRobot dataset card + v3.0 codebase tag.
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOCAL_ROOT = Path.home() / ".cache/huggingface/lerobot/CoRL2026/table1/ours_close_drawer"
TARGET_REPO_ID = "CoRL2026-CSI/SO101-CloseDrawer_Ours_100epi"


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"Local dataset root not found: {LOCAL_ROOT}")

    ds = LeRobotDataset(repo_id=TARGET_REPO_ID, root=LOCAL_ROOT)
    print(f"Loaded: {len(ds)} frames across {ds.meta.total_episodes} episodes")
    print(f"Pushing to: https://huggingface.co/datasets/{TARGET_REPO_ID}")

    ds.push_to_hub(
        tags=["LeRobot", "so101", "close-drawer", "phase1-phase2"],
        license="apache-2.0",
        push_videos=True,
        private=False,
        upload_large_folder=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
