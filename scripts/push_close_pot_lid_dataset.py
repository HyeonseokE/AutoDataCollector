"""Push close_pot_lid dataset to HF Hub under a custom repo_id.

Loads the locally recorded dataset from
  lerobot/outputs/datasets/CoRL2026-CSI/close_pot_lid/
and pushes it to
  https://huggingface.co/datasets/CoRL2026-CSI/SO101-teleop_close_pot_lid_100epi
with a generated LeRobot dataset card and a v3.0 codebase tag.
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOCAL_ROOT = Path("lerobot/outputs/datasets/CoRL2026-CSI/close_pot_lid").resolve()
TARGET_REPO_ID = "CoRL2026-CSI/SO101-teleop_close_pot_lid_100epi"


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"Local dataset root not found: {LOCAL_ROOT}")

    ds = LeRobotDataset(repo_id=TARGET_REPO_ID, root=LOCAL_ROOT)
    print(f"Loaded: {len(ds)} frames across {ds.meta.total_episodes} episodes")
    print(f"Pushing to: https://huggingface.co/datasets/{TARGET_REPO_ID}")

    ds.push_to_hub(
        tags=["LeRobot", "so101", "teleoperation", "close-pot-lid"],
        license="apache-2.0",
        push_videos=True,
        private=False,
        upload_large_folder=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
