"""Push the auto-collected cap_close_pot dataset to HF Hub.

Source (local):   ~/.cache/huggingface/lerobot/CoRL2026/Close_pot_10fps
Target (remote):  https://huggingface.co/datasets/CoRL2026-CSI/SO101-cap_close_pot_10fps
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOCAL_ROOT = Path("/home/lerobot3/.cache/huggingface/lerobot/CoRL2026/Close_pot_10fps").resolve()
TARGET_REPO_ID = "CoRL2026-CSI/SO101-cap_close_pot_10fps"


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"Local dataset root not found: {LOCAL_ROOT}")

    ds = LeRobotDataset(repo_id=TARGET_REPO_ID, root=LOCAL_ROOT)
    print(f"Loaded: {len(ds)} frames across {ds.meta.total_episodes} episodes")
    print(f"Pushing to: https://huggingface.co/datasets/{TARGET_REPO_ID}")

    ds.push_to_hub(
        tags=["LeRobot", "so101", "code-as-policies", "close-pot-lid", "auto-collected"],
        license="apache-2.0",
        push_videos=True,
        private=False,
        upload_large_folder=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
