"""Push pnp_ours_100 forward dataset to HF Hub under public repo name.

Loads the locally finalized LeRobotDataset cache from
  /home/lerobot/.cache/huggingface/lerobot/CoRL2026-CSI/pnp_ours_100_table1/
and pushes it to
  https://huggingface.co/datasets/CoRL2026-CSI/SO101-PickandPlace_Ours_100epi

Forward only — the reset companion (`pnp_ours_100_table1_reset`) is intentionally
not uploaded.

Prerequisite:
    huggingface-cli login          # or: export HF_TOKEN="hf_..."
Token must have write access to the `CoRL2026-CSI` organization.

Usage:
    cd /home/lerobot/AutoDataCollector && conda activate lerobot
    python scripts/push_pnp_ours_dataset.py
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOCAL_ROOT = Path(
    "/home/lerobot/.cache/huggingface/lerobot/CoRL2026-CSI/pnp_ours_100_table1"
).resolve()
TARGET_REPO_ID = "CoRL2026-CSI/SO101-PickandPlace_Ours_100epi"


def main() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"Local dataset root not found: {LOCAL_ROOT}")

    ds = LeRobotDataset(repo_id=TARGET_REPO_ID, root=LOCAL_ROOT)
    print(f"Loaded: {len(ds)} frames across {ds.meta.total_episodes} episodes")
    print(f"Pushing to: https://huggingface.co/datasets/{TARGET_REPO_ID}")

    ds.push_to_hub(
        tags=["LeRobot", "so101", "pick-and-place", "method3", "phase2"],
        license="apache-2.0",
        push_videos=True,
        private=False,
        upload_large_folder=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
