"""Robot Executor for Subgoal Prediction Model.

Executes predicted subgoals on real robot hardware.

Usage:
    python -m models.subgoal_predictor.robot_executor \
        --checkpoint models/subgoal_predictor/checkpoints/run_d64_l1/best.pt \
        --instruction "pick up the red block and place it on the blue dish" \
        --robot 3
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from models.subgoal_predictor.config import ModelConfig
from models.subgoal_predictor.inference import load_model, denormalize_pose


class SubgoalExecutor:
    """Executes predicted subgoals on real robot."""

    def __init__(
        self,
        checkpoint_path: str,
        robot_id: int = 3,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.robot_id = robot_id

        # Load model
        self._log(f"Loading model from {checkpoint_path}")
        self.model, self.norm_stats, self.config = load_model(checkpoint_path, self.device)
        self._log(f"Model: d_model={self.config.d_model}, n_layers={self.config.n_decoder_layers}")

        # Robot skills (lazy init)
        self.skills = None
        self.camera = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[SubgoalExecutor] {msg}")

    def connect(self):
        """Connect to robot and camera."""
        from skills.skills_lerobot import LeRobotSkills
        from cameras.realsense_camera import RealSenseCamera
        from cameras.camera_config import RealSenseCameraConfig

        # Camera
        self._log("Connecting camera...")
        camera_config = RealSenseCameraConfig(
            name="front",
            width=640,
            height=480,
            fps=30,
        )
        self.camera = RealSenseCamera(camera_config)
        self.camera.connect()

        # Robot
        self._log(f"Connecting robot {self.robot_id}...")
        robot_config = f"robot_configs/robot/so101_robot{self.robot_id}.yaml"
        self.skills = LeRobotSkills(
            robot_config=robot_config,
            frame="robot",  # Use robot base frame (model predicts in robot frame)
            verbose=self.verbose,
        )
        self.skills.connect()

        self._log("Connected!")

    def disconnect(self):
        """Disconnect from robot and camera."""
        if self.skills:
            self.skills.disconnect()
        if self.camera:
            self.camera.disconnect()
        self._log("Disconnected")

    def capture_image(self) -> np.ndarray:
        """Capture current camera frame."""
        return self.camera.read()

    def get_robot_state(self) -> np.ndarray:
        """Get current robot joint state (normalized), including gripper."""
        state_arm, _, _ = self.skills._get_current_state()  # (5,) arm only
        gripper_pos = self.skills.current_gripper_pos  # scalar
        state_full = np.concatenate([state_arm, [gripper_pos]])  # (6,)
        return state_full.astype(np.float32)

    @torch.no_grad()
    def predict_subgoals(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray,
    ) -> dict:
        """Predict subgoal sequence from current observation.

        Args:
            image: RGB image (H, W, 3), uint8
            instruction: Natural language instruction
            state: Robot joint state (6,), normalized

        Returns:
            dict with 'poses', 'grippers', 'n_subgoals'
        """
        # Preprocess image
        img_pil = Image.fromarray(image)
        img_pil = TF.resize(img_pil, list(self.config.image_size))
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Normalize state
        state_norm = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        state_tensor = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        # Forward
        out = self.model(img_tensor, [instruction], state_tensor)

        # Decode outputs
        pred_pose = out["pose"].cpu().numpy()[0]  # (N_max, 6)
        pred_gripper_logits = out["gripper"].cpu().numpy()[0, :, 0]  # (N_max,)
        pred_valid_logits = out["valid"].cpu().numpy()[0, :, 0]  # (N_max,)

        # Binary predictions
        pred_gripper = (pred_gripper_logits > 0).astype(float)
        pred_valid = (pred_valid_logits > 0).astype(float)

        # Count valid subgoals
        n_subgoals = int(pred_valid.sum())

        # Denormalize poses
        poses_denorm = denormalize_pose(pred_pose, self.norm_stats)

        return {
            "poses": poses_denorm[:n_subgoals],  # (N, 6) xyzrpy in robot frame
            "grippers": pred_gripper[:n_subgoals],  # (N,) 0=close, 1=open
            "n_subgoals": n_subgoals,
        }

    def execute_subgoals(
        self,
        poses: np.ndarray,
        grippers: np.ndarray,
        duration: float = 3.0,
        dry_run: bool = False,
    ) -> bool:
        """Execute predicted subgoals on robot.

        Args:
            poses: (N, 6) xyzrpy in robot base frame
            grippers: (N,) 0=close, 1=open
            duration: Movement duration per subgoal
            dry_run: If True, print commands without executing

        Returns:
            True if all subgoals executed successfully
        """
        n = len(poses)
        self._log(f"Executing {n} subgoals...")

        # Get current gripper state
        current_gripper_open = self.skills.current_gripper_pos > 0

        for i, (pose, grip) in enumerate(zip(poses, grippers)):
            xyz = pose[:3]
            target_gripper_open = grip > 0.5

            self._log(f"\n[Subgoal {i+1}/{n}]")
            self._log(f"  Position: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
            self._log(f"  Gripper: {'OPEN' if target_gripper_open else 'CLOSE'}")

            if dry_run:
                continue

            # 1. Move to position first
            self._log(f"  → Moving to position...")
            success = self.skills.move_to_position(
                position=xyz.tolist(),
                duration=duration,
            )

            if not success:
                self._log(f"  ERROR: Move failed at subgoal {i+1}")
                return False

            # 2. Then adjust gripper if state changed
            if target_gripper_open != current_gripper_open:
                if target_gripper_open:
                    self._log("  → Opening gripper...")
                    self.skills.gripper_open(duration=1.5)
                else:
                    self._log("  → Closing gripper...")
                    self.skills.gripper_close(duration=1.5)
                current_gripper_open = target_gripper_open

        self._log(f"\nAll {n} subgoals executed successfully!")
        return True

    def run(
        self,
        instruction: str,
        duration: float = 3.0,
        dry_run: bool = False,
        save_visualization: str = None,
    ) -> dict:
        """Full inference + execution pipeline.

        Args:
            instruction: Natural language instruction
            duration: Movement duration per subgoal
            dry_run: If True, predict but don't execute
            save_visualization: Path to save visualization image

        Returns:
            dict with prediction results
        """
        self._log(f"\n{'='*60}")
        self._log(f"Instruction: {instruction}")
        self._log(f"{'='*60}")

        # Capture observation
        self._log("\n[1] Capturing observation...")
        image = self.capture_image()
        state = self.get_robot_state()
        self._log(f"  Image: {image.shape}")
        self._log(f"  State: {state}")

        # Predict subgoals
        self._log("\n[2] Predicting subgoals...")
        result = self.predict_subgoals(image, instruction, state)
        self._log(f"  Predicted {result['n_subgoals']} subgoals")

        for i in range(result['n_subgoals']):
            pose = result['poses'][i]
            grip = "open" if result['grippers'][i] > 0.5 else "close"
            self._log(f"    {i+1}. xyz=[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] grip={grip}")

        # Save visualization
        if save_visualization:
            self._save_visualization(image, instruction, result, save_visualization)

        # Execute
        if dry_run:
            self._log("\n[3] DRY RUN - Skipping execution")
        else:
            self._log("\n[3] Executing on robot...")
            self.execute_subgoals(
                result['poses'],
                result['grippers'],
                duration=duration,
                dry_run=False,
            )

        return result

    def _load_coordinate_transformer(self):
        """Load coordinate transformer for robot->pixel projection."""
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator

            calib_path = f"robot_configs/pix2robot_matrices/robot{self.robot_id}_pix2robot_data.npz"
            calibrator = Pix2RobotCalibrator(robot_id=self.robot_id)
            if not calibrator.load(calib_path):
                self._log("  Warning: Pix2Robot calibration not loaded")
                return None

            return calibrator

        except Exception as e:
            self._log(f"  Warning: Could not load transformer: {e}")
            return None

    def _robot_to_pixel(
        self,
        xyz_robot: np.ndarray,
        calibrator,
    ) -> tuple[int, int] | None:
        """Convert robot base frame XYZ to pixel coordinates.

        Args:
            xyz_robot: (3,) position in robot base frame (meters)
            calibrator: Pix2RobotCalibrator instance

        Returns:
            (u, v) pixel coordinates or None if out of bounds
        """
        try:
            u, v = calibrator.robot_to_pixel(xyz_robot[0], xyz_robot[1], xyz_robot[2])
            return int(u), int(v)
        except Exception:
            return None

    def _save_visualization(
        self,
        image: np.ndarray,
        instruction: str,
        result: dict,
        save_path: str,
    ):
        """Save visualization of predicted subgoals with positions overlaid on image."""
        vis = image.copy()
        h, w = vis.shape[:2]

        # Load coordinate transformer
        calibrator = self._load_coordinate_transformer()

        # Colors for visualization
        COLORS = [
            (255, 0, 0),    # Red
            (255, 128, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (0, 255, 255),  # Cyan
            (0, 128, 255),  # Light blue
            (0, 0, 255),    # Blue
            (128, 0, 255),  # Purple
            (255, 0, 255),  # Magenta
            (255, 0, 128),  # Pink
        ]

        # Draw subgoal positions on image
        pixel_positions = []
        if calibrator is not None:
            for i in range(result['n_subgoals']):
                pose = result['poses'][i]
                xyz = pose[:3]
                pixel = self._robot_to_pixel(xyz, calibrator)

                if pixel is not None:
                    u, v = pixel
                    if 0 <= u < w and 0 <= v < h:
                        pixel_positions.append((i, u, v))

            # Draw lines connecting subgoals (trajectory)
            for idx in range(1, len(pixel_positions)):
                prev_i, prev_u, prev_v = pixel_positions[idx - 1]
                curr_i, curr_u, curr_v = pixel_positions[idx]
                color = COLORS[curr_i % len(COLORS)]
                cv2.line(vis, (prev_u, prev_v), (curr_u, curr_v), color, 2)

            # Draw circles and numbers at each subgoal
            for i, u, v in pixel_positions:
                color = COLORS[i % len(COLORS)]
                grip = result['grippers'][i] > 0.5

                # Circle size based on gripper state
                radius = 12 if grip else 8  # Larger circle for open gripper

                # Draw circle
                cv2.circle(vis, (u, v), radius, color, -1)
                cv2.circle(vis, (u, v), radius, (255, 255, 255), 2)  # White border

                # Draw subgoal number
                text = str(i + 1)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = u - text_size[0] // 2
                text_y = v + text_size[1] // 2
                cv2.putText(vis, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add text overlay (semi-transparent background)
        overlay = vis.copy()
        cv2.rectangle(overlay, (5, 5), (400, 70 + result['n_subgoals'] * 22), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

        # Instruction (truncate if too long)
        instr_display = instruction[:45] + "..." if len(instruction) > 45 else instruction
        cv2.putText(vis, f"Instruction: {instr_display}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"Predicted {result['n_subgoals']} subgoals", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Subgoal list with colors and gripper action
        prev_grip = None
        for i in range(result['n_subgoals']):
            pose = result['poses'][i]
            curr_grip = result['grippers'][i] > 0.5
            color = COLORS[i % len(COLORS)]

            # Determine gripper action
            if prev_grip is None:
                grip_action = "(open)" if curr_grip else "(close)"
            elif curr_grip and not prev_grip:
                grip_action = "->OPEN"
            elif not curr_grip and prev_grip:
                grip_action = "->CLOSE"
            else:
                grip_action = "(hold)"

            text = f"{i+1}. [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] {grip_action}"
            cv2.putText(vis, text, (10, 72 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            prev_grip = curr_grip

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self._log(f"  Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute Subgoal Predictions on Robot")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--instruction", "-i", type=str, required=True, help="Task instruction")
    parser.add_argument("--robot", type=int, default=3, help="Robot ID")
    parser.add_argument("--duration", type=float, default=3.0, help="Movement duration per subgoal")
    parser.add_argument("--dry-run", action="store_true", help="Predict without executing")
    parser.add_argument("--save-vis", type=str, default=None, help="Save visualization to path")
    args = parser.parse_args()

    executor = SubgoalExecutor(
        checkpoint_path=args.checkpoint,
        robot_id=args.robot,
    )

    try:
        executor.connect()
        executor.run(
            instruction=args.instruction,
            duration=args.duration,
            dry_run=args.dry_run,
            save_visualization=args.save_vis,
        )
    finally:
        executor.disconnect()


if __name__ == "__main__":
    main()
