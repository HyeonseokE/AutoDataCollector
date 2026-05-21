"""DCT-space single-step denoise U_VLA — paradigm step [4].

candidate 의 skill 단위 DCT feature ``z_cand = candidate.dct_target ∈ ℝ^{L0×dof}``
를 VLA forward 의 action 자리에 넣어 *한 번의* denoise loss 를 평가한다::

    z_noisy = z_cand + σ·ε
    ẑ      = VLA.denoise(o, I, skill, z_noisy, σ)
    U_VLA(ξ) = ‖ẑ − z_cand‖²

R=1 — 단일 noise sample 의 prediction error 가 그대로 epistemic surrogate.
smolvla 가 Phase 3 에서 ``skill_dct_target`` 로 학습됐다는 가정 하에 동작한다.

기존 ``LeRobotVLAInformativenessScorer`` 와 같은 ``VLAInformativenessScorer``
Protocol 을 따른다 — selector 측은 두 scorer 를 swap-in 가능.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from method3.phase2_mi_selection.mi_selector import Phase2Candidate

BatchBuilder = Callable[[Phase2Candidate], "dict"]


def _default_batch_builder(candidate: Phase2Candidate) -> object:
    if candidate.observations is None:
        raise ValueError(
            "default batch_builder expects candidate.observations to be a pre-built "
            "LeRobot batch dict. Either populate it or pass an explicit batch_builder."
        )
    return candidate.observations


@dataclass
class DCTDenoiseUncertainty:
    """paradigm step [4] — single-step DCT-space denoise loss.

    Args:
        policy: ``forward(batch, reduction="mean") -> (loss_tensor, info)`` 를
            노출하는 LeRobot policy 인스턴스 (frozen, eval mode 권장). smolvla
            가 Phase 3 에서 skill_dct_target 으로 학습됐다는 가정.
        batch_builder: ``candidate -> batch dict``. None 이면
            ``candidate.observations`` 를 그대로 batch 로 사용. action 자리에는
            본 scorer 가 ``candidate.dct_target`` 을 inject 한다.
        sigma: noise schedule 의 한 시점. None 이면 policy.forward 가 내부에서
            sample (R=1 single-step 의 noise/time stochasticity 는 policy 가
            책임). 명시하면 forward 의 ``time`` 인자로 전달.
    """

    policy: object
    batch_builder: BatchBuilder | None = None
    sigma: float | None = None

    def score(self, candidate: Phase2Candidate) -> float:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "DCTDenoiseUncertainty requires torch."
            ) from e

        if candidate.dct_target is None:
            raise ValueError(
                "DCTDenoiseUncertainty requires candidate.dct_target — "
                "use curobo_candidate_gen.candidates_from_trajectory_list (Phase 4)."
            )

        builder = self.batch_builder or _default_batch_builder
        batch = builder(candidate)
        if batch is None:
            raise ValueError(
                "batch_builder returned None — candidate may be missing observations."
            )
        if not isinstance(batch, dict):
            batch = dict(batch)

        # paradigm step [4]: action 자리에 z_cand 를 넣어 single-step denoise loss.
        z_cand = np.asarray(candidate.dct_target, dtype=np.float32)
        action_tensor = torch.from_numpy(z_cand).unsqueeze(0)  # (1, L0, dof)
        try:
            from lerobot.constants import ACTION  # type: ignore

            batch[ACTION] = action_tensor
        except Exception:
            batch["action"] = action_tensor

        forward_kwargs: dict = {}
        if self.sigma is not None:
            forward_kwargs["time"] = torch.tensor(
                [float(self.sigma)], dtype=torch.float32
            )

        was_training = getattr(self.policy, "training", False)
        try:
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            with torch.no_grad():
                out = self.policy.forward(batch, reduction="mean", **forward_kwargs)
                loss_t = out[0] if isinstance(out, tuple) else out
                u_vla = float(loss_t.detach().cpu().item())
        finally:
            if was_training and hasattr(self.policy, "train"):
                self.policy.train(True)
        return u_vla
