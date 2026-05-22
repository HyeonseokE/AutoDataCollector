"""VLA-side informativeness ``U_VLA(ξ)`` — final_method3_spec_useful_ood_updated §12.

Phase1-trained VLA ``π_θ^{(1)}`` 의 stochastic denoising loss 를 후보 trajectory
window 별로 R 번 평가·평균한 뒤 trajectory-level 로 aggregate 한다::

    U_VLA(ξ) = Agg_τ [ (1/R) Σ_r L_denoise^{(r)}(o_τ, I, p_τ, A_{τ:τ+H-1}; π_θ^{(1)}) ]

핵심 — 이 값은 **선택을 위한 prioritization signal** 이지 rejection signal 이
아니다 (§13.4). 따라서 본 모듈은 score 만 계산하고 임계 판정은 ``mi_selector``
의 Useful-OOD rule 이 ``M̃_MI`` constraint 안에서 한다.

구현 분리:
  * ``VLAInformativenessScorer`` — Protocol (외부 의존성 0).
  * ``ConstantScorer`` / ``ActionMagnitudeScorer`` — VLA 없이 mock 으로 selector
    경로를 검증할 때 쓴다.
  * ``LeRobotVLAInformativenessScorer`` — 실 VLA wrapping. pi0 / pi05 / smolvla
    의 통일된 ``policy.forward(batch, reduction)`` 인터페이스를 호출한다.
    batch 구성은 caller 가 등록한 ``batch_builder(candidate) -> dict`` 가 책임진다
    (LeRobot 의 OBS_STATE/OBS_IMAGE/ACTION 키 스키마는 config-dependent 이므로
    본 모듈은 어떤 키도 가정하지 않는다).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from method3.phase2_mi_selection.mi_selector import Phase2Candidate


class VLAInformativenessScorer(Protocol):
    """U_VLA(ξ) 를 계산하는 모듈의 Protocol — 외부 라이브러리 의존성 없이 합성 가능."""

    def score(self, candidate: Phase2Candidate) -> float:
        """후보 한 개의 model-side informativeness (높을수록 prioritize)."""
        ...


# ---------------------------------------------------------------------------
# Mock scorers — selector 경로 검증과 ablation 용
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantScorer:
    """모든 후보를 같은 값으로 채점. selector tie-breaking 검증/ablation 용."""

    value: float = 0.0

    def score(self, candidate: Phase2Candidate) -> float:  # noqa: ARG002
        return float(self.value)


@dataclass(frozen=True)
class ActionMagnitudeScorer:
    """후보 action chunk 의 평균 절대값. 실 VLA 가 없을 때 deterministic surrogate.

    의미상 U_VLA 는 아니지만, action 다양성이 큰 후보가 높은 score 를 받게 하여
    Useful-OOD rule 의 batch-level argmax 분기가 정상 작동하는지 검증할 수 있다.
    """

    def score(self, candidate: Phase2Candidate) -> float:
        chunks = np.asarray(candidate.action_chunks, dtype=np.float64)
        if chunks.size == 0:
            return 0.0
        return float(np.mean(np.abs(chunks)))


# ---------------------------------------------------------------------------
# LeRobot VLA adapter — 실 Phase1-trained VLA 의 R-stochastic denoise loss
# ---------------------------------------------------------------------------

# Phase2Candidate 에 batch_builder 가 사용할 obs/proprio/instruction payload 가
# 들어와야 한다. Phase2Candidate 의 ``payload`` 필드를 그대로 받아 사용한다 —
# scorer 는 그 형식을 모른다 (batch_builder 가 책임).
BatchBuilder = Callable[[Phase2Candidate], "dict"]


def _default_batch_builder(candidate: Phase2Candidate) -> object:
    """candidate.observations 가 이미 LeRobot batch dict 라면 그대로 사용.

    caller 가 candidate.observations 에 family-aware batch 를 미리 채워 넣는 패턴.
    family-specific transform (image preprocess, tokenize) 까지 한 채로 넘기면
    이 default 만으로 충분하다. 더 복잡한 변환은 explicit batch_builder 로.
    """
    if candidate.observations is None:
        raise ValueError(
            "default batch_builder expects candidate.observations to be a pre-built "
            "LeRobot batch dict. Either populate it or pass an explicit batch_builder."
        )
    return candidate.observations


@dataclass
class LeRobotVLAInformativenessScorer:
    """실 Phase1-trained VLA 의 denoising loss 로 ``U_VLA(ξ)`` 계산 (§12.1).

    pi0 / pi05 / smolvla 의 통일 인터페이스
    ``policy.forward(batch, reduction="mean") -> (loss_tensor, info)`` 를 호출한다.

    두 mode:
      * ``mode="default"`` — caller 가 batch_builder 로 만든 batch 그대로 forward.
        R-stochastic (noise/time random sample 매번 새로 일어남) → 평균/최대 loss.
      * ``mode="dct"`` — method3 DCT paradigm. candidate.dct_target 을 batch["action"]
        자리에 inject 후 single-step (R=1 강제) denoise loss. ``sigma`` 명시 시
        forward 의 ``time`` 인자로 전달 (deterministic), None 이면 policy 내부 schedule
        에서 random sample. policy 가 DCT-target 으로 학습됐다는 가정 (paradigm step
        [1][2]) — 그 가정이 안 맞으면 loss 의미 없음.

    Args:
        policy: ``forward(batch, reduction)`` 를 노출하는 LeRobot policy 인스턴스
            (freeze 상태로 전달되어야 한다 — 본 모듈은 grad 를 끄기만 한다).
        batch_builder: ``candidate -> dict`` — LeRobot batch 스키마를 만든다.
            None 이면 ``candidate.observations`` 를 그대로 batch 로 사용 (caller 가
            family-aware 로 미리 채워 넣는 패턴). family-별 transform 이 필요하면
            explicit builder 를 등록.
        R: §12.1 stochastic denoise eval 횟수. mode="dct" 에선 항상 1 (override).
        agg: ``mean`` 또는 ``max`` — R 번 loss 의 trajectory-level aggregator.
        mode: ``"default"`` | ``"dct"`` — 위 두 mode 참조.
        sigma: mode="dct" 전용. None=policy 내부 schedule sample, float=forward 의
            ``time`` 인자로 명시 전달 → score 가 sigma 에 대해 deterministic.
    """

    policy: object
    batch_builder: BatchBuilder | None = None
    R: int = 8
    agg: str = "mean"
    mode: str = "default"
    sigma: float | None = None

    def score(self, candidate: Phase2Candidate) -> float:
        """단일 candidate 의 U_VLA. dct mode 는 batched 경로(``score_batch``)로 위임.

        default mode 는 R-stochastic noise sampling 으로 candidate 1개를 평가한다.
        """
        if self.mode == "dct":
            return float(self.score_batch([candidate])[0])
        if self.mode != "default":
            raise ValueError(f"unknown mode={self.mode!r} (expected 'default' or 'dct')")
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "LeRobotVLAInformativenessScorer requires torch. "
                "Install torch or fall back to ActionMagnitudeScorer."
            ) from e

        builder = self.batch_builder or _default_batch_builder
        batch = builder(candidate)
        if batch is None:
            raise ValueError(
                "batch_builder returned None — candidate may be missing observations/proprio."
            )
        if not isinstance(batch, dict):
            batch = dict(batch)

        effective_R = int(max(1, self.R))
        losses: list[float] = []
        was_training = getattr(self.policy, "training", False)
        try:
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            with torch.no_grad():
                for _ in range(effective_R):
                    # forward(batch, reduction="mean") → (loss_tensor, info_dict)
                    out = self.policy.forward(batch, reduction="mean")
                    loss_t = out[0] if isinstance(out, tuple) else out
                    losses.append(float(loss_t.detach().cpu().item()))
        finally:
            if was_training and hasattr(self.policy, "train"):
                self.policy.train(True)

        if not losses:
            return 0.0
        if self.agg == "max":
            return float(np.max(losses))
        return float(np.mean(losses))

    def score_batch(self, candidates, batch_size: int = 128) -> np.ndarray:
        """여러 candidate 의 U_VLA 를 GPU batched forward 로 일괄 계산 (dct mode).

        Phase2 후보 수는 2^N (예: 128) 으로 잡으므로 ``batch_size`` (기본 128)
        단위로 묶어 GPU 에서 batched forward 한다 — candidate 1개씩 forward 하던 것
        대비 큰 속도 이득. 각 candidate 의 U_VLA 는 skill segment 의 *단일*
        DCT_50 target 에 대한 single-step denoise loss 이므로 batch 의 한 행에
        대응한다: skill 초기 관측(builder 결과의 첫 window)을 prefix,
        ``dct_target`` 을 suffix(action 자리)로 stack 하고 ``reduction="none"``
        으로 per-sample loss (B,) 를 받는다.

        default mode 는 per-candidate ``score`` 로 fallback.

        Returns:
            ``np.ndarray`` shape (len(candidates),) — 후보별 U_VLA.
        """
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "LeRobotVLAInformativenessScorer requires torch."
            ) from e

        n = len(candidates)
        if n == 0:
            return np.zeros(0, dtype=float)
        if self.mode != "dct":
            return np.array([self.score(c) for c in candidates], dtype=float)

        builder = self.batch_builder or _default_batch_builder
        # action 키 — lerobot constant 우선, 없으면 well-known 이름.
        try:
            from lerobot.constants import ACTION as _ACTION_KEY  # type: ignore
        except Exception:
            _ACTION_KEY = "action"
        # policy device — batched 텐서를 모두 이 device 로 (mock policy 는 None).
        _dev = None
        try:
            _param = next(self.policy.parameters(), None)
            if _param is not None:
                _dev = _param.device
        except (AttributeError, TypeError, StopIteration):
            _dev = None

        out = np.zeros(n, dtype=float)
        was_training = getattr(self.policy, "training", False)
        import time as _tmod
        _tb_build = _tb_fwd = 0.0
        try:
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            with torch.no_grad():
                for start in range(0, n, max(1, int(batch_size))):
                    _t_c = _tmod.perf_counter()
                    chunk = candidates[start:start + max(1, int(batch_size))]
                    # dct_target 필수 체크.
                    for c in chunk:
                        if c.dct_target is None:
                            raise ValueError(
                                "mode='dct' requires candidate.dct_target — use "
                                "curobo_candidate_gen.candidates_from_trajectory_list."
                            )
                    # image/instruction/proprio(첫 window=skill 시작) 는 chunk 내
                    # 모든 candidate 가 동일하다 — 같은 plan_and_select 의 obs.
                    # candidate 마다 다른 건 dct_target(아래 inject) 뿐이므로
                    # builder 를 1회만 호출하고 batch 차원으로 expand 한다.
                    # builder 가 score_batch 의 최대 병목이었다 (13~18s →
                    # 1회 ≈0.2s; candidate 마다 image resize/tokenize 중복).
                    b0 = builder(chunk[0])
                    if b0 is None:
                        raise ValueError(
                            "batch_builder returned None — candidate may be "
                            "missing observations/proprio."
                        )
                    if not isinstance(b0, dict):
                        b0 = dict(b0)
                    cn = len(chunk)
                    batch: dict = {}
                    for k, v in b0.items():
                        if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                            # 첫 window (1, ...) → (chunk_n, ...) 로 expand.
                            v1 = v[:1]
                            batch[k] = v1.expand(cn, *v1.shape[1:]).contiguous()
                        else:
                            batch[k] = v
                    # dct_target → action 자리 (chunk_n, L0, dof).
                    z = np.stack([
                        np.asarray(c.dct_target, dtype=np.float32) for c in chunk
                    ])
                    batch[_ACTION_KEY] = torch.from_numpy(z)
                    # device 정렬.
                    if _dev is not None:
                        batch = {
                            k: (v.to(_dev) if hasattr(v, "to") else v)
                            for k, v in batch.items()
                        }
                    # single-step time = sigma, batch 차원 (chunk_n,).
                    forward_kwargs: dict = {}
                    if self.sigma is not None:
                        _time = torch.full(
                            (len(chunk),), float(self.sigma), dtype=torch.float32
                        )
                        if _dev is not None:
                            _time = _time.to(_dev)
                        forward_kwargs["time"] = _time
                    # reduction="none" → per-sample loss (chunk_n,).
                    _t_f = _tmod.perf_counter()
                    _tb_build += _t_f - _t_c
                    out_t = self.policy.forward(
                        batch, reduction="none", **forward_kwargs
                    )
                    loss_t = out_t[0] if isinstance(out_t, tuple) else out_t
                    loss_np = np.asarray(
                        loss_t.detach().cpu().numpy(), dtype=float
                    ).reshape(-1)
                    out[start:start + len(chunk)] = loss_np[:len(chunk)]
                    _tb_fwd += _tmod.perf_counter() - _t_f
        finally:
            if was_training and hasattr(self.policy, "train"):
                self.policy.train(True)
        print(
            f"[timing:score_batch] builder={_tb_build * 1000:.0f}ms  "
            f"forward={_tb_fwd * 1000:.0f}ms  (n={n})",
            flush=True,
        )
        return out


def make_default_scorer() -> VLAInformativenessScorer:
    """VLA policy 가 주입되지 않은 환경의 default — Useful-OOD argmax 가 동작은
    하되 의미상 selection 우선순위는 trivial (action magnitude) 가 된다."""
    return ActionMagnitudeScorer()


# ---------------------------------------------------------------------------
# Family-aware batch builder — raw candidate fields → LeRobot policy.forward batch
# ---------------------------------------------------------------------------


@dataclass
class LeRobotBatchBuilder:
    """raw Phase2Candidate (observations + instruction + proprios + action_chunks)
    → LeRobot ``policy.forward(batch)`` 가 받는 dict.

    pi0 / pi05 / smolvla 의 공통 batch 키 schema:
        OBS_STATE                   — (B, P)             proprio
        OBS_LANGUAGE_TOKENS         — (B, L)             tokenizer 결과
        OBS_LANGUAGE_ATTENTION_MASK — (B, L)
        OBS_IMAGE.<cam>             — (B, 3, H, W)       각 카메라 별 image
        ACTION                      — (B, H, action_dim)

    B = candidate.proprios 의 window 수 T (window 단위 batch — 한 candidate 의
    모든 window 를 한 forward 로 평가). 각 window 가 동일한 obs/instruction 을
    공유하면 단순 broadcast 로 batch 채움.

    본 builder 는 *최소한의 변환* 만 한다:
      * proprios / action_chunks → torch.Tensor (B, …)
      * instruction → tokenizer 로 (L,) → (B, L) broadcast
      * observations 가 dict[cam_name, ndarray(H,W,3)] 이면 (B, 3, H, W) 정규화

    더 정교한 transform (image stats normalization, dtype conversion 등) 이
    필요하면 caller 가 explicit batch_builder 를 직접 등록.
    """

    policy: object  # LeRobot policy 인스턴스 — tokenizer / config 접근용

    def _get_tokenizer(self):
        """vlm tokenizer 를 lazy load + 캐시.

        탐색 순서:
          1. policy 가 이미 보유한 tokenizer (vlm_with_expert.processor.tokenizer
             등) — 재사용.
          2. config.vlm_model_name 으로 AutoTokenizer.from_pretrained
             (processor_smolvla.TokenizerProcessorStep 와 동일 소스).
        실패 시 None.
        """
        cached = getattr(self, "_tokenizer_cache", None)
        if cached is not None:
            return cached
        tok = None
        # 1) policy 내부 tokenizer 경로 탐색
        for path in (
            ("vlm_with_expert", "processor", "tokenizer"),
            ("model", "vlm_with_expert", "processor", "tokenizer"),
            ("language_tokenizer",),
            ("tokenizer",),
        ):
            obj = self.policy
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and callable(obj):
                tok = obj
                break
        # 2) config.vlm_model_name 으로 새로 load
        if tok is None:
            cfg = getattr(self.policy, "config", None)
            vlm_name = getattr(cfg, "vlm_model_name", None)
            if vlm_name:
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(vlm_name)
                except Exception as e:
                    print(f"  [LeRobotBatchBuilder] AutoTokenizer load failed: {e}", flush=True)
        object.__setattr__(self, "_tokenizer_cache", tok)
        return tok

    def __call__(self, candidate: Phase2Candidate) -> dict:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("LeRobotBatchBuilder requires torch.") from e

        if candidate.proprios is None or candidate.action_chunks is None:
            raise ValueError(
                "LeRobotBatchBuilder needs candidate.proprios and action_chunks. "
                "Generator should populate them."
            )

        proprios = np.asarray(candidate.proprios, dtype=np.float32)  # (T, P)
        actions = np.asarray(candidate.action_chunks, dtype=np.float32)  # (T, H, A)
        B = proprios.shape[0]

        batch: dict = {}
        # State / action — LeRobot constants 가 있으면 그 키 사용, 아니면 well-known
        # naming convention 으로 fallback.
        try:
            from lerobot.constants import OBS_STATE, ACTION  # type: ignore
            batch[OBS_STATE] = torch.from_numpy(proprios)
            batch[ACTION] = torch.from_numpy(actions)
        except Exception:
            batch["observation.state"] = torch.from_numpy(proprios)
            batch["action"] = torch.from_numpy(actions)

        # Instruction tokenize — smolvla processor_smolvla.py 의 TokenizerProcessorStep
        # 재현: task 에 newline 추가 → vlm tokenizer 로 (max_length, padding) 토큰화 →
        # OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK 키로 batch 에 주입.
        # scorer 는 policy.forward 를 직접 호출 (preprocessor pipeline skip) 하므로
        # 본 builder 가 그 step 을 대신한다.
        instr = candidate.instruction or ""
        if instr:
            try:
                from lerobot.utils.constants import (
                    OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK,
                )
                tok = self._get_tokenizer()
                if tok is not None:
                    cfg = getattr(self.policy, "config", None)
                    max_len = int(getattr(cfg, "tokenizer_max_length", 48) or 48)
                    instr_nl = instr if instr.endswith("\n") else instr + "\n"
                    enc = tok(
                        instr_nl, return_tensors="pt",
                        padding="max_length", max_length=max_len, truncation=True,
                    )
                    ids = enc["input_ids"]  # (1, L) — token id 는 Long 유지
                    mask = enc.get("attention_mask", torch.ones_like(ids))
                    batch[OBS_LANGUAGE_TOKENS] = ids.expand(B, -1)
                    # attention mask 는 bool 이어야 한다 — smolvla forward 의
                    # make_att_2d_masks 가 pad_masks 를 곱/AND 하고, 그 결과가
                    # eager_attention_forward 의 torch.where condition 으로 쓰인다.
                    # tokenizer 는 int64 mask 를 주므로(정식 preprocessor 는 bool
                    # 변환) 여기서 명시 변환하지 않으면 condition 이 Long 이 되어
                    # "where expected condition to be a boolean tensor" 로 죽는다.
                    batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.expand(B, -1).bool()
            except Exception as _e:
                print(f"  [LeRobotBatchBuilder] language tokenize failed: {_e}", flush=True)

        # Observations — dict[cam, ndarray(H,W,3)] 면 (B, 3, H, W) 정규화. 그 외
        # 형식은 caller 가 미리 batch dict 로 변환했다는 가정 하에 그대로 merge.
        # policy image feature 의 target (H, W) 로 resize — raw capture
        # (640×480 등) 와 policy expect (256×256 등) 불일치 보정.
        _img_hw = None
        try:
            _ifeat = getattr(getattr(self.policy, "config", None), "image_features", {}) or {}
            if _ifeat:
                _shp = list(_ifeat.values())[0].shape  # (C, H, W)
                _img_hw = (int(_shp[1]), int(_shp[2]))
        except Exception:
            _img_hw = None

        obs = candidate.observations
        if isinstance(obs, dict):
            for cam, frame in obs.items():
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    arr = frame
                    # HWC → target (H, W) resize (필요 시).
                    if _img_hw is not None and arr.shape[:2] != _img_hw:
                        import cv2
                        arr = cv2.resize(
                            arr, (_img_hw[1], _img_hw[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    # HWC uint8/float → BCHW float32 (0~1).
                    t = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32) / 255.0)
                    t = t.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
                    key = f"observation.images.{cam}" if not str(cam).startswith("observation.") else cam
                    batch[key] = t
                else:
                    # already-built tensor or unknown — 그대로 사용.
                    batch[str(cam)] = frame

        # 모든 tensor 를 policy device 로 이동 — scorer 는 policy.forward 를
        # 직접 호출해 preprocessor 의 DeviceProcessorStep 을 거치지 않으므로,
        # CPU tensor (torch.from_numpy / tokenizer 결과) 와 cuda policy weight
        # 의 device mismatch 가 발생한다. 여기서 일괄 .to(device).
        try:
            _dev = None
            _params = getattr(self.policy, "parameters", None)
            if callable(_params):
                _dev = next(self.policy.parameters()).device
            if _dev is None:
                _dev = getattr(getattr(self.policy, "config", None), "device", None)
            if _dev is not None:
                batch = {
                    k: (v.to(_dev) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
        except Exception as _e:
            print(f"  [LeRobotBatchBuilder] device move failed: {_e}", flush=True)
        return batch
