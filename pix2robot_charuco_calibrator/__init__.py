"""Charuco 기반 카메라-로봇 캘리브레이션 (Phase 1~4 통합 파이프라인)."""

from .geometry import (
    pixel_to_camera_3d,
    affine_align_3d,
    rigid_align_kabsch,
    apply_transform,
    reprojection_residuals,
    compose_transform_4x4,
    decompose_transform_4x4,
)
from .charuco_detector import (
    CharucoBoardSpec,
    CharucoDetector,
    Detection,
    auto_detect_dictionary,
    group_compatible_dictionaries,
    STANDARD_DICTIONARIES,
)

__all__ = [
    "pixel_to_camera_3d",
    "affine_align_3d",
    "rigid_align_kabsch",
    "apply_transform",
    "reprojection_residuals",
    "compose_transform_4x4",
    "decompose_transform_4x4",
    "CharucoBoardSpec",
    "CharucoDetector",
    "Detection",
    "auto_detect_dictionary",
    "group_compatible_dictionaries",
    "STANDARD_DICTIONARIES",
]
