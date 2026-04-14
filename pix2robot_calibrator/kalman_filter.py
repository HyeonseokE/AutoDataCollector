"""
Kalman filter for 3D position smoothing.

대상: pix2robot 변환 출력([x, y, z]) 시계열을 부드럽게 만들기 위함.

모델:
- 상태 벡터 x = [x, y, z]  (3D 위치만, 속도 없음 — 물체는 정적이라 가정)
- 측정 벡터 z = [x, y, z]  (동일)
- 상태 천이 F = I (constant-position / random walk)
- 관측 행렬 H = I (위치를 직접 측정)

용도:
- 다중 프레임 detection 누적 시 노이즈 감쇄 (homography 오차 + 픽셀 지터 + depth 노이즈)
- confidence가 높을수록 측정값을 더 신뢰 (R 가중치 감소)

단일 측정(one-shot) 케이스에서는 사용하지 않음 — 시계열 평균이 의미가 없음.
"""

from typing import Dict, Optional

import numpy as np


class KalmanFilter3D:
    """3D 위치 칼만 필터 (constant-position 모델).

    Args:
        process_noise: 프로세스 노이즈 표준편차 (m). 물체가 얼마나 움직일 수 있는지.
                      정적 물체면 작게(0.001~0.005), 느리게 움직이면 크게.
        measurement_noise: 측정 노이즈 표준편차 (m). pix2robot 변환의 대략적인 오차.
                          보통 0.01~0.05 (1~5cm).
        initial_covariance: 초기 상태 공분산 (첫 측정을 얼마나 믿을지).
                           크게 잡으면 첫 측정으로 빠르게 수렴.
    """

    def __init__(
        self,
        process_noise: float = 0.002,
        measurement_noise: float = 0.02,
        initial_covariance: float = 1.0,
    ):
        self.q = float(process_noise) ** 2   # Q 대각 성분 (분산)
        self.r = float(measurement_noise) ** 2  # R 대각 성분 (분산)
        self.p0 = float(initial_covariance)

        # 상태: (3,) 벡터 None이면 미초기화
        self.state: Optional[np.ndarray] = None
        # 상태 공분산: (3, 3)
        self.P: np.ndarray = np.eye(3) * self.p0

        # 통계
        self.update_count: int = 0

    def reset(self) -> None:
        """필터 초기화 (새 에피소드/세션 시작 시 호출)."""
        self.state = None
        self.P = np.eye(3) * self.p0
        self.update_count = 0

    def update(
        self,
        measurement: np.ndarray,
        confidence: float = 1.0,
    ) -> np.ndarray:
        """
        측정값으로 상태 업데이트 후 필터링된 위치 반환.

        Args:
            measurement: 측정된 3D 위치 [x, y, z] (m)
            confidence: 측정 신뢰도 (0~1). 높을수록 측정값을 더 신뢰.
                       R_effective = R / max(confidence, 0.1)

        Returns:
            필터링된 3D 위치 [x, y, z] (numpy array, shape (3,))
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(3)

        # 첫 측정: 그대로 상태로 설정 (수렴 가속)
        if self.state is None:
            self.state = z.copy()
            self.P = np.eye(3) * self.p0
            self.update_count = 1
            return self.state.copy()

        # --- Predict ---
        # F = I 이므로 x_pred = x
        # P_pred = F P F^T + Q = P + Q
        P_pred = self.P + np.eye(3) * self.q
        x_pred = self.state  # constant-position

        # --- Update ---
        # confidence 가중치: conf ↑ → R ↓ → 측정값을 더 신뢰
        conf = max(float(confidence), 0.1)
        R_eff = (self.r / conf) * np.eye(3)

        # S = H P_pred H^T + R = P_pred + R  (H = I)
        S = P_pred + R_eff
        # K = P_pred H^T S^-1 = P_pred S^-1
        K = P_pred @ np.linalg.inv(S)

        # x_new = x_pred + K (z - H x_pred) = x_pred + K (z - x_pred)
        innovation = z - x_pred
        self.state = x_pred + K @ innovation

        # P_new = (I - K H) P_pred = (I - K) P_pred
        self.P = (np.eye(3) - K) @ P_pred

        self.update_count += 1
        return self.state.copy()

    @property
    def initialized(self) -> bool:
        """첫 측정이 들어왔는지 여부."""
        return self.state is not None


class MultiObjectKalmanTracker:
    """여러 객체를 객체 이름(key)으로 동시에 추적.

    Usage:
        tracker = MultiObjectKalmanTracker(process_noise=0.002, measurement_noise=0.02)
        smoothed = tracker.update("red block", raw_pos, confidence=0.85)
        tracker.reset("red block")   # 특정 객체만 리셋
        tracker.reset_all()           # 전체 리셋
    """

    def __init__(
        self,
        process_noise: float = 0.002,
        measurement_noise: float = 0.02,
        initial_covariance: float = 1.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self._filters: Dict[str, KalmanFilter3D] = {}

    def _ensure(self, key: str) -> KalmanFilter3D:
        if key not in self._filters:
            self._filters[key] = KalmanFilter3D(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                initial_covariance=self.initial_covariance,
            )
        return self._filters[key]

    def update(
        self,
        key: str,
        measurement: np.ndarray,
        confidence: float = 1.0,
    ) -> np.ndarray:
        """객체 `key`의 위치 측정값 업데이트."""
        return self._ensure(key).update(measurement, confidence)

    def get(self, key: str) -> Optional[np.ndarray]:
        """객체 `key`의 현재 필터 상태 (없으면 None)."""
        f = self._filters.get(key)
        if f is None or not f.initialized:
            return None
        return f.state.copy()

    def reset(self, key: str) -> None:
        """특정 객체 필터 초기화."""
        if key in self._filters:
            self._filters[key].reset()

    def reset_all(self) -> None:
        """모든 객체 필터 초기화."""
        for f in self._filters.values():
            f.reset()

    def update_counts(self) -> Dict[str, int]:
        """각 객체별 업데이트 횟수 (디버깅용)."""
        return {k: f.update_count for k, f in self._filters.items()}
