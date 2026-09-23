"""Split-calibrated utility intervals and auditable routing decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetricUtilityCalibrator:
    """A split-conformal symmetric residual interval for utility predictions."""

    radius: float
    alpha: float
    calibration_size: int

    @classmethod
    def fit(
        cls,
        predicted: np.ndarray,
        observed: np.ndarray,
        *,
        alpha: float = 0.1,
    ) -> "SymmetricUtilityCalibrator":
        prediction = np.asarray(predicted, dtype=float)
        target = np.asarray(observed, dtype=float)
        if prediction.ndim != 1 or prediction.shape != target.shape or prediction.size == 0:
            raise ValueError("predicted and observed must be equal-length nonempty vectors")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
            raise ValueError("calibration values must be finite")
        residual = np.abs(target - prediction)
        level = min(1.0, np.ceil((prediction.size + 1) * (1.0 - alpha)) / prediction.size)
        radius = float(np.quantile(residual, level, method="higher"))
        return cls(radius=radius, alpha=alpha, calibration_size=prediction.size)

    def interval(self, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(predicted, dtype=float)
        if np.any(np.isnan(prediction)):
            raise ValueError("predicted utilities must not contain NaN")
        return prediction - self.radius, prediction + self.radius

    def candidate_decision(self, predicted: np.ndarray) -> tuple[str, int | None]:
        """Return select, certified_stop, or uncertain_stop for one routing state."""

        prediction = np.asarray(predicted, dtype=float)
        if prediction.ndim != 1 or prediction.size == 0:
            raise ValueError("predicted utilities must be a nonempty vector")
        lower, upper = self.interval(prediction)
        best = int(np.argmax(lower))
        if lower[best] > 0.0:
            return "select", best
        if np.all(upper <= 0.0):
            return "certified_stop", None
        return "uncertain_stop", None
