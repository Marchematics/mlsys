import numpy as np
import pytest

from mur.calibration import SymmetricUtilityCalibrator


def test_calibrator_uses_finite_sample_adjusted_residual_quantile() -> None:
    prediction = np.zeros(9)
    observed = np.arange(9, dtype=float) / 10.0
    calibrator = SymmetricUtilityCalibrator.fit(prediction, observed, alpha=0.2)
    assert calibrator.radius == 0.8
    lower, upper = calibrator.interval(np.asarray([0.5]))
    assert lower[0] == pytest.approx(-0.3)
    assert upper[0] == pytest.approx(1.3)


def test_calibrated_decisions_separate_selection_stop_and_uncertainty() -> None:
    calibrator = SymmetricUtilityCalibrator(radius=0.2, alpha=0.1, calibration_size=100)
    assert calibrator.candidate_decision(np.asarray([0.3, 0.1])) == ("select", 0)
    assert calibrator.candidate_decision(np.asarray([-0.4, -0.3])) == (
        "certified_stop",
        None,
    )
    assert calibrator.candidate_decision(np.asarray([0.1, -0.1])) == (
        "uncertain_stop",
        None,
    )
