"""Decision-facing metrics for context selection."""

from __future__ import annotations

import numpy as np


def decision_metrics(
    base_loss: np.ndarray,
    routed_loss: np.ndarray,
    oracle_loss: np.ndarray,
    selected_counts: np.ndarray,
    *,
    harm_margin: float = 0.0,
) -> dict[str, float]:
    """Compute gain, utility recovery, harm rate, and gain per selected context."""

    base = np.asarray(base_loss, dtype=float)
    routed = np.asarray(routed_loss, dtype=float)
    oracle = np.asarray(oracle_loss, dtype=float)
    counts = np.asarray(selected_counts, dtype=float)
    if not (base.shape == routed.shape == oracle.shape == counts.shape) or base.ndim != 1:
        raise ValueError("all metric inputs must be equal-length vectors")
    if base.size == 0 or not all(np.all(np.isfinite(v)) for v in (base, routed, oracle, counts)):
        raise ValueError("metric inputs must be nonempty and finite")
    if np.any(counts < 0.0) or harm_margin < 0.0:
        raise ValueError("counts and harm_margin must be nonnegative")
    gain = base - routed
    oracle_gain = base - oracle
    eligible = oracle_gain > 0.0
    recovery = np.divide(
        gain[eligible], oracle_gain[eligible], out=np.full(np.sum(eligible), np.nan), where=True
    )
    used = counts > 0.0
    efficiency = np.divide(gain[used], counts[used]) if np.any(used) else np.asarray([])
    return {
        "mean_prediction_gain": float(np.mean(gain)),
        "negative_transfer_rate": float(np.mean(routed > base + harm_margin)),
        "mean_utility_recovery": float(np.nanmean(recovery)) if np.any(eligible) else float("nan"),
        "mean_selected_contexts": float(np.mean(counts)),
        "mean_gain_per_selected_context": (
            float(np.nanmean(efficiency)) if efficiency.size else float("nan")
        ),
    }
