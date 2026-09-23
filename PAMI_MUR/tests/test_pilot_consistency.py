"""Tests for the split-half pilot filter.

The filter's value is that it certifies a verdict, so the piece that has to be
pinned is what counts as consistent: both halves must resolve and agree in
direction. A lenient rule that accepts two unresolved halves, or two halves of
the same sign, would inflate coverage and destroy the property the papers claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from analyze_pilot_consistency import verdict  # noqa: E402


def test_resolved_halves_are_detected() -> None:
    up = np.full(12, 0.01) + np.linspace(-0.0004, 0.0004, 12)
    down = -up
    assert verdict(up, draws=1500, seed=0) == "AHEAD"
    assert verdict(down, draws=1500, seed=0) == "BELOW"
    # same direction on both halves would be the consistent case
    assert verdict(up, draws=1500, seed=0) == verdict(up * 1.5, draws=1500, seed=0)


def test_unresolved_halves_do_not_count_as_consistent() -> None:
    flat = np.zeros(12)
    assert verdict(flat, draws=1000, seed=0) == "UNRESOLVED"


def test_opposite_halves_are_inconsistent() -> None:
    up = np.full(12, 0.01)
    down = np.full(12, -0.01)
    assert verdict(up, draws=1000, seed=0) != verdict(down, draws=1000, seed=0)


def test_noise_halves_are_usually_unresolved() -> None:
    rng = np.random.default_rng(3)
    unresolved = 0
    for i in range(20):
        values = rng.normal(scale=0.004, size=12)
        unresolved += verdict(values, draws=400, seed=i) == "UNRESOLVED"
    assert unresolved >= 12, "small noisy halves should mostly fail to resolve"
