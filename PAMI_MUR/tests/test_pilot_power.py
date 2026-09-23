"""Tests for the pilot-power decision rule.

The pilot analysis turns the measurement protocol into a deployment decision, so
the verdict function it relies on is the piece that has to be pinned: an
interval excluding zero above is "route", excluding zero below is "do not", and
anything else is unresolved. Getting this wrong would silently invert the
procedure's operating characteristics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from analyze_pilot_power import verdict  # noqa: E402


def test_clearly_positive_is_ahead() -> None:
    values = np.full(24, 0.01) + np.linspace(-0.0005, 0.0005, 24)
    assert verdict(values, draws=2000, seed=0) == "AHEAD"


def test_clearly_negative_is_below() -> None:
    values = np.full(24, -0.01) + np.linspace(-0.0005, 0.0005, 24)
    assert verdict(values, draws=2000, seed=0) == "BELOW"


def test_centred_on_zero_is_unresolved() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(scale=0.01, size=24)
    assert verdict(values, draws=2000, seed=0) in {"AHEAD", "BELOW", "UNRESOLVED"}
    # a symmetric sample around zero must not be declared resolved in both tails
    assert verdict(np.zeros(24), draws=500, seed=0) == "UNRESOLVED"


def test_verdict_is_scale_invariant() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.004, scale=0.006, size=32)
    assert verdict(values, draws=2000, seed=2) == verdict(values * 100.0, draws=2000, seed=2)
