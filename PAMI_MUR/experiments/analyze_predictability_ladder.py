#!/usr/bin/env python3
"""Does residual predictability, rather than accuracy, explain realized value?

The expert ladder shows the predicted monotone decline in realized share within
one architecture family, with the ridge expert as an exception: worst
anchor-only accuracy but the largest structural opportunity and the largest
absolute learned gain. Corollary 1 attributes identifiability to the
predictability of the predictor's residual, not to its accuracy. This script
joins the ladder with the identifiability probes run on the same cells and tests
which quantity tracks realized share.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_probe(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    probes = report.get("probes", {})
    entry = probes.get("state_plus_response_ridge") or probes.get("state_only_ridge")
    if entry is None:
        return None
    return {
        "probe_r2": entry.get("r2"),
        "probe_rank": entry.get("within_episode_spearman"),
        "probe_permutation_r2": entry.get("permutation_r2"),
        "oracle_static_gain": report.get("oracle_static_gain"),
        "random_gain": report.get("random_gain"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))["rungs"]
    rows = []
    for rung in ladder:
        probe = load_probe(args.probe_root / f"R139_probe_{rung['rung']}" / "observability.json")
        rows.append({
            "rung": rung["rung"],
            "architecture": rung["architecture"],
            "anchor_only_mse": rung["anchor_only_mse"],
            "oracle_static_gain": rung["oracle_static_gain"],
            "learned_gain": rung["learned_gain"],
            "realized_share": rung["realized_share"],
            "probe_r2": None if probe is None else probe["probe_r2"],
            "probe_rank": None if probe is None else probe["probe_rank"],
            "probe_permutation_r2": None if probe is None else probe["probe_permutation_r2"],
        })
    with_probe = [r for r in rows if r["probe_r2"] is not None]

    def corr(key_a: str, key_b: str, subset):
        a = np.asarray([r[key_a] for r in subset], dtype=float)
        b = np.asarray([r[key_b] for r in subset], dtype=float)
        if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "rows": rows,
        "probed_rungs": len(with_probe),
        "corr_realized_vs_anchor_mse": corr("anchor_only_mse", "realized_share", rows),
        "corr_realized_vs_probe_r2": corr("probe_r2", "realized_share", with_probe),
        "corr_realized_vs_probe_rank": corr("probe_rank", "realized_share", with_probe),
        "corr_learned_vs_probe_rank": corr("probe_rank", "learned_gain", with_probe),
        "corr_learned_vs_anchor_mse": corr("anchor_only_mse", "learned_gain", rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictability_vs_realized.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'rung':10s} {'anchor MSE':>10s} {'probe R2':>9s} {'probe rank':>11s} {'realized':>9s} {'learned':>9s}")
    for row in rows:
        def fmt(value):
            return "   n/a" if value is None else f"{value:+9.4f}" if abs(value) < 100 else f"{value:9.4f}"
        print(f"{row['rung']:10s} {row['anchor_only_mse']:10.4f} "
              f"{('n/a' if row['probe_r2'] is None else format(row['probe_r2'], '+9.4f')):>9s} "
              f"{('n/a' if row['probe_rank'] is None else format(row['probe_rank'], '+11.4f')):>11s} "
              f"{row['realized_share']:9.3f} {row['learned_gain']:+9.4f}")
    print()
    for key, value in summary.items():
        if key.startswith("corr_"):
            print(f"  {key:34s} {value:+.3f}")


if __name__ == "__main__":
    main()
