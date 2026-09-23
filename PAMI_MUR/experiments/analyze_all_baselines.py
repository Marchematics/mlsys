#!/usr/bin/env python3
"""Unified baseline comparison on one protocol.

Merges every policy evaluated on the strong-backbone gate cells (frozen
routers, response-geometry policies, extra diversity/submodular baselines,
policy-gradient selector, and the theory-shaped variants) into one ranking with
paired confidence intervals against the strongest non-oracle baseline, so a
SOTA claim can be read directly off the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "KBS_MUR" / "scripts"))

from run_traffic_response_router_sweep import paired_ci  # noqa: E402


def load_gains(source: Path, prefix: str = "") -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Load episode gains, namespacing policy names so variants do not collide."""
    cells: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    if not source.exists():
        return cells
    for cell in sorted(source.glob("target*_seed*")):
        gains = cell / "episode_gains.npz"
        result = cell / "result.json"
        if not gains.exists() and not result.exists():
            continue
        target = int(cell.name.split("_")[0].replace("target", ""))
        seed = int(cell.name.split("_")[1].replace("seed", ""))
        entry = cells.setdefault((target, seed), {})
        if gains.exists():
            with np.load(gains) as arrays:
                for key in arrays.files:
                    entry[f"{prefix}{key}"] = arrays[key]
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[], help="extra run dir")
    parser.add_argument("--labels", default="", help="comma-separated policy names to rank")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cells = load_gains(args.gate)
    for source in args.source:
        label = source.name.split("_")[0]
        for key, entry in load_gains(source, prefix=f"{label}:").items():
            cells.setdefault(key, {}).update(entry)
    if not cells:
        raise SystemExit("no cells found")
    policies = sorted({name for entry in cells.values() for name in entry})
    print(f"cells {len(cells)}  policies {len(policies)}")

    rows = []
    for policy in policies:
        per_cell = {
            key: float(entry[policy].mean()) for key, entry in cells.items() if policy in entry
        }
        if not per_cell:
            continue
        namespace = policy.split(":")[0] + ":" if ":" in policy else ""
        reference = f"{namespace}pool_all" if f"{namespace}pool_all" in cells[next(iter(cells))] else (
            "pool_all" if "pool_all" in cells[next(iter(cells))] else None
        )
        deltas = [
            per_cell[key] - float(cells[key][reference].mean())
            for key in per_cell
            if reference and reference in cells[key]
        ]
        rows.append(
            {
                "policy": policy,
                "cells": len(per_cell),
                "mean_gain": float(np.mean(list(per_cell.values()))),
                "vs_pool_all": paired_ci(np.asarray(deltas)) if deltas else None,
            }
        )
    rows.sort(key=lambda row: -row["mean_gain"])
    table = "\n".join(
        f"{row['policy']:34s} {row['mean_gain']:+.5f}  n={row['cells']:3d}"
        + (
            f"  vs pool_all {row['vs_pool_all']['mean']:+.5f} "
            f"[{row['vs_pool_all']['low']:+.5f},{row['vs_pool_all']['high']:+.5f}]"
            if row["vs_pool_all"]
            else ""
        )
        for row in rows
    )
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "leaderboard.json").write_text(
        json.dumps({"dataset": args.dataset, "cells": len(cells), "ranking": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out / "leaderboard.txt").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
