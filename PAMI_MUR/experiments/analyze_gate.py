"""Can a label-free gate tell when the router will beat pooling? (R159)

The router wins on average but loses on 39--52% of episodes, and an oracle
episode-level gate would add +.015 to +.078 over pooling (R158). This script
asks the question that decides whether that headroom is reachable: is per-episode
router correctness predictable from statistics available at decision time?

Four candidate signals are recorded per episode, each averaged over the four
selection steps:

* ``screen_margin`` --- top-two margin of the cached screen (how decisive the
  cheap view is);
* ``pred_margin`` --- top-two margin of the utility model inside the shortlist;
* ``pred_top1`` --- the utility model's top score;
* ``response_norm`` --- prediction-change norm of the candidate it picked.

For each signal the script reports the AUC for predicting "the router beats
pooling on this episode", and the realised gain of a gate that routes only when
the signal exceeds a threshold chosen on a held-out half of the episodes.

Usage
-----
python experiments/analyze_gate.py \
    --run-root /root/icl_ess_threshold/KBS_MUR/results/raw/R159_gate_20260928_1000 \
    --out results/derived/R159_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGNALS = ["gate_screen_margin", "gate_pred_margin", "gate_pred_top1", "gate_response_norm"]


def load(root: Path, dataset: str) -> dict[str, np.ndarray]:
    """Concatenate per-episode gains and gate signals over all cells."""

    gains_router, gains_pool, signals = [], [], {name: [] for name in SIGNALS}
    for cell in sorted((root / dataset).glob("target*_seed*")):
        gains_path = cell / "episode_gains.npz"
        diag_path = cell / "diagnostics.npz"
        if not gains_path.exists() or not diag_path.exists():
            continue
        g = np.load(gains_path)
        if "ranked_response_q4" not in g or "pool_all" not in g:
            continue
        d = np.load(diag_path)
        if not all(name in d for name in SIGNALS):
            continue
        gains_router.append(g["ranked_response_q4"])
        gains_pool.append(g["pool_all"])
        for name in SIGNALS:
            signals[name].append(np.asarray(d[name], dtype=np.float64).mean(axis=1))
    if not gains_router:
        return {}
    out = {
        "router": np.concatenate(gains_router),
        "pool": np.concatenate(gains_pool),
    }
    for name in SIGNALS:
        out[name] = np.concatenate(signals[name])
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, ties at 0.5."""

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks over ties
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    positives = labels.astype(bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, default=41)
    args = parser.parse_args()

    summary = {"run_root": str(args.run_root), "datasets": {}}
    for dataset_dir in sorted(p for p in args.run_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        data = load(args.run_root, dataset)
        if not data:
            continue
        router, pool = data["router"], data["pool"]
        better = router > pool
        entry = {
            "episodes": int(router.size),
            "router_mean": float(router.mean()),
            "pool_mean": float(pool.mean()),
            "router_wins_fraction": float(better.mean()),
            "oracle_gate_mean": float(np.maximum(router, pool).mean()),
            "signals": {},
        }
        # half the episodes choose the threshold, the other half realise the gate
        half = router.size // 2
        for name in SIGNALS:
            values = data[name]
            entry["signals"][name] = {"auc": auc(values, better)}
            cut_points = np.quantile(values[:half], np.linspace(0.05, 0.95, args.thresholds))
            best_cut, best_gain = None, -np.inf
            for cut in cut_points:
                route = values[:half] >= cut
                gain = float(
                    np.where(route, router[:half], pool[:half]).mean()
                )
                if gain > best_gain:
                    best_cut, best_gain = float(cut), gain
            route = values[half:] >= best_cut
            realised = float(np.where(route, router[half:], pool[half:]).mean())
            entry["signals"][name].update(
                {
                    "threshold": best_cut,
                    "held_out_gate_mean": realised,
                    "held_out_pool_mean": float(pool[half:].mean()),
                    "held_out_router_mean": float(router[half:].mean()),
                    "gate_minus_pool": realised - float(pool[half:].mean()),
                    "route_fraction": float(route.mean()),
                }
            )
        entry["best_signal"] = max(
            SIGNALS, key=lambda n: entry["signals"][n]["gate_minus_pool"], default=None
        )
        summary["datasets"][dataset] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate.json").write_text(json.dumps(summary, indent=1))

    for dataset, entry in summary["datasets"].items():
        print(f"\n=== {dataset}  ({entry['episodes']} episodes)")
        print(f"  router {entry['router_mean']:+.4f}  pool {entry['pool_mean']:+.4f}  "
              f"router wins {entry['router_wins_fraction']:.2f}  oracle gate {entry['oracle_gate_mean']:+.4f}")
        print(f"  {'signal':20s} {'AUC':>7s} {'gate mean':>10s} {'gate-pool':>10s} {'routes':>7s}")
        for name in SIGNALS:
            s = entry["signals"][name]
            print(f"  {name:20s} {s['auc']:>7.3f} {s['held_out_gate_mean']:>+10.4f} "
                  f"{s['gate_minus_pool']:>+10.4f} {s['route_fraction']:>7.2f}")
        print(f"  best signal: {entry['best_signal']}")
    print(f"\nwrote {args.out / 'gate.json'}")


if __name__ == "__main__":
    main()
