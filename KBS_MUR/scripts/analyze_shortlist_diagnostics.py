#!/usr/bin/env python3
"""Shortlist diagnostics: screening, gap decomposition, and sign calibration.

Reads the per-state traces saved by the multi-target routing runs and produces
the three analyses that the routing regret decomposition calls for:

* screening: recall of the truly best candidate in the top-q of the cached
  screen, and the screening regret, for q in {1,2,4,8,16};
* decomposition: how much of the one-step routing regret comes from screening
  and how much from reranking, at the deployed shortlist size;
* calibration: precision, recall and reliability of the positive-utility
  decision, plus an episode-level threshold sweep for the stopping rule.

The episode-level sweep uses the reconstruction identity that was verified
against the stored routed losses: a policy that stops at the first
uncertified step follows a prefix of the stored trajectory, so its loss is
``base_loss`` minus the sum of the true marginals of the accepted steps.

Outputs: ``results/derived/R203_shortlist_diagnostics``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/derived/R203_shortlist_diagnostics"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
RUN_DIRS = {
    101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
}
Q_VALUES = (1, 2, 4, 8, 16)
TAU_GRID = (-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def complete_runs(dataset: str) -> list[tuple[int, Path]]:
    found = []
    for seed, root in sorted(RUN_DIRS.items()):
        data_dir = root / dataset
        if not data_dir.exists():
            continue
        paths = sorted(data_dir.glob("target*_seed*/diagnostics.npz"))
        if paths:
            found.append((seed, data_dir))
    # only datasets where every run finished all 32 targets
    out = []
    for seed, data_dir in found:
        paths = sorted(data_dir.glob("target*_seed*/diagnostics.npz"))
        if len(paths) >= 32:
            out.append((seed, data_dir))
    return out


def analyse(dataset: str, stride: int) -> dict[str, object]:
    screening = {q: {"recall": [], "regret": [], "normalized_regret": []} for q in Q_VALUES}
    decomposition = {"screening": [], "reranking": [], "total": [], "oracle_q": []}
    chosen_score: list[np.ndarray] = []
    chosen_true: list[np.ndarray] = []
    cells: list[dict[str, np.ndarray]] = []
    for _seed, data_dir in complete_runs(dataset):
        for target_dir in sorted(data_dir.glob("target*_seed*")):
            path = target_dir / "diagnostics.npz"
            if not path.exists():
                continue
            with np.load(path) as arrays:
                cached = arrays["cached_score"][:, ::stride, :].astype(np.float64)
                ranked = arrays["ranked_score"][:, ::stride, :].astype(np.float64)
                true = arrays["true_marginal"][:, ::stride, :].astype(np.float64)
                state_mask = arrays["state_mask"][:, ::stride, :]
                choice = arrays["choice"][:, ::stride]
                base = arrays["base_loss"][::stride].astype(np.float64)
            steps, episodes, candidates = true.shape
            rows = np.arange(episodes)
            available = ~state_mask                                  # (T, E, K)
            masked_true = np.where(available, true, -np.inf)
            best_overall = np.argmax(masked_true, axis=2)            # (T, E)
            best_value = np.take_along_axis(masked_true, best_overall[:, :, None], axis=2)[:, :, 0]
            masked_cached = np.where(available, cached, -np.inf)
            for q in Q_VALUES:
                q_eff = min(q, candidates)
                order = np.argsort(-masked_cached, axis=2, kind="stable")[:, :, :q_eff]
                hits = (order == best_overall[:, :, None]).any(axis=2)
                screening[q]["recall"].append(hits.reshape(-1))
                best_in_q = np.take_along_axis(masked_true, order, axis=2).max(axis=2)
                regret = np.where(np.isfinite(best_value), best_value - best_in_q, 0.0)
                screening[q]["regret"].append(regret.reshape(-1))
                scale = np.maximum(np.abs(best_value), 1e-6)
                screening[q]["normalized_regret"].append((regret / scale).reshape(-1))
            q_deployed = min(4, candidates)
            order = np.argsort(-masked_cached, axis=2, kind="stable")[:, :, :q_deployed]
            best_in_q = np.take_along_axis(masked_true, order, axis=2).max(axis=2)
            chosen = np.take_along_axis(true, choice[:, :, None], axis=2)[:, :, 0]
            screening_loss = best_value - best_in_q
            reranking_loss = best_in_q - chosen
            decomposition["screening"].append(screening_loss.reshape(-1))
            decomposition["reranking"].append(reranking_loss.reshape(-1))
            decomposition["total"].append((best_value - chosen).reshape(-1))
            decomposition["oracle_q"].append(best_in_q.reshape(-1))
            chosen_score.append(np.take_along_axis(ranked, choice[:, :, None], axis=2)[:, :, 0].reshape(-1))
            chosen_true.append(chosen.reshape(-1))
            cells.append({"scores": ranked, "truth": true, "choice": choice, "base": base})

    scores = np.concatenate(chosen_score)
    truths = np.concatenate(chosen_true)
    finite = np.isfinite(scores) & np.isfinite(truths)
    scores, truths = scores[finite], truths[finite]
    positive_pred = scores > 0.0
    positive_true = truths > 0.0
    tp = int(np.count_nonzero(positive_pred & positive_true))
    fp = int(np.count_nonzero(positive_pred & ~positive_true))
    fn = int(np.count_nonzero(~positive_pred & positive_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    bins = np.quantile(scores, np.linspace(0, 1, 11))
    bins[-1] += 1e-9
    reliability = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (scores >= low) & (scores < high)
        if np.count_nonzero(mask) >= 50:
            reliability.append(
                {
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "mean_predicted": float(scores[mask].mean()),
                    "mean_true": float(truths[mask].mean()),
                    "positive_rate": float(positive_true[mask].mean()),
                }
            )

    # episode-level threshold sweep using the verified prefix identity
    sweep = {}
    for tau in TAU_GRID:
        all_gains = []
        all_contexts = []
        all_base = []
        for cell in cells:
            step_scores = cell["scores"]
            step_truth = cell["truth"]
            step_choice = cell["choice"]
            base = cell["base"]
            count = base.size
            steps, candidates = step_truth.shape[0], step_truth.shape[2]
            rows = np.arange(count)
            gains = np.zeros(count)
            contexts = np.zeros(count)
            selected = np.zeros((count, candidates), dtype=bool)
            for t in range(steps):
                score = step_scores[t][rows, step_choice[t]]
                add = (score >= tau) & ~selected[rows, step_choice[t]]
                if not np.any(add):
                    continue
                gains += np.where(add, step_truth[t][rows, step_choice[t]], 0.0)
                contexts += add
                selected[rows[add], step_choice[t][add]] = True
            all_gains.append(gains)
            all_contexts.append(contexts)
            all_base.append(base)
        gains = np.concatenate(all_gains)
        contexts = np.concatenate(all_contexts)
        base = np.concatenate(all_base)
        loss = base - gains
        sweep[f"{tau:.2f}"] = {
            "threshold": float(tau),
            "mean_gain": float(np.mean(gains)),
            "negative_transfer_rate": float(np.mean(loss > base)),
            "mean_contexts": float(np.mean(contexts)),
            "coverage": float(np.mean(contexts > 0)),
        }

    return {
        "screening": {
            str(q): {
                "recall": float(np.mean(np.concatenate(screening[q]["recall"]))),
                "mean_regret": float(np.mean(np.concatenate(screening[q]["regret"]))),
                "median_regret": float(np.median(np.concatenate(screening[q]["regret"]))),
                "mean_normalized_regret": float(np.mean(np.concatenate(screening[q]["normalized_regret"]))),
            }
            for q in Q_VALUES
        },
        "decomposition": {
            key: {
                "mean": float(np.mean(np.concatenate(values))),
                "median": float(np.median(np.concatenate(values))),
            }
            for key, values in decomposition.items()
        },
        "calibration": {
            "precision": float(precision),
            "recall": float(recall),
            "states": int(scores.size),
            "reliability": reliability,
        },
        "threshold_sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    summary: dict[str, object] = {
        "provenance": {
            "run_dirs": {str(seed): str(path) for seed, path in RUN_DIRS.items()},
            "episode_stride": args.stride,
            "q_values": list(Q_VALUES),
            "tau_grid": list(TAU_GRID),
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        if not complete_runs(dataset):
            continue
        entry = analyse(dataset, args.stride)
        summary["datasets"][dataset] = entry
        screening = entry["screening"]
        decomposition = entry["decomposition"]
        print(
            f"{dataset}: recall@4={screening['4']['recall']:.3f} "
            f"screening={decomposition['screening']['mean']:.4f} "
            f"reranking={decomposition['reranking']['mean']:.4f} "
            f"total={decomposition['total']['mean']:.4f} "
            f"precision={entry['calibration']['precision']:.3f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
