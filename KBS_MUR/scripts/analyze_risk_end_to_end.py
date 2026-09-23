#!/usr/bin/env python3
"""End-to-end selective routing from the saved trajectories.

The multi-target runs store, for every decision step of the greedy router, the
shortlisted candidate that the router chose, its response-aware score, and the
true marginal utility of every candidate at that state. A gated policy that
stops at the first step where no candidate is certified follows a prefix of
that trajectory, so its episode loss can be reconstructed exactly:

    loss_lambda(e) = base_loss(e) - sum_t m_t(e, j_t) * 1{score_t(e, j_t) >= lambda}

The identity was verified against the stored routed loss to float precision
before this script was written.

Two calibrations are evaluated on the same episodes:
  * state-mass control  -- the CRC threshold of ``analyze_risk_control.py``;
  * episode control     -- the CRC threshold for the expected negative-transfer
                           rate, which is the quantity reported below.

Outputs land in ``results/derived/R201_risk_end_to_end`` with provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/derived/R201_risk_end_to_end"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
RUN_DIRS = {
    101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
}
ALPHAS = (0.01, 0.02, 0.05, 0.10)
BASELINES = [
    ("pool_all", "Pool-all"),
    ("relevance", "Relevance"),
    ("mmr", "MMR"),
    ("static_utility", "Standalone Utility"),
    ("cached_mur", "Cached-only router"),
    ("ranked_response_q4", "Uncalibrated router"),
    ("oracle_greedy", "Sequential oracle"),
]


def cells(dataset: str) -> list[tuple[int, Path]]:
    found = []
    for seed in sorted(RUN_DIRS):
        data_dir = RUN_DIRS[seed] / dataset
        if not data_dir.exists():
            continue
        for target_dir in sorted(data_dir.glob("target*_seed*")):
            if (target_dir / "diagnostics.npz").exists():
                found.append((seed, target_dir))
    return found


def load_cell(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "diagnostics.npz") as arrays:
        data = {
            "base_loss": arrays["base_loss"].astype(np.float64),
            "choice": arrays["choice"].astype(np.int64),
            "ranked_score": arrays["ranked_score"].astype(np.float64),
            "true_marginal": arrays["true_marginal"].astype(np.float64),
            "state_mask": arrays["state_mask"],
        }
    return data


def episode_curve(cell: dict[str, np.ndarray], thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Episode losses, negative-transfer indicators and context counts per threshold."""

    steps = cell["true_marginal"].shape[0]
    episodes = cell["base_loss"].size
    base = cell["base_loss"]
    rows = np.arange(episodes)
    chosen_score = np.stack(
        [cell["ranked_score"][t][rows, cell["choice"][t]] for t in range(steps)], axis=1
    )
    chosen_marginal = np.stack(
        [cell["true_marginal"][t][rows, cell["choice"][t]] for t in range(steps)], axis=1
    )
    loss = np.empty((thresholds.size, episodes), dtype=np.float64)
    contexts = np.empty((thresholds.size, episodes), dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        selected = np.zeros((episodes, cell["true_marginal"].shape[2]), dtype=bool)
        total = np.zeros(episodes, dtype=np.float64)
        count = np.zeros(episodes, dtype=np.float64)
        for t in range(steps):
            add = (chosen_score[:, t] >= threshold) & ~selected[rows, cell["choice"][t]]
            if not np.any(add):
                continue
            total += np.where(add, chosen_marginal[:, t], 0.0)
            count += add
            selected[rows[add], cell["choice"][t][add]] = True
        loss[index] = base - total
        contexts[index] = count
    return {"loss": loss, "contexts": contexts}


def crc_threshold(losses: np.ndarray, alpha: float, *, episode_loss: bool) -> float:
    """Smallest feasible threshold for an episode-level or state-level budget."""

    order = np.argsort(-losses["score"], kind="stable")
    harmful = losses["harm"][order].astype(float)
    scores = losses["score"][order]
    n = scores.size
    cumulative = np.cumsum(harmful)
    corrected = (cumulative * (n / (n + 1)) + 1.0 / (n + 1)) / n
    feasible = np.flatnonzero(corrected <= alpha)
    if feasible.size == 0:
        return float("inf")
    last = int(feasible[-1])
    if last + 1 >= n:
        return float(scores[-1])
    return float(scores[last + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=int, default=400)
    args = parser.parse_args()

    grid = np.concatenate([np.linspace(-0.5, 1.0, args.thresholds), [np.inf]])
    summary: dict[str, dict] = {
        "provenance": {
            "run_dirs": {str(seed): str(path) for seed, path in RUN_DIRS.items()},
            "threshold_grid": [float(grid[0]), float(grid[-2]), int(args.thresholds)],
            "identity": "loss_lambda = base_loss - sum of true marginals of certified selections",
        },
        "datasets": {},
    }

    for dataset in DATASETS:
        found = cells(dataset)
        if not found:
            continue
        # target-level aggregation, split into calibration and evaluation halves
        by_target: dict[int, list[tuple[int, Path]]] = {}
        for seed, path in found:
            target = int(path.name.split("_")[0].replace("target", ""))
            by_target.setdefault(target, []).append((seed, path))
        targets = sorted(by_target)
        half = targets[: len(targets) // 2]
        calib_targets = set(half)

        per_threshold: dict[str, list[dict[str, np.ndarray]]] = {"calib": [], "eval": []}
        baselines_eval: dict[str, list[float]] = {}
        baselines_ntr: dict[str, list[float]] = {}
        state_scores: list[np.ndarray] = []
        state_truth: list[np.ndarray] = []
        for target in targets:
            for _seed, path in by_target[target]:
                cell = load_cell(path)
                curve = episode_curve(cell, grid)
                split = "calib" if target in calib_targets else "eval"
                per_threshold[split].append(curve)
                if split == "eval":
                    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
                    for key, _label in BASELINES:
                        baselines_eval.setdefault(key, []).append(
                            float(result["metrics"][key]["mean_prediction_gain"])
                        )
                        baselines_ntr.setdefault(key, []).append(
                            float(result["metrics"][key]["negative_transfer_rate"])
                        )
                if split == "calib":
                    n_steps = cell["true_marginal"].shape[0]
                    topq = np.load(path / "diagnostics.npz")["topq"]
                    score = np.take_along_axis(cell["ranked_score"], topq, axis=2).reshape(-1)
                    truth = np.take_along_axis(cell["true_marginal"], topq, axis=2).reshape(-1)
                    state_scores.append(score)
                    state_truth.append(truth)

        calib = {
            key: np.concatenate([curve[key] for curve in per_threshold["calib"]], axis=1)
            for key in ("loss", "contexts")
        }
        evaluation = {
            key: np.concatenate([curve[key] for curve in per_threshold["eval"]], axis=1)
            for key in ("loss", "contexts")
        }
        base_calib = np.concatenate(
            [load_cell(path)["base_loss"] for target in targets if target in calib_targets for _s, path in by_target[target]]
        )
        base_eval = np.concatenate(
            [load_cell(path)["base_loss"] for target in targets if target not in calib_targets for _s, path in by_target[target]]
        )

        entry: dict[str, object] = {"targets": len(targets), "episodes_eval": int(base_eval.size)}
        pooled_state_score = np.concatenate(state_scores) if state_scores else np.empty(0)
        pooled_state_truth = np.concatenate(state_truth) if state_truth else np.empty(0)
        mass_harm = (pooled_state_truth < 0.0).astype(float)
        mass_order = np.argsort(-pooled_state_score, kind="stable")
        mass_cumulative = np.cumsum(mass_harm[mass_order])
        mass_n = pooled_state_score.size
        for alpha in ALPHAS:
            # episode-level calibration on the negative-transfer indicator
            calib_gain = base_calib[None, :] - calib["loss"]
            calib_harm = (calib["loss"] > base_calib[None, :]).astype(float)
            threshold = crc_threshold(
                {"score": grid, "harm": calib_harm.mean(axis=1)}, alpha, episode_loss=True
            )
            index = int(np.argmin(np.abs(grid - threshold))) if np.isfinite(threshold) else grid.size - 1
            loss_eval = evaluation["loss"][index]
            gain = base_eval - loss_eval
            entry[f"alpha_{alpha:.2f}"] = {
                "threshold": float(grid[index]),
                "gain": float(gain.mean()),
                "negative_transfer_rate": float((loss_eval > base_eval).mean()),
                "contexts": float(evaluation["contexts"][index].mean()),
                "coverage": float((evaluation["contexts"][index] > 0).mean()),
            }
            # state-mass calibration on the same data, for comparison
            corrected = (mass_cumulative * (mass_n / (mass_n + 1)) + 1.0 / (mass_n + 1)) / mass_n
            feasible = np.flatnonzero(corrected <= alpha)
            if feasible.size and int(feasible[-1]) + 1 < mass_n:
                mass_threshold = float(pooled_state_score[mass_order][int(feasible[-1]) + 1])
            else:
                mass_threshold = float(np.max(pooled_state_score))
            mass_index = int(np.argmin(np.abs(grid - mass_threshold)))
            loss_mass = evaluation["loss"][mass_index]
            entry[f"alpha_{alpha:.2f}"]["state_mass_threshold"] = mass_threshold
            entry[f"alpha_{alpha:.2f}"]["state_mass_gain"] = float((base_eval - loss_mass).mean())
            entry[f"alpha_{alpha:.2f}"]["state_mass_ntr"] = float((loss_mass > base_eval).mean())
            entry[f"alpha_{alpha:.2f}"]["state_mass_contexts"] = float(evaluation["contexts"][mass_index].mean())

        entry["sign_gate"] = {
            "threshold": 0.0,
            "gain": float((base_eval - evaluation["loss"][int(np.argmin(np.abs(grid - 0.0)))]).mean()),
            "negative_transfer_rate": float(
                (evaluation["loss"][int(np.argmin(np.abs(grid - 0.0)))] > base_eval).mean()
            ),
            "contexts": float(evaluation["contexts"][int(np.argmin(np.abs(grid - 0.0)))].mean()),
        }
        entry["baselines"] = {
            label: {
                "gain": float(np.mean(baselines_eval[key])),
                "negative_transfer_rate": float(np.mean(baselines_ntr[key])),
            }
            for key, label in BASELINES
            if key in baselines_eval
        }
        summary["datasets"][dataset] = entry
        print(f"{dataset}: targets={len(targets)} eval episodes={base_eval.size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'summary.json'}")

    print("\nEnd-to-end selective routing (episode level):")
    print(f"{'dataset':9s} {'metric':22s} " + " ".join(f"a={a:<5g}" for a in ALPHAS))
    for dataset, entry in summary["datasets"].items():
        for metric in ("gain", "negative_transfer_rate", "contexts"):
            cells_text = " ".join(f"{entry[f'alpha_{a:.2f}'][metric]:11.4f}" for a in ALPHAS)
            print(f"{dataset:9s} {metric:22s} {cells_text}")
        print(
            f"{dataset:9s} {'uncalibrated':22s} "
            + " ".join(
                f"{entry['baselines']['Uncalibrated router'][m]:11.4f}" for m in ("gain", "negative_transfer_rate")
            )
        )


if __name__ == "__main__":
    main()
