#!/usr/bin/env python3
"""Risk-controlled selective routing: calibration analysis on saved states.

Primary evidence path for the KBS reliability paper. It reads the per-state
diagnostic traces saved by the multi-target router runs (``diagnostics.npz``:
predicted scores, shortlist indices, response features, and true marginal
utilities) and evaluates calibrated lower confidence bounds for the selection
decision.

Protocols
---------
P1 within target   : calibrate on the first half of the targets, evaluate on
                     the second half of the same dataset.
P2 cross dataset   : calibrate on every other dataset, evaluate on the held-out
                     dataset.
P3 cross run       : calibrate on one independent run, evaluate on another,
                     which shifts the score distribution without changing the
                     task.
P4 radius families : one global radius, group-conditional radii on score
                     deciles, and a learned state-dependent radius, all at the
                     same nominal level.

Quantities
----------
coverage        : P(true marginal >= lower bound) over shortlisted states
harm mass       : P(selected candidate is harmful) over all states; the
                  calibrated bound controls this at the nominal level
selective risk  : fraction of selected states whose candidate is harmful
abstention      : fraction of decision states with no certified candidate
gain            : mean true marginal of the selected candidate over all states

Outputs are written to ``results/derived/R200_risk_control`` with a provenance
header naming the source run directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/derived/R200_risk_control"

DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
RUN_DIRS = {
    101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
    202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
    303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
}
ALPHAS = (0.05, 0.10, 0.20)
EPISODE_STRIDE = 16
FEATURE_NAMES = [
    "score",
    "cached_score",
    "resp_base_mean",
    "resp_after_mean",
    "resp_delta_mean",
    "resp_delta_std",
    "resp_delta_absmean",
    "resp_delta_norm",
    "resp_delta_max",
    "step",
    "selected_count",
]


def cells_with_diagnostics(dataset: str) -> dict[int, list[Path]]:
    """Run id -> diagnostic traces that exist for this dataset."""

    found: dict[int, list[Path]] = {}
    for seed, root in RUN_DIRS.items():
        data_dir = root / dataset
        if not data_dir.exists():
            continue
        paths = sorted(data_dir.glob("target*_seed*/diagnostics.npz"))
        if paths:
            found[seed] = paths
    return found


def load_cells(paths: list[Path], seed: int, stride: int) -> dict[str, np.ndarray]:
    """Flatten the shortlisted decision states of a set of diagnostic traces."""

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}
    chunks["true_marginal"] = []
    chunks["target"] = []
    chunks["seed"] = []
    for path in paths:
        target = int(path.parent.name.split("_")[0].replace("target", ""))
        with np.load(path) as arrays:
            ranked = arrays["ranked_score"][:, ::stride, :]
            cached = arrays["cached_score"][:, ::stride, :]
            true = arrays["true_marginal"][:, ::stride, :]
            topq = arrays["topq"][:, ::stride, :]
            response = arrays["response_q"][:, ::stride, :, :]
            state_mask = arrays["state_mask"][:, ::stride, :]
        steps, episodes, _ = ranked.shape
        candidates = topq.shape[2]
        values = {
            "score": np.take_along_axis(ranked, topq, axis=2).reshape(-1),
            "cached_score": np.take_along_axis(cached, topq, axis=2).reshape(-1),
            "true_marginal": np.take_along_axis(true, topq, axis=2).reshape(-1),
            "step": np.repeat(np.arange(steps), episodes * candidates).astype(np.float32),
            "selected_count": np.repeat(
                state_mask.sum(axis=2).astype(np.float32).reshape(-1), candidates
            ),
        }
        response_flat = response.reshape(-1, response.shape[-1])
        for index, name in enumerate(FEATURE_NAMES[2:9]):
            values[name] = response_flat[:, index]
        for name, array in values.items():
            chunks[name].append(np.asarray(array, dtype=np.float32))
        chunks["target"].append(np.full(values["score"].shape, target, dtype=np.int32))
        chunks["seed"].append(np.full(values["score"].shape, seed, dtype=np.int32))
    return {name: np.concatenate(parts) for name, parts in chunks.items() if parts}


def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal radius with the finite-sample corrected level."""

    n = residuals.size
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def adaptive_radius(features: np.ndarray, residuals: np.ndarray, alpha: float, *, seed: int = 0):
    """Gradient-boosted quantile regression of |residual| on state features."""

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=1.0 - alpha,
        max_depth=3,
        max_iter=150,
        learning_rate=0.08,
        l2_regularization=1e-2,
        random_state=seed,
    )
    model.fit(features, residuals)
    return model


def evaluate(radius, score: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Coverage, harm, abstention and gain of the calibrated gate."""

    radius = np.asarray(radius, dtype=float) if not np.isscalar(radius) else radius
    lower = score - radius
    covered = true >= lower
    select = lower > 0.0
    harmful = true < 0.0
    selected = int(np.count_nonzero(select))
    harmful_selected = int(np.count_nonzero(select & harmful))
    return {
        "states": int(true.size),
        "coverage": float(np.mean(covered)),
        "harm_mass": float(np.mean(select & harmful)),
        "selective_risk": float(harmful_selected / selected) if selected else 0.0,
        "abstention": float(1.0 - selected / true.size),
        "gain": float(np.mean(np.where(select, true, 0.0))),
        "selected_fraction": float(selected / true.size),
    }


def threshold_control(score: np.ndarray, true: np.ndarray, selected_fraction: float) -> dict[str, float]:
    """Free-threshold control matched to the abstaining policy's coverage."""

    if not 0.0 < selected_fraction < 1.0:
        return {}
    threshold = float(np.quantile(score, 1.0 - selected_fraction))
    select = score > threshold
    selected = int(np.count_nonzero(select))
    harmful = int(np.count_nonzero(select & (true < 0.0)))
    return {
        "threshold": threshold,
        "gain": float(np.mean(np.where(select, true, 0.0))),
        "selective_risk": float(harmful / selected) if selected else 0.0,
        "harm_mass": float(np.mean(select & (true < 0.0))),
        "selected_fraction": float(selected / true.size),
    }


def crc_threshold(score: np.ndarray, true: np.ndarray, alpha: float) -> float:
    """Conformal risk control threshold for the expected harmful-selection mass.

    The decision rule selects where ``score >= lambda``; the loss
    ``1{selected and harmful}`` is monotone non-increasing in ``lambda``, so the
    finite-sample corrected empirical risk minimiser is the smallest threshold
    whose corrected risk is at most ``alpha``.
    """

    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    harmful = (true[order] < 0.0).astype(float)
    n = sorted_score.size
    if n == 0:
        return float("inf")
    cumulative = np.cumsum(harmful)                      # harmful mass if we select the first k
    corrected = (cumulative * (n / (n + 1)) + 1.0 / (n + 1)) / n
    feasible = np.flatnonzero(corrected <= alpha)
    if feasible.size == 0:
        return float("inf")
    last = int(feasible[-1])                             # largest prefix that is still safe
    if last + 1 >= n:
        return float(sorted_score[-1])
    return float(sorted_score[last + 1])


def crc_report(threshold: float, score: np.ndarray, true: np.ndarray) -> dict[str, float]:
    if not np.isfinite(threshold):
        return {
            "threshold": float("inf"),
            "states": int(true.size),
            "harm_mass": 0.0,
            "selective_risk": 0.0,
            "abstention": 1.0,
            "gain": 0.0,
            "selected_fraction": 0.0,
        }
    select = score >= threshold
    selected = int(np.count_nonzero(select))
    harmful = int(np.count_nonzero(select & (true < 0.0)))
    return {
        "threshold": float(threshold),
        "states": int(true.size),
        "harm_mass": float(np.mean(select & (true < 0.0))),
        "selective_risk": float(harmful / selected) if selected else 0.0,
        "abstention": float(1.0 - selected / true.size),
        "gain": float(np.mean(np.where(select, true, 0.0))),
        "selected_fraction": float(selected / true.size),
    }


def subsample(table: dict[str, np.ndarray], max_rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    size = table["true_marginal"].size
    if size <= max_rows:
        return table
    keep = rng.choice(size, size=max_rows, replace=False)
    return {key: value[keep] for key, value in table.items()}


def feature_matrix(table: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([table[name] for name in FEATURE_NAMES], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=EPISODE_STRIDE)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    availability = {dataset: cells_with_diagnostics(dataset) for dataset in DATASETS}
    availability = {dataset: runs for dataset, runs in availability.items() if runs}
    if not availability:
        raise SystemExit("no dataset has saved diagnostics")
    print("runs with diagnostics:", {k: sorted(v) for k, v in availability.items()})

    # Analysis run: prefer the most complete run per dataset.
    data: dict[str, dict[str, np.ndarray]] = {}
    for dataset, runs in availability.items():
        primary = max(runs, key=lambda seed: len(runs[seed]))
        table = load_cells(runs[primary], primary, args.stride)
        data[dataset] = subsample(table, args.max_rows, rng)
    for dataset, table in data.items():
        print(
            f"{dataset:9s} states={table['true_marginal'].size:8d} "
            f"targets={np.unique(table['target']).size:3d} run={int(table['seed'][0])}"
        )

    summary: dict[str, dict] = {
        "provenance": {
            "run_dirs": {str(seed): str(path) for seed, path in RUN_DIRS.items()},
            "episode_stride": args.stride,
            "max_rows_per_dataset": args.max_rows,
            "feature_names": FEATURE_NAMES,
        },
        "protocols": {},
    }

    # ------------------------------------------------------------------ P1
    p1: dict[str, dict] = {}
    for dataset, table in data.items():
        targets = np.unique(table["target"])
        half = targets[: targets.size // 2]
        calib = np.isin(table["target"], half)
        evaluation = ~calib
        residuals = np.abs(table["score"][calib] - table["true_marginal"][calib])
        entry: dict[str, dict] = {}
        for alpha in ALPHAS:
            radius = conformal_radius(residuals, alpha)
            result = evaluate(radius, table["score"][evaluation], table["true_marginal"][evaluation])
            result["radius"] = radius
            control = threshold_control(
                table["score"][evaluation], table["true_marginal"][evaluation], result["selected_fraction"]
            )
            if control:
                result["matched_threshold"] = control
            entry[f"alpha_{alpha:.2f}"] = result
        entry["uncalibrated"] = evaluate(0.0, table["score"][evaluation], table["true_marginal"][evaluation])
        p1[dataset] = entry
    summary["protocols"]["P1_within_target"] = p1

    # ------------------------------------------------------------------ P2
    p2: dict[str, dict] = {}
    for held_out in data:
        parts = [
            np.abs(data[dataset]["score"] - data[dataset]["true_marginal"])
            for dataset in data
            if dataset != held_out
        ]
        residuals = np.concatenate(parts)
        entry = {}
        for alpha in ALPHAS:
            radius = conformal_radius(residuals, alpha)
            result = evaluate(radius, data[held_out]["score"], data[held_out]["true_marginal"])
            result["radius"] = radius
            entry[f"alpha_{alpha:.2f}"] = result
        p2[held_out] = entry
    summary["protocols"]["P2_cross_dataset"] = p2

    # ------------------------------------------------------------------ P3
    p3: dict[str, dict] = {}
    for dataset, runs in availability.items():
        seeds = sorted(runs)
        if len(seeds) < 2:
            continue
        calib_seed, eval_seed = seeds[0], seeds[-1]
        calib_table = subsample(load_cells(runs[calib_seed], calib_seed, args.stride), args.max_rows, rng)
        eval_table = subsample(load_cells(runs[eval_seed], eval_seed, args.stride), args.max_rows, rng)
        residuals = np.abs(calib_table["score"] - calib_table["true_marginal"])
        entry = {"calibration_run": calib_seed, "evaluation_run": eval_seed, "levels": {}}
        for alpha in ALPHAS:
            radius = conformal_radius(residuals, alpha)
            result = evaluate(radius, eval_table["score"], eval_table["true_marginal"])
            result["radius"] = radius
            entry["levels"][f"alpha_{alpha:.2f}"] = result
        entry["levels"]["uncalibrated"] = evaluate(0.0, eval_table["score"], eval_table["true_marginal"])
        p3[dataset] = entry
    summary["protocols"]["P3_cross_run"] = p3

    # ------------------------------------------------------------------ P4
    p4: dict[str, dict] = {}
    for dataset, table in data.items():
        features = feature_matrix(table)
        targets = np.unique(table["target"])
        half = targets[: targets.size // 2]
        calib = np.isin(table["target"], half)
        evaluation = ~calib
        residuals_all = np.abs(table["score"] - table["true_marginal"])
        entry = {}
        for alpha in ALPHAS:
            global_radius = conformal_radius(residuals_all[calib], alpha)
            global_result = evaluate(global_radius, table["score"][evaluation], table["true_marginal"][evaluation])

            edges = np.quantile(table["score"][calib], np.linspace(0, 1, 11)[1:-1])
            bins_calib = np.digitize(table["score"][calib], edges)
            bins_eval = np.digitize(table["score"][evaluation], edges)
            radius_per_bin = np.full(11, global_radius, dtype=float)
            for bin_index in range(11):
                mask = bins_calib == bin_index
                if np.count_nonzero(mask) >= 50:
                    radius_per_bin[bin_index] = conformal_radius(residuals_all[calib][mask], alpha)
            conditional_result = evaluate(
                radius_per_bin[bins_eval],
                table["score"][evaluation],
                table["true_marginal"][evaluation],
            )

            model = adaptive_radius(features[calib], residuals_all[calib], alpha)
            adaptive_result = evaluate(
                model.predict(features[evaluation]),
                table["score"][evaluation],
                table["true_marginal"][evaluation],
            )

            entry[f"alpha_{alpha:.2f}"] = {
                "global": global_result,
                "conditional": conditional_result,
                "adaptive": adaptive_result,
                "global_radius": global_radius,
                "bin_radii": radius_per_bin.tolist(),
            }
        p4[dataset] = entry
    summary["protocols"]["P4_radius_families"] = p4

    # ------------------------------------------------------------------ P5
    p5: dict[str, dict] = {}
    for dataset, table in data.items():
        features = feature_matrix(table)
        targets = np.unique(table["target"])
        half = targets[: targets.size // 2]
        calib = np.isin(table["target"], half)
        evaluation = ~calib
        eval_true = table["true_marginal"][evaluation]
        eval_score = table["score"][evaluation]
        eval_target = table["target"][evaluation]
        codes = np.searchsorted(targets, eval_target)
        entry = {}
        for alpha in (0.005, 0.01, 0.02, 0.05):
            global_threshold = crc_threshold(table["score"][calib], table["true_marginal"][calib], alpha)
            global_result = crc_report(global_threshold, eval_score, eval_true)
            global_result["ci"] = bootstrap_ci(eval_score, eval_true, codes, targets.size, global_threshold)

            edges = np.quantile(table["score"][calib], np.linspace(0, 1, 6)[1:-1])
            bins_calib = np.digitize(table["score"][calib], edges)
            bins_eval = np.digitize(eval_score, edges)
            conditional_thresholds = np.full(5, global_threshold, dtype=float)
            for bin_index in range(5):
                mask = bins_calib == bin_index
                if np.count_nonzero(mask) >= 200:
                    conditional_thresholds[bin_index] = crc_threshold(
                        table["score"][calib][mask], table["true_marginal"][calib][mask], alpha
                    )
            conditional_select = eval_score >= conditional_thresholds[bins_eval]
            conditional_harm = int(np.count_nonzero(conditional_select & (eval_true < 0.0)))
            conditional_selected = int(np.count_nonzero(conditional_select))
            conditional_result = {
                "thresholds": conditional_thresholds.tolist(),
                "harm_mass": float(np.mean(conditional_select & (eval_true < 0.0))),
                "selective_risk": float(conditional_harm / conditional_selected) if conditional_selected else 0.0,
                "abstention": float(1.0 - conditional_selected / eval_true.size),
                "gain": float(np.mean(np.where(conditional_select, eval_true, 0.0))),
                "selected_fraction": float(conditional_selected / eval_true.size),
            }

            interval_radius = conformal_radius(
                np.abs(table["score"][calib] - table["true_marginal"][calib]), alpha
            )
            interval_result = evaluate(interval_radius, eval_score, eval_true)
            interval_result["radius"] = interval_radius

            matched = threshold_control(eval_score, eval_true, global_result["selected_fraction"])
            matched["basis"] = "free-threshold control at the same selected fraction"

            oracle_select = eval_true > 0.0
            oracle_result = {
                "harm_mass": 0.0,
                "abstention": float(1.0 - np.mean(oracle_select)),
                "gain": float(np.mean(np.where(oracle_select, eval_true, 0.0))),
                "selected_fraction": float(np.mean(oracle_select)),
            }

            entry[f"alpha_{alpha:.3f}"] = {
                "global": global_result,
                "conditional": conditional_result,
                "interval_lcb": interval_result,
                "matched_threshold": matched,
                "oracle_positive": oracle_result,
            }
        entry["uncalibrated"] = evaluate(0.0, eval_score, eval_true)
        entry["always_select"] = {
            "states": int(eval_true.size),
            "coverage": 1.0,
            "selected_fraction": 1.0,
            "harm_mass": float(np.mean(eval_true < 0.0)),
            "selective_risk": float(np.mean(eval_true < 0.0)),
            "gain": float(np.mean(eval_true)),
            "abstention": 0.0,
        }
        p5[dataset] = entry
    summary["protocols"]["P5_conformal_risk_control"] = p5

    # ------------------------------------------------------------------ P6
    p6: dict[str, dict] = {}
    for held_out in data:
        parts_score = [data[dataset]["score"] for dataset in data if dataset != held_out]
        parts_true = [data[dataset]["true_marginal"] for dataset in data if dataset != held_out]
        calib_score = np.concatenate(parts_score)
        calib_true = np.concatenate(parts_true)
        entry = {}
        for alpha in (0.005, 0.01, 0.02, 0.05):
            threshold = crc_threshold(calib_score, calib_true, alpha)
            entry[f"alpha_{alpha:.3f}"] = crc_report(
                threshold, data[held_out]["score"], data[held_out]["true_marginal"]
            )
        p6[held_out] = entry
    summary["protocols"]["P6_cross_dataset_threshold"] = p6

    # ------------------------------------------------------------------ P7
    p7: dict[str, dict] = {}
    for dataset, table in data.items():
        targets = np.unique(table["target"])
        half = targets[: targets.size // 2]
        calib = np.isin(table["target"], half)
        evaluation = ~calib
        eval_score = table["score"][evaluation]
        eval_true = table["true_marginal"][evaluation]
        entry = {}
        for alpha in (0.02, 0.05, 0.10):
            share_threshold = share_control_threshold(
                table["score"][calib], table["true_marginal"][calib], alpha
            )
            share_result = crc_report(share_threshold, eval_score, eval_true)
            mass_threshold = crc_threshold(table["score"][calib], table["true_marginal"][calib], alpha)
            mass_result = crc_report(mass_threshold, eval_score, eval_true)
            entry[f"alpha_{alpha:.2f}"] = {"share_control": share_result, "mass_control": mass_result}
        entry["risk_coverage"] = risk_coverage_curve(
            eval_score, eval_true, (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
        )
        dense = np.linspace(0.01, 1.0, 100)
        order = np.argsort(-eval_score, kind="stable")
        harmful_sorted = (eval_true[order] < 0.0).astype(float)
        cumulative = np.cumsum(harmful_sorted)
        counts = np.maximum((dense * harmful_sorted.size).astype(int), 1)
        entry["risk_coverage_dense"] = {
            "coverage": dense.tolist(),
            "harmful_share": (cumulative[counts - 1] / counts).tolist(),
        }
        entry["uncalibrated"] = evaluate(0.0, eval_score, eval_true)
        p7[dataset] = entry
    summary["protocols"]["P7_share_vs_mass"] = p7

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'summary.json'}")
    print_table(summary)


def bootstrap_ci(
    score: np.ndarray,
    true: np.ndarray,
    codes: np.ndarray,
    target_count: int,
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Target-level bootstrap intervals for the CRC rule's realised quantities."""

    if not np.isfinite(threshold):
        return {"harm_mass": [0.0, 0.0], "selected_fraction": [0.0, 0.0], "gain": [0.0, 0.0]}
    select = score >= threshold
    harmful = select & (true < 0.0)
    gains = np.where(select, true, 0.0)
    counts = np.bincount(codes, minlength=target_count).astype(float)
    harm_per_target = np.bincount(codes, weights=harmful.astype(float), minlength=target_count)
    select_per_target = np.bincount(codes, weights=select.astype(float), minlength=target_count)
    gain_per_target = np.bincount(codes, weights=gains.astype(float), minlength=target_count)
    rng = np.random.default_rng(seed)
    out = {"harm_mass": [], "selected_fraction": [], "gain": []}
    for _ in range(n_boot):
        draw = rng.integers(0, target_count, size=target_count)
        total = counts[draw].sum()
        if total <= 0:
            continue
        out["harm_mass"].append(float(harm_per_target[draw].sum() / total))
        out["selected_fraction"].append(float(select_per_target[draw].sum() / total))
        out["gain"].append(float(gain_per_target[draw].sum() / total))
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in out.items()
        if values
    }


def share_control_threshold(
    score: np.ndarray,
    true: np.ndarray,
    alpha: float,
    *,
    grid: int = 200,
    delta: float = 0.05,
) -> float:
    """Threshold whose harmful share is certified at ``alpha`` with confidence ``1-delta``.

    Accepted states are those with ``score >= lambda``. Among them the harmful
    indicator is Bernoulli with mean equal to the selective risk, so a Hoeffding
    upper bound with a union bound over the threshold grid certifies the share.
    The smallest feasible threshold maximises coverage.
    """

    thresholds = np.quantile(score, np.linspace(0.0, 1.0, grid))
    penalty = np.sqrt(np.log(max(grid, 2) / delta) / 2.0)
    for threshold in thresholds:
        accepted = score >= threshold
        count = int(np.count_nonzero(accepted))
        if count == 0:
            continue
        harmful = int(np.count_nonzero(accepted & (true < 0.0)))
        share = harmful / count
        if share + penalty / np.sqrt(count) <= alpha:
            return float(threshold)
    return float("inf")


def risk_coverage_curve(score: np.ndarray, true: np.ndarray, coverages) -> dict[str, float]:
    """Smallest harmful share achievable by any score threshold at each coverage.

    Sorting by score descending and reading the cumulative harmful rate gives
    the risk--coverage frontier: no threshold rule using this score can do
    better at the same coverage.
    """

    order = np.argsort(-score, kind="stable")
    harmful = (true[order] < 0.0).astype(float)
    cumulative = np.cumsum(harmful)
    n = harmful.size
    out = {}
    for coverage in coverages:
        k = max(1, int(round(coverage * n)))
        out[f"{coverage:.2f}"] = float(cumulative[k - 1] / k)
    return out


def print_table(summary: dict) -> None:
    print("\nP1 within-target (alpha = .10), calibrated gate vs uncalibrated:")
    print(f"{'dataset':9s} {'radius':>8s} {'coverage':>9s} {'sel_risk':>9s} {'harm_mass':>10s} {'abstain':>8s} {'gain':>8s}")
    for dataset, entry in summary["protocols"]["P1_within_target"].items():
        result = entry["alpha_0.10"]
        print(
            f"{dataset:9s} {result['radius']:8.4f} {result['coverage']:9.4f} {result['selective_risk']:9.4f} "
            f"{result['harm_mass']:10.4f} {result['abstention']:8.4f} {result['gain']:8.4f}"
        )
    print("\nP2 cross-dataset transfer (alpha = .10):")
    print(f"{'dataset':9s} {'radius':>8s} {'coverage':>9s} {'harm_mass':>10s} {'abstain':>8s}")
    for dataset, entry in summary["protocols"]["P2_cross_dataset"].items():
        result = entry["alpha_0.10"]
        print(
            f"{dataset:9s} {result['radius']:8.4f} {result['coverage']:9.4f} "
            f"{result['harm_mass']:10.4f} {result['abstention']:8.4f}"
        )
    print("\nP4 radius families (alpha = .10): coverage / selective risk / abstention")
    for dataset, entry in summary["protocols"]["P4_radius_families"].items():
        level = entry["alpha_0.10"]
        row = " ".join(
            f"{name}:{level[name]['coverage']:.3f}/{level[name]['selective_risk']:.3f}/{level[name]['abstention']:.3f}"
            for name in ("global", "conditional", "adaptive")
        )
        print(f"{dataset:9s} {row}")
    print("\nP5 conformal risk control: alpha / harm mass / selected fraction / selective risk / gain")
    for dataset, entry in summary["protocols"]["P5_conformal_risk_control"].items():
        cells = []
        for alpha in (0.005, 0.01, 0.02, 0.05):
            level = entry[f"alpha_{alpha:.3f}"]["global"]
            cells.append(
                f"{alpha:.3f}:{level['harm_mass']:.4f}/{level['selected_fraction']:.3f}/"
                f"{level['selective_risk']:.3f}/{level['gain']:.4f}"
            )
        print(f"{dataset:9s} " + "  ".join(cells))
    print("\nP6 cross-dataset threshold transfer (alpha = .02): harm mass vs nominal")
    for dataset, entry in summary["protocols"]["P6_cross_dataset_threshold"].items():
        level = entry["alpha_0.020"]
        print(
            f"{dataset:9s} threshold={level['threshold']:.4f} harm_mass={level['harm_mass']:.4f} "
            f"selected={level['selected_fraction']:.3f} selective_risk={level['selective_risk']:.3f}"
        )


if __name__ == "__main__":
    main()
