#!/usr/bin/env python3
"""Generate per-seed LaTeX tables for the KBS supplement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "supplement_tables.tex"


def fmt(value: float, digits: int = 3) -> str:
    """Format without a leading zero, matching the manuscript tables."""

    if value is None or not np.isfinite(value):
        return "--"
    text = f"{value:.{digits}f}"
    if text.startswith("-0"):
        return "-" + text[2:]
    if text.startswith("0"):
        return text[1:]
    return text


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def r070_table() -> str:
    raw = load_json(ROOT / "results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json")
    rows = []
    for record in sorted(raw["records"], key=lambda r: (float(r["redundancy"]), int(r["seed"]))):
        metrics = record["metrics"]
        os = metrics["oracle_static"]["mean_prediction_gain"]
        mur = metrics["mur_light"]["mean_prediction_gain"]
        og = metrics["oracle"]["mean_prediction_gain"]
        rows.append(
            f"{float(record['redundancy']):g} & {int(record['seed'])} & "
            f"{fmt(os)} & {fmt(mur)} & {fmt(og)} & {fmt((og - os) / og)} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed controlled redundancy ladder at $K=16$ and $B=4$. Gains are loss reductions relative to the anchor-only predictor.}}
\label{{tab:supp-redundancy}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
\toprule
$r_{{\mathrm{{dup}}}}$ & Seed & Oracle-Static & R-MUR & Oracle-Greedy & $H_{{\mathrm{{state}}}}$\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def r071_table(raw_root: Path) -> str:
    audit_path = ROOT / "results/derived/R071_margin_audit/per_seed.csv"
    audit = {}
    if audit_path.exists():
        for row in csv.DictReader(audit_path.open()):
            audit[(int(row["candidate_count"]), int(row["seed"]))] = row
    rows = []
    for k_dir in sorted(raw_root.glob("k*"), key=lambda p: int(p.name[1:])):
        k = int(k_dir.name[1:])
        for result_path in sorted(k_dir.glob("*/result.json")):
            result = load_json(result_path)
            seed = int(result["config"]["seed"])
            mur = result["metrics"]["mur_light"]
            diag = result["state_diagnostics"]["mur_rollout"]
            extra = audit.get((k, seed))
            nmae = fmt(float(extra["nmae_sd"])) if extra else "--"
            decision = fmt(float(extra["decision_accuracy"])) if extra else "--"
            margin = fmt(float(extra["mean_positive_top2_margin"])) if extra else "--"
            strict = fmt(float(extra["strict_top1_accuracy"])) if extra else fmt(diag["top1_accuracy"], 4)
            regret = fmt(float(extra["mean_regret"]), 4) if extra else fmt(diag["one_step_regret"], 4)
            rows.append(
                f"{k} & {seed} & {fmt(mur['mean_utility_recovery'])} & "
                f"{nmae} & {margin} & {strict} & "
                f"{decision} & {regret} \\\\"
            )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed candidate-count scaling under fixed redundancy. Normalized MAE divides pointwise error by the standard deviation of the true marginal targets. Decision accuracy counts a selection as correct when it attains the best true utility, including exact ties.}}
\label{{tab:supp-candidate-scaling}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{rcccccccc}}
\toprule
$K$ & Seed & R-MUR recovery & Normalized MAE & Positive margin & Strict top-1 & Decision accuracy & One-step regret\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""

def r072_table() -> str:
    path = ROOT / "results/raw/R072_mur_conservative_h025_h050_a005_a050_s5_20260920_0000/summary.csv"
    rows = list(csv.DictReader(path.open()))
    lines = []
    for harm in ("0.25", "0.5"):
        subset = [r for r in rows if r["harmful_fraction"] == harm and r["policy"] == "mur_interval"]
        subset.sort(key=lambda r: float(r["alpha"]))
        for row in subset:
            lines.append(
                f"{harm} & {float(row['alpha']):.2f} & "
                f"{fmt(float(row['mean_prediction_gain_mean']))} & "
                f"{fmt(float(row['negative_transfer_rate_mean']), 4)} & "
                f"{fmt(float(row['mean_selected_contexts_mean']))} \\\\"
            )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Calibrated-stopping levels aggregated over five seeds. Each row reports a harmful-candidate fraction and calibration level.}}
\label{{tab:supp-conservative}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
$r_{{\mathrm{{harm}}}}$ & $\alpha$ & Prediction gain & Negative-transfer rate & Selected contexts\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def real_data_table() -> str:
    lines = []
    for path in sorted((ROOT / "results/raw").glob("R073_electricity_oracle_audit_s*/result.json")):
        result = load_json(path)
        q = result["oracle_quantities"]
        seed = int(result["config"]["seed"])
        lines.append(
            f"Electricity & {seed} & {fmt(q['gain_base_to_oracle_static'])} & "
            f"{fmt(q['gain_base_to_oracle_greedy'])} & {fmt(q['state_conditioning_headroom'])} \\\\"
        )
    traffic_oracle = load_json(
        ROOT / "results/raw/R074_traffic_oracle_gate_metrla_k16_b4_s5_20260920_0100/aggregate.json"
    )
    for record in sorted(traffic_oracle["records"], key=lambda r: int(r["config"]["seed"])):
        q = record["oracle_quantities"]
        seed = int(record["config"]["seed"])
        lines.append(
            f"METR-LA & {seed} & {fmt(q['gain_base_to_oracle_static'])} & "
            f"{fmt(q['gain_base_to_oracle_greedy'])} & {fmt(q['state_conditioning_headroom'])} \\\\"
        )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed real-data oracle audits for the sensor-level traffic protocol and the client-load boundary audit.}}
\label{{tab:supp-real}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llccc}}
\toprule
Task & Seed & Oracle-Static & Oracle-Greedy & $H_{{\mathrm{{state}}}}$\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def electricity_diagnostic_table() -> str:
    lines = []
    for path in sorted((ROOT / "results/raw").glob("R073_electricity_oracle_audit_s*/result.json")):
        result = load_json(path)
        q = result["oracle_quantities"]
        lines.append(
            f"{int(result['config']['seed'])} & {fmt(q['exact_delta_gain'])} \\\\"
        )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed Electricity exact-response diagnostic gain. The diagnostic observes counterfactual prediction changes and is separate from the cached router.}}
\label{{tab:supp-electricity-diagnostic}}
\begin{{tabular}}{{cc}}
\toprule
Seed & Exact-response R-MUR gain\\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""

def r075_table() -> str:
    path = ROOT / "results/raw/R075_traffic_full_router_metrla_k16_b4_s5_20260920_0115/aggregate.json"
    raw = load_json(path)
    lines = []
    for record in sorted(raw["records"], key=lambda r: int(r["config"]["seed"])):
        metrics = record["metrics"]
        lines.append(
            f"{int(record['config']['seed'])} & {fmt(metrics['static_utility']['mean_prediction_gain'])} & "
            f"{fmt(metrics['mur']['mean_prediction_gain'])} & "
            f"{fmt(metrics['static_utility']['negative_transfer_rate'], 4)} & "
            f"{fmt(metrics['mur']['negative_transfer_rate'], 4)} \\\\"
        )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed METR-LA cached routing comparison.}}
\label{{tab:supp-cached-router}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{ccccc}}
\toprule
Seed & Standalone Utility gain & Cached-only router gain & Standalone Utility negative-transfer rate & Cached-only router negative-transfer rate\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def r076_table() -> str:
    path = ROOT / "results/raw/R076_traffic_utility_observability_s5_20260920_0300/per_seed.csv"
    rows = list(csv.DictReader(path.open()))
    probe_labels = {"x1": r"$\mathcal{X}_1$", "x2": r"$\mathcal{X}_2$", "x3": r"$\mathcal{X}_3$"}
    body = "\n".join(
        f"{row['seed']} & {probe_labels.get(row['probe'], row['probe'])} & {fmt(float(row['mae']), 4)} & "
        f"{fmt(float(row['spearman']))} & {fmt(float(row['positive_auc']))} & "
        f"{fmt(float(row['top1_accuracy']))} & {fmt(float(row['one_step_regret']), 4)} \\\\"
        for row in rows
    )
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed METR-LA observability probe. $\mathcal{{X}}_1$ uses cached context features, $\mathcal{{X}}_2$ adds scalar response summaries, and $\mathcal{{X}}_3$ adds full forecast trajectories. MAE is the mean absolute error and AUC is the area under the receiver operating characteristic curve.}}
\label{{tab:supp-observability}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llccccc}}
\toprule
Seed & Input & MAE & Spearman & Positive AUC & Top-1 & Regret\\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def r079_table() -> str:
    path = ROOT / "results/derived/R079_six_dataset_summary/summary.csv"
    columns = [
        ("static_utility", "Standalone Utility"),
        ("cached_mur", "Cached-only router"),
        ("ranked_rmur_q2", "$q=2$"),
        ("ranked_rmur_q4", "$q=4$"),
        ("ranked_rmur_q8", "$q=8$"),
        ("ranked_rmur_full", "$q=K$"),
    ]
    rows = []
    means = {key: [] for key, _ in columns}
    for row in csv.DictReader(path.open()):
        cells = []
        for key, _label in columns:
            value = float(row[key])
            means[key].append(value)
            cells.append(fmt(value))
        rows.append(f"{row['dataset']} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("Mean & " + " & ".join(fmt(float(np.mean(means[key]))) for key, _ in columns) + " \\\\")
    rows.append("Predictor evaluations & 0 & 0 & 12.5\\% & 25\\% & 50\\% & 100\\% \\\\")
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Shortlist-size consistency check under the single-target protocol with five runs. Values are prediction gains; the final row reports the share of the candidate pool that receives a predictor evaluation.}}
\label{{tab:supp-shortlist}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
Dataset & Standalone & Cached-only & $q=2$ & $q=4$ & $q=8$ & $q=K$\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""



def r070_ladder_table() -> str:
    import json as _json

    path = ROOT / "results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json"
    records = _json.loads(path.read_text())["records"]
    ratios = sorted({float(r["redundancy"]) for r in records})
    rows = []
    for ratio in ratios:
        subset = [r for r in records if float(r["redundancy"]) == ratio]

        def mean(policy):
            values = np.asarray([r["metrics"][policy]["mean_prediction_gain"] for r in subset], dtype=float)
            return f"{values.mean():.4f}".lstrip("0")

        headroom = np.asarray([r["metrics"]["state_conditioning_headroom"] for r in subset], dtype=float)
        rows.append(
            f"{ratio:g} & {mean('oracle_static')} & {mean('mur_light')} & {mean('oracle')} & "
            f"{f'{headroom.mean():.4f}'.lstrip('0')} " + "\\\\"
        )
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Controlled redundancy ladder at $K=16$ and context budget four, averaged over five runs. Gains are loss reductions relative to the anchor-only predictor.}}
\label{{tab:supp-redundancy-ladder}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Duplicate ratio & Standalone oracle & R-MUR & Sequential oracle & Structural headroom\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def r080a_table() -> str:
    path = ROOT / "results/derived/R080a_multitarget_summary/summary.csv"
    rows = []
    total = 0
    for row in csv.DictReader(path.open()):
        total += int(row["seed_target_runs"])
        head_mean = f"{float(row['H_mean']):.3f}".lstrip("0")
        head_median = f"{float(row['H_median']):.3f}".lstrip("0")
        rows.append(
            f"{row['dataset']} & {head_mean} & {head_median} & "
            f"{float(row['positive_target_fraction']):.3f} " + "\\\\"
        )
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Structural headroom per dataset over 32 fixed targets and five runs ({total} target-run evaluations). Every evaluation has positive headroom.}}
\label{{tab:supp-headroom}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrr}}
\toprule
Dataset & Mean headroom & Median headroom & Positive fraction\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""



def subset_baseline_table() -> str:
    """Four content-based subset-selection baselines over the multi-target protocol."""

    import json as _json

    roots = sorted((ROOT / "results/raw").glob("R202_subset_baselines_s3_*"))
    if not roots:
        return ""
    root = roots[-1]
    policies = [
        ("facility_location", "Facility location"),
        ("dpp_greedy", "DPP greedy"),
        ("facility_location_relevance", "Facility loc. + relevance"),
        ("kcenter", "$k$-center"),
    ]
    datasets = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
    rows = []
    means = {key: [] for key, _ in policies}
    for dataset in datasets:
        per_target: dict[int, dict[str, list[float]]] = {}
        for target_dir in sorted((root / dataset).glob("target*_seed*")):
            path = target_dir / "result.json"
            if not path.exists():
                continue
            record = _json.loads(path.read_text())
            bucket = per_target.setdefault(int(record["target_sensor"]), {})
            for key, _label in policies:
                bucket.setdefault(key, []).append(float(record["metrics"][key]["mean_prediction_gain"]))
        cells = []
        for key, _label in policies:
            target_means = [np.mean(bucket[key]) for bucket in per_target.values() if key in bucket]
            value = float(np.mean(target_means)) if target_means else float("nan")
            means[key].append(value)
            cells.append(fmt(value))
        rows.append(f"{dataset} & " + " & ".join(cells) + " " + "\\\\")
    rows.append("\\midrule")
    rows.append("Mean & " + " & ".join(fmt(float(np.mean(means[key]))) for key, _ in policies) + " " + "\\\\")
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Content-based subset-selection baselines under the multi-target protocol, averaged over 32 fixed targets and three runs. None of these rules trains a model or evaluates the predictor during selection.}}
\label{{tab:supp-subset-baselines}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
Dataset & Facility location & DPP greedy & Facility loc. + rel. & $k$-center\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""



def r080b_table() -> str:
    """Per-seed multi-target results for the learned policies."""

    import json as _json

    runs = {
        101: ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340",
        202: ROOT / "results/raw/R080b_multitarget_rmur_s202_diag_20260921",
        303: ROOT / "results/raw/R080b_multitarget_rmur_s303_diag_20260921",
    }
    policies = [
        ("static_utility", "Standalone"),
        ("cached_mur", "Cached-only"),
        ("ranked_response_q4", "R-MUR"),
    ]
    datasets = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
    rows = []
    for dataset in datasets:
        for seed, root in sorted(runs.items()):
            values: dict[str, list[float]] = {key: [] for key, _ in policies}
            found = False
            for target_dir in sorted((root / dataset).glob(f"target*_seed{seed}")):
                path = target_dir / "result.json"
                if not path.exists():
                    continue
                found = True
                record = _json.loads(path.read_text())
                for key, _label in policies:
                    values[key].append(float(record["metrics"][key]["mean_prediction_gain"]))
            if not found:
                continue
            cells = [fmt(float(np.mean(values[key]))) for key, _ in policies]
            rows.append(f"{dataset} & {seed} & " + " & ".join(cells) + " \\\\")
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Per-seed multi-target prediction gains for the learned policies, averaged over the 32 fixed targets of each dataset.}}
\label{{tab:supp-multitarget}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrr}}
\toprule
Dataset & Run & Standalone & Cached-only & R-MUR\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def main() -> None:
    raw_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/raw/R071_candidate_size_r050_b4_s5_20260920_1730"
    tables = [
        r070_table(),
        r071_table(raw_root),
        r076_table(),
        r080a_table(),
        r080b_table(),
        subset_baseline_table(),
    ]
    OUT.write_text("\n".join(tables), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
