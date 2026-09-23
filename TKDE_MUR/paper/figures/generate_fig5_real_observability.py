from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

from figures4papers_style import (
    FigureStyle,
    PALETTE,
    annotate_bars,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_grouped_bar,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig5_real_observability"


def read_policy_csv(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(path.open()):
        out[row["policy"]] = {
            k: (float(v) if v not in ("", "nan", "NaN") else float("nan"))
            for k, v in row.items()
            if k != "policy"
        }
    return out


def main() -> None:
    apply_publication_style(FigureStyle(font_size=9.0, axes_linewidth=1.6))

    electricity = [
        json.loads(Path(path).read_text())
        for path in glob.glob(str(ROOT / "results/raw/R073_electricity_oracle_audit_s*_*/result.json"))
    ]
    traffic_oracle = read_policy_csv(
        ROOT / "results/raw/R074_traffic_oracle_gate_metrla_k16_b4_s5_20260920_0100/summary.csv"
    )
    traffic_router = read_policy_csv(
        ROOT / "results/raw/R075_traffic_full_router_metrla_k16_b4_s5_20260920_0115/summary.csv"
    )

    probe: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(
        (ROOT / "results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv").open()
    ):
        probe[row["probe"]] = {"mean": float(row["metric_mean"]), "std": float(row["metric_std"])}

    fig, axes = create_subplots(2, 2, figsize=(11.0, 5.8))

    # A. Oracle headroom in both real tasks.
    tasks = ["Electricity", "METR-LA"]
    elec_static = np.asarray([r["oracle_quantities"]["gain_base_to_oracle_static"] for r in electricity])
    elec_greedy = np.asarray([r["oracle_quantities"]["gain_base_to_oracle_greedy"] for r in electricity])
    elec_head = np.asarray([r["oracle_quantities"]["state_conditioning_headroom"] for r in electricity])
    static = np.asarray([elec_static.mean(), traffic_oracle["oracle_static"]["mean_prediction_gain_mean"]])
    static_err = np.asarray([elec_static.std(ddof=1), traffic_oracle["oracle_static"]["mean_prediction_gain_std"]])
    greedy = np.asarray([elec_greedy.mean(), traffic_oracle["oracle_greedy"]["mean_prediction_gain_mean"]])
    greedy_err = np.asarray([elec_greedy.std(ddof=1), traffic_oracle["oracle_greedy"]["mean_prediction_gain_std"]])
    headroom = np.asarray([elec_head.mean(), 0.1095])

    make_grouped_bar(
        axes[0],
        tasks,
        [static, greedy],
        ["Oracle-Static", "Oracle-Greedy"],
        ylabel="Prediction gain",
        colors=[PALETTE["green_3"], "#4D4D4D"],
        errors=[static_err, greedy_err],
        annotate=False,
    )
    for xi, h in enumerate(headroom):
        axes[0].text(xi, max(static[xi], greedy[xi]) + 0.018, f"$H_{{\\mathrm{{state}}}}={h:.3f}$",
                     ha="center", fontsize=7.5)
    axes[0].set_ylim(0, 0.215)
    axes[0].legend(frameon=False, loc="upper left")

    # B. Traffic learned routing leaves the oracle gap largely unrealized.
    traffic_order = ["oracle_static", "oracle_greedy", "static_utility", "mur"]
    traffic_labels = ["Oracle-\nStatic", "Oracle-\nGreedy", "Static\nUtility", "MUR"]
    means = np.asarray([traffic_router[p]["mean_prediction_gain_mean"] for p in traffic_order])
    stds = np.asarray([traffic_router[p]["mean_prediction_gain_std"] for p in traffic_order])
    colors = [PALETTE["green_3"], "#4D4D4D", PALETTE["red_strong"], PALETTE["blue_main"]]
    xb = np.arange(len(traffic_order))
    bars = axes[1].bar(xb, means, yerr=stds, color=colors, edgecolor="black", linewidth=1.0,
                       capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0})
    axes[1].axhline(0, color="#4D4D4D", linewidth=0.8)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(traffic_labels)
    axes[1].set_ylabel("Prediction gain")
    axes[1].set_ylim(-0.025, 0.205)
    annotate_bars(axes[1], bars, fmt="{:.4f}", padding=3)
    axes[1].annotate(
        "cached policies leave\nthis headroom unrealized",
        xy=(3.0, -0.0026),
        xytext=(1.8, 0.095),
        fontsize=7,
        color=PALETTE["blue_main"],
        arrowprops=dict(arrowstyle="-", color=PALETTE["blue_main"], lw=0.9),
    )

    # C. Probe scores: response information improves ranking.
    metric_keys = ["spearman", "positive_auc", "top1_accuracy"]
    metric_labels = ["Spearman", "Positive AUC", "Top-1"]
    probes = [("x1", "X1 cached", PALETTE["neutral"]),
              ("x2", "X2 response", PALETTE["blue_main"]),
              ("x3", "X3 trajectory", PALETTE["red_strong"])]
    values = [
        np.asarray([probe[f"{name}:{metric}"]["mean"] for metric in metric_keys])
        for name, _, _ in probes
    ]
    errors = [
        np.asarray([probe[f"{name}:{metric}"]["std"] for metric in metric_keys])
        for name, _, _ in probes
    ]
    make_grouped_bar(
        axes[2],
        metric_labels,
        values,
        [label for _, label, _ in probes],
        ylabel="Probe score",
        colors=[color for _, _, color in probes],
        errors=errors,
    )
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(frameon=False, loc="upper left")

    # D. One-step regret decreases when response summaries are available.
    regret_names = ["x1", "x2", "x3"]
    regret_labels = ["X1 cached", "X2 +response", "X3 +trajectory"]
    reg_means = np.asarray([probe[f"{name}:one_step_regret"]["mean"] for name in regret_names])
    reg_stds = np.asarray([probe[f"{name}:one_step_regret"]["std"] for name in regret_names])
    xr = np.arange(len(regret_names))
    regret_bars = axes[3].bar(
        xr,
        reg_means,
        yerr=reg_stds,
        color=[PALETTE["neutral"], PALETTE["blue_main"], PALETTE["red_strong"]],
        edgecolor="black",
        linewidth=1.0,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    axes[3].set_xticks(xr)
    axes[3].set_xticklabels(regret_labels)
    axes[3].set_ylabel("One-step regret\n(lower is better)")
    axes[3].set_ylim(0, 0.09)
    annotate_bars(axes[3], regret_bars, fmt="{:.4f}", padding=3)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
