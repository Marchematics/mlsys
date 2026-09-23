"""Figure: utility predictability of cached and response-aware probes.

Source: ``results/raw/R076_traffic_utility_observability_s5_20260920_0300``
(five seeds on METR-LA). Panel A reports rank correlation, positive-utility
AUC, and top-1 accuracy for three input sets; panel B reports the one-step
regret of the same selection rule.
"""

from __future__ import annotations

import csv
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
RAW = ROOT / "results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv"
OUT = Path(__file__).resolve().parent / "fig5_real_observability"

PROBES = [
    ("x1", "Cached features", PALETTE["neutral"]),
    ("x2", "+ Response summaries", PALETTE["blue_main"]),
    ("x3", "+ Forecast trajectories", PALETTE["red_strong"]),
]


def load_probe() -> dict[str, dict[str, float]]:
    probe: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(RAW.open()):
        probe[row["probe"]] = {
            "mean": float(row["metric_mean"]),
            "std": float(row["metric_std"]),
        }
    return probe


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))
    probe = load_probe()
    fig, axes = create_subplots(1, 2, figsize=(5.4, 2.5), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax_a, ax_b = axes

    metric_keys = ["spearman", "positive_auc", "top1_accuracy"]
    metric_labels = ["Spearman", "Positive-utility AUC", "Top-1 accuracy"]
    values = [
        np.asarray([probe[f"{name}:{metric}"]["mean"] for metric in metric_keys])
        for name, _, _ in PROBES
    ]
    errors = [
        np.asarray([probe[f"{name}:{metric}"]["std"] for metric in metric_keys])
        for name, _, _ in PROBES
    ]
    bars = make_grouped_bar(
        ax_a,
        metric_labels,
        values,
        [label for _, label, _ in PROBES],
        ylabel="Probe score",
        colors=[color for _, _, color in PROBES],
        errors=errors,
    )
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(frameon=False, loc="upper left", fontsize=6.0)
    annotate_bars(ax_a, bars, fmt="{:.2f}", fontsize=5.8)

    regret_means = np.asarray([probe[f"{name}:one_step_regret"]["mean"] for name, _, _ in PROBES])
    regret_stds = np.asarray([probe[f"{name}:one_step_regret"]["std"] for name, _, _ in PROBES])
    positions = np.arange(len(PROBES))
    regret_bars = ax_b.bar(
        positions,
        regret_means,
        yerr=regret_stds,
        color=[color for _, _, color in PROBES],
        edgecolor="black",
        linewidth=1.0,
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([label.replace("+ ", "+\n") for _, label, _ in PROBES])
    ax_b.set_ylabel("One-step regret\n(lower is better)")
    ax_b.set_ylim(0, 0.092)
    annotate_bars(ax_b, regret_bars, fmt="{:.4f}", fontsize=5.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
