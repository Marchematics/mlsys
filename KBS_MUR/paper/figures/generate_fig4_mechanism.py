"""Figure: mechanisms behind the routing gain.

Panel A: candidate-pool growth compresses the decision margin
(``results/derived/R071_margin_audit/summary.json``). Panels B and C: cached
features, compact response summaries, and full forecast trajectories
(``results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv``).
"""

from __future__ import annotations

import csv
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
    make_trend,
)

ROOT = Path(__file__).resolve().parents[2]
MARGIN = ROOT / "results/derived/R071_margin_audit/summary.json"
PROBE = ROOT / "results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv"
OUT = Path(__file__).resolve().parent / "fig4_mechanism"

PROBES = [
    ("x1", "Cached features", PALETTE["neutral"]),
    ("x2", "+ Response summaries", PALETTE["blue_main"]),
    ("x3", "+ Forecast trajectories", PALETTE["red_strong"]),
]


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))
    fig, axes = create_subplots(
        1, 3, figsize=(5.4, 2.15), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]}
    )
    ax_a, ax_b, ax_c = axes

    summary = json.loads(MARGIN.read_text(encoding="utf-8"))
    ks = np.asarray(sorted(int(k) for k in summary), dtype=float)
    normalized = np.asarray([summary[str(int(k))]["nmae_sd"]["mean"] for k in ks])
    ratio = np.asarray([summary[str(int(k))]["median_ratio_max_error_margin_positive"]["mean"] for k in ks])
    make_trend(
        ax_a,
        ks,
        [normalized, ratio],
        labels=["Error / SD$(m)$", "Error / margin"],
        ylabel="Normalized error",
        xlabel="Candidate count $K$",
        colors=[PALETTE["blue_main"], PALETTE["red_strong"]],
        linewidth=1.8,
        markersize=3.4,
    )
    ax_a.axhline(1.0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels([str(int(k)) for k in ks])
    ax_a.set_ylim(0, 3.8)
    ax_a.legend(frameon=False, loc="upper left", fontsize=5.8)

    probe = {}
    for row in csv.DictReader(PROBE.open()):
        probe[row["probe"]] = {"mean": float(row["metric_mean"]), "std": float(row["metric_std"])}

    metric_keys = ["spearman", "positive_auc", "top1_accuracy"]
    metric_labels = ["Spearman", "Pos. AUC", "Top-1"]
    values = [np.asarray([probe[f"{name}:{m}"]["mean"] for m in metric_keys]) for name, _, _ in PROBES]
    errors = [np.asarray([probe[f"{name}:{m}"]["std"] for m in metric_keys]) for name, _, _ in PROBES]
    bars = make_grouped_bar(
        ax_b,
        metric_labels,
        values,
        ["Cached features", "+ Response summaries", "+ Forecast trajectories"],
        ylabel="Probe score",
        colors=[color for _, _, color in PROBES],
        errors=errors,
        capsize=2.0,
    )
    ax_b.set_ylim(0, 1.32)
    ax_b.legend(frameon=False, loc="upper left", fontsize=5.2, ncol=1, handlelength=1.1,
                borderaxespad=0.1, labelspacing=0.25)
    annotate_bars(ax_b, bars, fmt="{:.2f}", fontsize=5.2)

    regrets = np.asarray([probe[f"{name}:one_step_regret"]["mean"] for name, _, _ in PROBES])
    regret_err = np.asarray([probe[f"{name}:one_step_regret"]["std"] for name, _, _ in PROBES])
    positions = np.arange(len(PROBES))
    regret_bars = ax_c.bar(
        positions,
        regrets,
        yerr=regret_err,
        color=[color for _, _, color in PROBES],
        edgecolor="black",
        linewidth=0.9,
        capsize=2.0,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(["Cached", "+Response", "+Trajectory"], fontsize=5.6)
    ax_c.set_ylabel("One-step regret")
    ax_c.set_ylim(0, 0.095)
    annotate_bars(ax_c, regret_bars, fmt="{:.4f}", fontsize=5.2)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
