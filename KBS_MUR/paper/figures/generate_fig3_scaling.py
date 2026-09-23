from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from figures4papers_style import (
    FigureStyle,
    PALETTE,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_trend,
)


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/derived/R071_margin_audit/summary.json"
PER_SEED = ROOT / "results/derived/R071_margin_audit/per_seed.csv"
OUT = Path(__file__).resolve().parent / "fig3_scaling"


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ks = np.asarray(sorted(int(k) for k in summary), dtype=float)
    fig, axes = create_subplots(2, 2, figsize=(5.4, 4.3))

    def vect(metric: str):
        means = np.asarray([summary[str(int(k))][metric]["mean"] for k in ks])
        stds = np.asarray([summary[str(int(k))][metric]["std"] for k in ks])
        return means, stds

    # A. Error normalized by the scale of the true utility.
    nmae_means, stds = vect("nmae_sd")
    make_trend(
        axes[0], ks, [nmae_means], ylabel="Normalized utility error\nMAE / SD$(m)$",
        xlabel="Candidate count $K$", colors=[PALETTE["blue_main"]],
        errors=[stds], linewidth=2.2, markersize=4.2,
    )

    # B. Maximum utility error relative to the positive top-two margin.
    means, stds = vect("median_ratio_max_error_margin_positive")
    make_trend(
        axes[1], ks, [means], ylabel="Max error / decision margin\n(median)",
        xlabel="Candidate count $K$", colors=[PALETTE["red_strong"]],
        errors=[stds], linewidth=2.2, markersize=4.2,
    )
    axes[1].axhline(1.0, color="#4D4D4D", linewidth=0.9, linestyle="--")
    axes[1].annotate("noise exceeds margin\nat every $K$", xy=(16, float(vect("median_ratio_max_error_margin_positive")[0][2])),
                     xytext=(5, 1.15), fontsize=6, color="#4D4D4D")

    # C. Strict top-1 and tie-aware decision accuracy.
    strict_mean, strict_std = vect("strict_top1_accuracy")
    decision_mean, decision_std = vect("decision_accuracy")
    make_trend(
        axes[2], ks, [decision_mean, strict_mean],
        labels=["Tie-aware accuracy", "Strict top-1"],
        ylabel="Selection accuracy", xlabel="Candidate count $K$",
        colors=[PALETTE["blue_main"], PALETTE["neutral"]],
        errors=[decision_std, strict_std], linewidth=2.0, markersize=4.0,
    )
    axes[2].legend(frameon=False, loc="upper right")

    # D. Realized decision regret.
    means, stds = vect("mean_regret")
    make_trend(
        axes[3], ks, [means], ylabel="One-step regret",
        xlabel="Candidate count $K$", colors=[PALETTE["blue_main"]],
        errors=[stds], linewidth=2.2, markersize=4.2,
    )

    for ax in axes:
        ax.set_xticks(ks)
        ax.set_xticklabels([str(int(k)) for k in ks])

    axes[0].annotate("normalized error rises", xy=(64, float(nmae_means[-1])),
                     xytext=(6, float(nmae_means[-1]) * 1.12), fontsize=6,
                     color=PALETTE["blue_main"])
    axes[3].annotate("regret rises at moderate $K$\nand remains elevated",
                     xy=(16, float(means[2])),
                     xytext=(12, float(means[2]) * 0.55), fontsize=6,
                     color=PALETTE["blue_main"])

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
