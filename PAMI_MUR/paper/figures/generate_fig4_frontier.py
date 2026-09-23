from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from figures4papers_style import (
    FigureStyle,
    PALETTE,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_scatter,
    make_trend,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/R072_mur_conservative_h025_h050_a005_a050_s5_20260920_0000/summary.csv"
OUT = Path(__file__).resolve().parent / "fig4_frontier"


def main() -> None:
    apply_publication_style(FigureStyle(font_size=9.0, axes_linewidth=1.6))
    rows = list(csv.DictReader(RAW.open()))
    fig, axes = create_subplots(1, 2, figsize=(8.6, 2.8), sharey=True)

    for ax, harm in zip(axes, ("0.25", "0.5")):
        subset = [r for r in rows if r["harmful_fraction"] == harm]

        anchor = [r for r in subset if r["policy"] == "mur_light"]
        if anchor:
            row = anchor[0]
            make_scatter(
                ax,
                [float(row["negative_transfer_rate_mean"])],
                [float(row["mean_prediction_gain_mean"])],
                label="MUR anchor",
                color=PALETTE["blue_main"],
                size=52,
                marker="s",
            )

        curve_specs = [
            ("mur_interval", "MUR-Conservative", PALETTE["green_3"], "o"),
            ("mur_free_threshold", "Free threshold", PALETTE["teal"], "D"),
        ]
        for policy, label, color, marker in curve_specs:
            data = sorted(
                [r for r in subset if r["policy"] == policy],
                key=lambda r: float(r["alpha"]),
            )
            x = np.asarray([float(r["negative_transfer_rate_mean"]) for r in data])
            y = np.asarray([float(r["mean_prediction_gain_mean"]) for r in data])
            yerr = np.asarray([float(r["mean_prediction_gain_std"]) for r in data])
            make_trend(
                ax,
                x,
                [y],
                labels=[label],
                colors=[color],
                errors=[yerr],
                linewidth=1.8,
                marker=marker,
                markersize=4.0,
            )
            if data:
                ax.annotate(
                    r"$\alpha=.05$",
                    (x[0], y[0]),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=6.5,
                    color=color,
                )
                ax.annotate(
                    r"$\alpha=.50$",
                    (x[-1], y[-1]),
                    textcoords="offset points",
                    xytext=(-4, 6),
                    fontsize=6.5,
                    color=color,
                    ha="right",
                )
        ax.set_xlabel(f"Negative-transfer rate ($r_{{\\mathrm{{harm}}}}={harm}$)")

    axes[0].set_ylabel("Prediction gain")
    axes[0].legend(frameon=False, loc="upper left")
    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.08)


if __name__ == "__main__":
    main()
