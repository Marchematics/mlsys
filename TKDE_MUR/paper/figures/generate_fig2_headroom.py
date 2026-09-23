from __future__ import annotations

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
RAW = ROOT / "results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540"
OUT = Path(__file__).resolve().parent / "fig2_headroom"


def main() -> None:
    apply_publication_style(FigureStyle(font_size=9.0, axes_linewidth=1.6))
    records = json.loads((RAW / "aggregate.json").read_text())["records"]
    ratios = sorted({float(r["redundancy"]) for r in records})

    def metric(rows, policy, name):
        return np.asarray([r["metrics"][policy][name] for r in rows], dtype=float)

    headroom = {}
    grouped = {}
    for ratio in ratios:
        rows = [r for r in records if float(r["redundancy"]) == ratio]
        og = metric(rows, "oracle", "mean_prediction_gain")
        os = metric(rows, "oracle_static", "mean_prediction_gain")
        headroom[ratio] = (og - os) / og
        grouped[ratio] = {
            policy: metric(rows, policy, "mean_prediction_gain")
            for policy in ("static_utility", "oracle_static", "mur_light", "oracle")
        }
        grouped[ratio]["duplicate"] = {
            policy: metric(rows, policy, "duplicate_selection_rate")
            for policy in ("static_utility", "mur_light")
        }

    x = np.asarray(ratios, dtype=float)
    fig, axes = create_subplots(1, 3, figsize=(10.5, 2.7))
    ax_a, ax_b, ax_c = axes

    h_mean = np.asarray([headroom[r].mean() for r in ratios])
    h_std = np.asarray([headroom[r].std(ddof=1) for r in ratios])
    make_trend(
        ax_a,
        x,
        [h_mean],
        ylabel=r"$H_{\mathrm{state}}$",
        xlabel="Duplicate ratio",
        colors=[PALETTE["green_3"]],
        errors=[h_std],
        linewidth=2.2,
        markersize=4.5,
    )
    for xi, yi in zip(x, h_mean):
        ax_a.annotate(
            f"{yi:.3f}".lstrip("0"),
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
        )
    ax_a.set_ylim(-0.02, 0.34)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{v:g}" for v in x])

    policy_meta = [
        ("static_utility", "Static Utility", PALETTE["red_strong"]),
        ("oracle_static", "Oracle-Static", PALETTE["green_3"]),
        ("mur_light", "MUR", PALETTE["blue_main"]),
        ("oracle", "Oracle-Greedy", "#4D4D4D"),
    ]
    for policy, label, color in policy_meta:
        means = np.asarray([grouped[r][policy].mean() for r in ratios])
        stds = np.asarray([grouped[r][policy].std(ddof=1) for r in ratios])
        make_trend(
            ax_b,
            x,
            [means],
            labels=[label],
            colors=[color],
            errors=[stds],
            ylabel="Prediction gain",
            xlabel="Duplicate ratio",
            linewidth=1.8,
            markersize=3.8,
        )
    ax_b.set_ylim(0, 0.78)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{v:g}" for v in x])
    ax_b.text(
        0.03,
        0.97,
        "MUR > Oracle-Static\nat $r=.50$ and $r=.75$",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=PALETTE["blue_main"],
    )

    for policy, label, color in policy_meta[:1] + policy_meta[2:3]:
        means = np.asarray([grouped[r]["duplicate"][policy].mean() for r in ratios])
        stds = np.asarray([grouped[r]["duplicate"][policy].std(ddof=1) for r in ratios])
        make_trend(
            ax_c,
            x,
            [means],
            labels=[label],
            colors=[color],
            errors=[stds],
            ylabel="Duplicate-selection rate",
            xlabel="Duplicate ratio",
            linewidth=1.8,
            markersize=3.8,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{v:g}" for v in x])

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.08)


if __name__ == "__main__":
    main()
