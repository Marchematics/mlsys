from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from figures4papers_style import FigureStyle, PALETTE, apply_publication_style, finalize_figure


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340"
OUT = Path(__file__).resolve().parent / "fig3_multitarget_delta"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LABELS = ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def main() -> None:
    apply_publication_style(FigureStyle(font_size=9.0, axes_linewidth=1.6))
    import matplotlib.pyplot as plt

    values = []
    fractions = []
    for dataset in DATASETS:
        stats = json.loads((RAW / dataset / "target_statistics.json").read_text())
        deltas = np.asarray(
            [t["ranked_rmur_q4_gain"] - t["static_utility_gain"] for t in stats["target_records"]],
            dtype=float,
        )
        values.append(deltas)
        fractions.append(float(np.mean(deltas > 0.0)))

    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    positions = np.arange(len(values))
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for box in bp["boxes"]:
        box.set(facecolor=PALETTE["blue_secondary"], alpha=.18, edgecolor=PALETTE["blue_main"], linewidth=1.0)
    rng = np.random.default_rng(0)
    for idx, vals in enumerate(values):
        jitter = rng.uniform(-.14, .14, size=vals.size)
        ax.scatter(
            idx + jitter,
            vals,
            s=8,
            color=PALETTE["blue_main"],
            alpha=.55,
            linewidth=0,
            zorder=3,
        )
        ax.text(idx, ax.get_ylim()[1], f"{fractions[idx]:.2f}", ha="center", va="bottom", fontsize=7, color="#4D4D4D")
    ax.axhline(0.0, color=PALETTE["red_strong"], linewidth=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylabel("R-MUR gain over\nStandalone Utility")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", alpha=.2, linewidth=.5)
    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
