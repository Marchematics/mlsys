"""Figure 6: generality evidence -- strong backbone and second task family.

Left: METR-LA policy comparison with the subset-capable nonlinear backbone
(16 targets x 3 seeds). Right: second task family, R-MUR against the two
response-free learned routers per benchmark.

Both panels are drawn directly from the released derived artifacts; nothing is
hard-coded.
"""

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
)

ROOT = Path(__file__).resolve().parents[2]
STRONG = ROOT / "results/raw/R103_strong_backbone_gate_20260921_0350/METRLA/target_statistics.json"
SECOND = ROOT / "results/derived/R110_demo_selection_summary/gate_table.csv"
OUT = Path(__file__).resolve().parent / "fig6_generality"


def main() -> None:
    apply_publication_style(FigureStyle(font_size=8.5, axes_linewidth=1.4))
    fig, axes = create_subplots(1, 2, figsize=(7.4, 2.9))

    strong = json.loads(STRONG.read_text(encoding="utf-8"))
    means = strong["policy_means"]
    order = [
        ("random", "Random"),
        ("relevance", "Relevance"),
        ("mmr", "MMR"),
        ("static_utility", "Static"),
        ("cached_mur", "Cached"),
        ("ranked_response_q4", "R-MUR q4"),
        ("ranked_response_q8", "R-MUR q8"),
        ("pool_all", "Pool all"),
    ]
    labels = [label for _, label in order]
    values = [means[key] for key, _ in order]
    colours = [PALETTE["neutral"]] * 4 + [PALETTE["teal"], PALETTE["blue_main"], PALETTE["blue_main"], PALETTE["neutral"]]
    ax = axes[0]
    positions = np.arange(len(values))
    ax.bar(positions, values, color=colours, width=0.68)
    ax.axhline(means["oracle_static"], color=PALETTE["red_strong"], linestyle="--", linewidth=1.2)
    ax.axhline(means["oracle_greedy"], color=PALETTE["red_strong"], linestyle=":", linewidth=1.2)
    ax.text(len(values) - 0.4, means["oracle_static"], " standalone oracle", color=PALETTE["red_strong"], fontsize=7, va="bottom", ha="right")
    ax.text(len(values) - 0.4, means["oracle_greedy"], " sequential oracle", color=PALETTE["red_strong"], fontsize=7, va="bottom", ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Gain over anchor-only")
    ax.set_title("Nonlinear backbone (METR-LA, 16 targets)", fontsize=9)

    rows = list(csv.DictReader(SECOND.open()))
    q4 = {row["benchmark"]: row for row in rows if row["policy"] == "ranked_response_q4"}
    benchmarks = ["cifar10", "cifar100", "svhn", "eurosat", "dtd"]
    pretty = ["CIFAR-10", "CIFAR-100", "SVHN", "EuroSAT", "DTD"]
    ax = axes[1]
    width = 0.26
    positions = np.arange(len(benchmarks))
    series = {
        "Static": [float(q4[b]["delta_static"]) for b in benchmarks],
        "Cached": [float(q4[b]["delta_cached"]) for b in benchmarks],
    }
    low = {
        "Static": [float(q4[b]["ds_lo"]) for b in benchmarks],
        "Cached": [float(q4[b]["dc_lo"]) for b in benchmarks],
    }
    high = {
        "Static": [float(q4[b]["ds_hi"]) for b in benchmarks],
        "Cached": [float(q4[b]["dc_hi"]) for b in benchmarks],
    }
    for offset, (name, values) in enumerate(series.items()):
        errors = np.asarray([
            np.clip(np.asarray(values, dtype=float) - np.asarray(low[name], dtype=float), 0, None),
            np.clip(np.asarray(high[name], dtype=float) - np.asarray(values, dtype=float), 0, None),
        ])
        ax.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            yerr=errors,
            capsize=1.5,
            color=PALETTE["blue_main"] if name == "Static" else PALETTE["teal"],
            label=f"R-MUR - {name}",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(pretty, rotation=25, ha="right")
    ax.set_ylabel("Paired gain difference (nats)")
    ax.set_title("Second family: demonstration selection", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    finalize_figure(fig, OUT)


if __name__ == "__main__":
    main()
