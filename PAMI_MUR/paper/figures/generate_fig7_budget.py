"""Figure 7: the budget/identifiability gap under a strong predictor.

Left: mean gain of random k-subsets and of the greedy oracle at budget k on the
strong backbone, with the pool-all level marked. Right: the same gap expressed
as the share of the oracle gain that each deployable family realises.
Drawn from results/derived/R121_budget_and_direction/budget_and_direction.json
and the R103 gate run; nothing is hard-coded.
"""

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
)

ROOT = Path(__file__).resolve().parents[2]
CURVE = ROOT / "results/derived/R121_budget_and_direction/budget_and_direction.json"
GATE = ROOT / "results/raw/R103_strong_backbone_gate_20260921_0350/METRLA/target_statistics.json"
OUT = Path(__file__).resolve().parent / "fig7_budget_gap"


def main() -> None:
    apply_publication_style(FigureStyle(font_size=8.5, axes_linewidth=1.4))
    report = json.loads(CURVE.read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))["policy_means"]
    ks = sorted(int(k) for k in report["random_curve"])
    random_curve = [report["random_curve"][str(k)] for k in ks]
    oracle_curve = [report["oracle_curve"][str(k)] for k in ks]

    fig, axes = create_subplots(1, 2, figsize=(7.4, 2.8))
    ax = axes[0]
    ax.plot(ks, oracle_curve, marker="o", markersize=3, color=PALETTE["red_strong"], label="oracle at budget k")
    ax.plot(ks, random_curve, marker="s", markersize=3, color=PALETTE["blue_main"], label="random k-subset")
    ax.axhline(gate["ranked_response_q4"], color=PALETTE["teal"], linestyle="--", linewidth=1.2,
               label="best learned selection (R-MUR q4)")
    ax.axvline(4, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.text(4.15, max(oracle_curve) * 0.35, "protocol budget", fontsize=7, color=PALETTE["neutral"])
    ax.set_xlabel("context budget k")
    ax.set_ylabel("gain over anchor-only")
    ax.set_title("Perfect selection wants fewer contexts", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    families = {
        "Random 4": gate["random"],
        "Relevance": gate["relevance"],
        "MMR": gate["mmr"],
        "DPP": gate["dpp"],
        "Static util.": gate["static_utility"],
        "Cached MUR": gate["cached_mur"],
        "R-MUR q4": gate["ranked_response_q4"],
        "R-MUR q8": gate["ranked_response_q8"],
        "Pool all": gate["pool_all"],
    }
    oracle = gate["oracle_static"]
    names = list(families)
    shares = [families[name] / oracle for name in names]
    colours = [PALETTE["neutral"]] * (len(names) - 1) + [PALETTE["teal"]]
    positions = np.arange(len(names))
    ax.bar(positions, shares, color=colours, width=0.66)
    ax.axhline(1.0, color=PALETTE["red_strong"], linestyle="--", linewidth=1.0)
    ax.text(len(names) - 0.4, 1.02, "standalone oracle", fontsize=7, ha="right", color=PALETTE["red_strong"])
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("share of standalone oracle realised")
    ax.set_ylim(0, 1.15)
    ax.set_title("Every deployable policy is far below the oracle", fontsize=9)

    finalize_figure(fig, OUT)


if __name__ == "__main__":
    main()
