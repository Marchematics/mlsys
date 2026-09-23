"""Figure: risk--coverage frontier with the operating points of each gate.

Source: ``results/derived/R200_risk_control/summary.json`` (protocols P5 and
P7), produced by ``scripts/analyze_risk_control.py`` from the saved deployment
traces of the multi-target runs.
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
    make_trend,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/derived/R200_risk_control/summary.json"
OUT = Path(__file__).resolve().parent / "fig_risk_coverage"
LABELS = {
    "METRLA": "METR-LA",
    "PEMSBAY": "PEMS-BAY",
    "PEMS03": "PEMS03",
    "PEMS04": "PEMS04",
    "PEMS07": "PEMS07",
    "PEMS08": "PEMS08",
}


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.5, axes_linewidth=1.2))
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    p7 = summary["protocols"]["P7_share_vs_mass"]
    p5 = summary["protocols"]["P5_conformal_risk_control"]

    fig, axes = create_subplots(1, 2, figsize=(5.4, 2.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax_a, ax_b = axes

    # A. Risk--coverage frontier per dataset, with the mean highlighted.
    mean_curve = None
    for index, (dataset, entry) in enumerate(p7.items()):
        dense = entry["risk_coverage_dense"]
        coverage = np.asarray(dense["coverage"], dtype=float)
        share = np.asarray(dense["harmful_share"], dtype=float)
        ax_a.plot(coverage, share, color="0.75", linewidth=0.9, zorder=1)
        mean_curve = share if mean_curve is None else np.minimum(mean_curve, share)
        if index == 0:
            ax_a.plot([], [], color="0.75", linewidth=0.9, label="Individual datasets")
    mean_curve = np.asarray(mean_curve, dtype=float)
    coverage = np.asarray(p7[next(iter(p7))]["risk_coverage_dense"]["coverage"], dtype=float)
    make_trend(
        ax_a,
        coverage,
        [mean_curve],
        labels=["Best achievable over datasets"],
        colors=[PALETTE["blue_main"]],
        ylabel="Harmful share among accepted decisions",
        xlabel="Coverage",
        linewidth=1.8,
        show_shadow=False,
    )

    # Operating points at alpha = .02, averaged over datasets.
    markers = [
        ("Always", [1.0], [np.mean([e["always_select"]["selective_risk"] for e in p5.values()])], "#4D4D4D", "s"),
        (
            "Sign gate",
            [np.mean([e["uncalibrated"]["selected_fraction"] for e in p5.values()])],
            [np.mean([e["uncalibrated"]["selective_risk"] for e in p5.values()])],
            PALETTE["red_strong"],
            "^",
        ),
    ]
    for label, x, y, color, marker in markers:
        ax_a.scatter(x, y, s=26, color=color, marker=marker, zorder=4, label=label)
    interval_x = np.mean([e["alpha_0.020"]["interval_lcb"]["selected_fraction"] for e in p5.values()])
    interval_y = np.mean([e["alpha_0.020"]["interval_lcb"]["selective_risk"] for e in p5.values()])
    budget_x = np.mean([e["alpha_0.020"]["global"]["selected_fraction"] for e in p5.values()])
    budget_y = np.mean([e["alpha_0.020"]["global"]["selective_risk"] for e in p5.values()])
    ax_a.scatter([interval_x], [interval_y], s=30, color=PALETTE["violet"], marker="v", zorder=4, label="Interval gate")
    ax_a.scatter([budget_x], [budget_y], s=34, color=PALETTE["green_3"], marker="o", zorder=4, label="Budget gate")
    for x, y, text in (
        (interval_x, interval_y, "interval"),
        (budget_x, budget_y, "budget"),
    ):
        ax_a.annotate(text, (x, y), textcoords="offset points", xytext=(4, -8), fontsize=6, color="#4D4D4D")
    ax_a.set_xlim(0, 1.02)
    ax_a.set_ylim(0, 0.6)
    ax_a.legend(frameon=False, loc="lower right", fontsize=6)

    # B. Coverage and realised harm against the nominal budget.
    alphas = np.asarray([0.005, 0.01, 0.02, 0.05])
    coverage_mean = np.asarray(
        [
            np.mean([entry[f"alpha_{a:.3f}"]["global"]["selected_fraction"] for entry in p5.values()])
            for a in alphas
        ]
    )
    coverage_lo = np.asarray(
        [
            np.mean([entry[f"alpha_{a:.3f}"]["global"]["ci"]["selected_fraction"][0] for entry in p5.values()])
            for a in alphas
        ]
    )
    coverage_hi = np.asarray(
        [
            np.mean([entry[f"alpha_{a:.3f}"]["global"]["ci"]["selected_fraction"][1] for entry in p5.values()])
            for a in alphas
        ]
    )
    harm_mean = np.asarray(
        [np.mean([entry[f"alpha_{a:.3f}"]["global"]["harm_mass"] for entry in p5.values()]) for a in alphas]
    )
    make_trend(
        ax_b,
        alphas,
        [coverage_mean, harm_mean],
        labels=["Coverage", "Realised harmful mass"],
        colors=[PALETTE["blue_main"], PALETTE["red_strong"]],
        ylabel="Fraction of decisions",
        xlabel="Nominal budget $\\alpha$",
        errors=[np.vstack([coverage_mean - coverage_lo, coverage_hi - coverage_mean]), None],
        linewidth=1.8,
        show_shadow=False,
    )
    ax_b.plot(alphas, alphas, color="#4D4D4D", linewidth=0.9, linestyle="--", label="Nominal $\\alpha$")
    ax_b.set_xscale("log")
    ax_b.set_xticks(alphas)
    ax_b.set_xticklabels([f"{a:g}" for a in alphas])
    ax_b.legend(frameon=False, loc="upper left", fontsize=6)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
