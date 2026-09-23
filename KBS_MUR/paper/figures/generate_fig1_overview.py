"""Figure: R-MUR overview.

Three blocks, left to right: the prediction context, the response-aware
marginal router, and the outcome-supervised training signal. No reinforcement
learning terminology is used: the router is a supervised decision policy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures4papers_style import FigureStyle, PALETTE, apply_publication_style, finalize_figure

OUT = Path(__file__).resolve().parent / "fig1_overview"


def rounded_box(ax, xy, width, height, text, color, *, fontsize=6.4, facealpha=0.12,
                linewidth=1.0, weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=color,
            edgecolor=color,
            alpha=facealpha,
            linewidth=linewidth,
            zorder=2,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1A1A1A", weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, start, end, color=PALETTE["blue_main"], lw=1.1, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                        linewidth=lw, color=color, zorder=4)
    )


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))
    fig, ax = plt.subplots(figsize=(5.4, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Column headings.
    for x, title in ((0.15, "Prediction context"),
                     (3.55, "Response-aware marginal router"),
                     (7.05, "Outcome-supervised training")):
        ax.text(x, 4.72, title, fontsize=6.8, weight="bold", color="#1A1A1A")

    # Column 1: prediction context.
    rounded_box(ax, (0.15, 3.35), 2.9, 0.95,
                "Anchor history $S_0$\nand query $x$", PALETTE["blue_main"], facealpha=0.10)
    rounded_box(ax, (0.15, 1.95), 2.9, 0.95,
                "Candidate pool\n$\\mathcal{C}=\\{C_1,\\dots,C_K\\}$", PALETTE["blue_secondary"], facealpha=0.10)
    rounded_box(ax, (0.15, 0.55), 2.9, 0.95,
                "Fixed predictor $f$\n(trained once, then frozen)", "#4D4D4D", facealpha=0.08)

    # Column 2: router.
    rounded_box(ax, (3.55, 3.35), 3.1, 0.95,
                "Cached screening\n$\\widehat m_C(j\\mid A)$", PALETTE["teal"], facealpha=0.14)
    rounded_box(ax, (3.55, 1.95), 3.1, 0.95,
                "Predictor response on shortlist\n$f_A$, $f_{A\\cup\\{j\\}}\\ \\rightarrow\\ r_{A,j}$",
                PALETTE["green_3"], facealpha=0.18)
    rounded_box(ax, (3.55, 0.55), 3.1, 0.95,
                "Response-aware marginal scorer\n$\\widehat m_R(j\\mid A)$", PALETTE["blue_main"], facealpha=0.14)

    # Column 3: training signal.
    rounded_box(ax, (7.05, 3.35), 2.8, 0.95,
                "Observed marginal\n$\\ell(f_A(x),y)-\\ell(f_{A\\cup\\{j\\}}(x),y)$", PALETTE["violet"], facealpha=0.12)
    rounded_box(ax, (7.05, 1.95), 2.8, 0.95,
                "Regression + gap-weighted\nranking objective", PALETTE["red_strong"], facealpha=0.12)
    rounded_box(ax, (7.05, 0.55), 2.8, 0.95,
                "Sequential selection\nselect, then stop when $\\widehat m_R \\leq 0$",
                "#4D4D4D", facealpha=0.08)

    # Arrows between columns.
    for y in (3.83, 2.43):
        arrow(ax, (3.10, y), (3.50, y))
    arrow(ax, (6.70, 3.83), (7.00, 3.83), PALETTE["violet"])
    arrow(ax, (6.70, 2.43), (7.00, 2.43), PALETTE["violet"])
    # Internal flow.
    arrow(ax, (5.10, 3.30), (5.10, 2.95))
    arrow(ax, (5.10, 1.90), (5.10, 1.55))
    arrow(ax, (8.45, 3.30), (8.45, 2.95), PALETTE["red_strong"])
    arrow(ax, (5.10, 0.50), (5.10, 0.20), PALETTE["blue_main"], style="-")
    arrow(ax, (5.10, 0.20), (9.60, 0.20), PALETTE["blue_main"], style="-")
    arrow(ax, (9.60, 0.20), (9.60, 1.90), PALETTE["blue_main"])
    ax.text(7.05, 0.06, "deploy: update $A \\leftarrow A \\cup \\{j^\\star\\}$ and repeat",
            fontsize=5.8, color=PALETTE["blue_main"])

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.08)


if __name__ == "__main__":
    main()
