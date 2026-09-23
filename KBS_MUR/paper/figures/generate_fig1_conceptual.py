from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures4papers_style import FigureStyle, PALETTE, apply_publication_style, finalize_figure


OUT = Path(__file__).resolve().parent / "fig1_conceptual"


def rounded_box(ax, xy, width, height, text, color, *, fontsize=7.2,
                facealpha=0.12, linewidth=1.2, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=.025,rounding_size=.07",
        linewidth=linewidth,
        edgecolor=color,
        facecolor=color,
        alpha=facealpha,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1A1A1A",
        fontweight=weight,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=PALETTE["blue_main"], style="-|>", lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=color,
            zorder=4,
        )
    )


def main() -> None:
    apply_publication_style(FigureStyle(font_size=8.5, axes_linewidth=1.4))
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.0))
    left, right = axes

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    # ------------------------------------------------------------------ Left
    left.text(0.05, 5.72, "A | Value changes after context selection",
              fontsize=9, weight="bold", color="#1A1A1A")
    left.text(0.05, 5.18, "Candidate pool", fontsize=7.4, color="#4D4D4D")

    card_y, card_h, card_w = 3.85, 1.00, 2.75
    card_specs = [
        (0.10, "C1", "relevance high\nstandalone high", PALETTE["blue_main"], PALETTE["blue_secondary"]),
        (3.55, "C2", "relevance high\nstandalone high", PALETTE["neutral"], "#767676"),
        (7.00, "C3", "relevance moderate\nstandalone moderate", PALETTE["green_3"], PALETTE["green_3"]),
    ]
    for x, name, body, edge, accent in card_specs:
        rounded_box(left, (x, card_y), card_w, card_h, "", edge, facealpha=0.10, linewidth=1.4)
        left.add_patch(
            FancyBboxPatch(
                (x, card_y + card_h - 0.16),
                card_w,
                0.16,
                boxstyle="square,pad=0",
                facecolor=accent,
                edgecolor="none",
                alpha=0.55,
                zorder=3,
            )
        )
        left.text(x + 0.10, card_y + card_h - 0.30, name, ha="left", va="center",
                  fontsize=8.2, weight="bold", color="#1A1A1A", zorder=5)
        left.text(x + 0.10, card_y + 0.48, body, ha="left", va="center",
                  fontsize=7.2, color="#1A1A1A", zorder=5, linespacing=1.4)

    arrow(left, (1.45, card_y - 0.03), (1.45, 2.72), PALETTE["blue_main"])
    rounded_box(left, (0.10, 1.95), 4.10, 0.72,
                "Static ranking selects C1", PALETTE["blue_main"],
                fontsize=7.8, facealpha=0.16, linewidth=1.4, weight="bold")
    left.text(0.18, 1.60, "selected set A = {C1}", fontsize=7.2, color="#4D4D4D")

    rounded_box(left, (0.10, 0.55), 4.05, 0.78, "", PALETTE["red_strong"],
                facealpha=0.10, linewidth=1.2)
    left.text(0.28, 0.94, r"$C2$: $m(C2\mid\{C1\})\approx 0$",
              fontsize=7.4, color=PALETTE["red_strong"], va="center")
    left.text(0.28, 0.66, "redundant after C1", fontsize=6.8, color="#767676")

    rounded_box(left, (5.35, 0.55), 4.05, 0.78, "", PALETTE["green_3"],
                facealpha=0.16, linewidth=1.2)
    left.text(5.53, 0.94, r"$C3$: $m(C3\mid\{C1\})>0$",
              fontsize=7.4, color="#2F6B2F", va="center")
    left.text(5.53, 0.66, "complementary after C1", fontsize=6.8, color="#767676")

    arrow(left, (2.12, 1.93), (2.12, 1.35), PALETTE["red_strong"], lw=1.1)
    arrow(left, (7.38, 1.93), (7.38, 1.35), PALETTE["green_3"], lw=1.1)
    rounded_box(left, (0.10, 0.02), 4.05, 0.40, "Standalone utility → C2",
                PALETTE["red_strong"], fontsize=7.0, facealpha=0.16, linewidth=1.0)
    rounded_box(left, (5.35, 0.02), 4.05, 0.40, "R-MUR → C3",
                PALETTE["blue_main"], fontsize=7.0, facealpha=0.16, linewidth=1.0)

    # ----------------------------------------------------------------- Right
    right.text(0.05, 5.72, "B | Structural headroom and observability",
               fontsize=9, weight="bold", color="#1A1A1A")
    right.text(0.05, 5.20, "Structural opportunity", fontsize=7.6, weight="bold",
               color="#4D4D4D")

    rounded_box(right, (0.10, 4.10), 3.85, 0.78,
                "Perfect standalone ranking\nOracle-Static", PALETTE["green_3"],
                fontsize=7.4, facealpha=0.16, linewidth=1.3)
    rounded_box(right, (5.65, 4.10), 3.85, 0.78,
                "Sequential oracle\nOracle-Greedy", "#4D4D4D",
                fontsize=7.4, facealpha=0.12, linewidth=1.3)
    arrow(right, (4.02, 4.49), (5.60, 4.49), "#4D4D4D", lw=1.3)
    right.text(4.81, 4.58, r"$H_{\mathrm{state}}$", ha="center",
               fontsize=8.4, color=PALETTE["blue_main"], weight="bold")
    right.text(0.10, 3.76, "gap = structural headroom created by state conditioning",
               fontsize=6.9, color="#4D4D4D")

    right.plot([0.10, 9.90], [3.48, 3.48], color="#D9D9D9", linewidth=1.0)
    right.text(0.05, 3.12, "Predictability from deployment observables",
               fontsize=7.6, weight="bold", color="#4D4D4D")

    rounded_box(right, (0.10, 2.00), 3.85, 0.78,
                "Cached context\nstate", PALETTE["neutral"],
                fontsize=7.4, facealpha=0.20, linewidth=1.3)
    right.text(0.20, 1.74, "limited utility observability", fontsize=6.9,
               color="#4D4D4D")
    rounded_box(right, (5.65, 2.00), 3.85, 0.78,
                "Utility model with\npredictor response", PALETTE["blue_main"],
                fontsize=7.4, facealpha=0.14, linewidth=1.3)
    right.text(5.75, 1.74, "stronger utility observability", fontsize=6.9,
               color=PALETTE["blue_main"])
    arrow(right, (4.02, 2.39), (5.60, 2.39), PALETTE["blue_main"], lw=1.3)
    right.text(4.81, 2.48, "+ predictor response", ha="center",
               fontsize=6.9, color=PALETTE["blue_main"])

    rounded_box(right, (0.10, 0.25), 9.40, 0.62,
                "Structural headroom measures the opportunity.\nObservability determines how much of it a deployment interface reveals.",
                PALETTE["highlight"], fontsize=7.0, facealpha=0.16, linewidth=1.0)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
