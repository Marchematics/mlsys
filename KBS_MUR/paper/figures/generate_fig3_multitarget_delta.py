"""Figure: the multi-target comparison over all nine deployable policies.

Panel A ranks the nine policies by their mean gain across the six benchmarks
(per-dataset target means shown as small markers), so that the figure covers
exactly the same policy set as the baseline matrix of the manuscript. Panel B
shows the target-level distribution of the two contrasts that separate the
components of R-MUR.

Sources: the multi-target routing runs and the content-based baseline runs
(see ``multitarget_data.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from figures4papers_style import (
    FigureStyle,
    PALETTE,
    apply_publication_style,
    create_subplots,
    finalize_figure,
)
from multitarget_data import (
    ALL_POLICIES,
    DATASETS,
    LABELS,
    load_target_level,
    load_target_level_subset,
    sem,
)


OUT = Path(__file__).resolve().parent / "fig3_multitarget_delta"

COLORS = {
    "relevance": PALETTE["neutral"],
    "mmr": PALETTE["teal"],
    "static_utility": PALETTE["red_strong"],
    "cached_mur": PALETTE["green_3"],
    "ranked_response_q4": PALETTE["blue_main"],
    "facility_location": PALETTE["violet"],
    "facility_location_relevance": "#8C6BB1",
    "dpp_greedy": "#B08FC7",
    "kcenter": "#C9B3DA",
    "forward_val": PALETTE["orange_strong"] if "orange_strong" in PALETTE else "#E08214",
    "adaptive_k": "#F0A860",
    "ids_cluster": "#F5C99B",
    "lookahead_val": "#C98A3C",
    "grad_influence": "#8A6A4A",
}
LEARNED = {key for key, _ in ALL_POLICIES[:5]}


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))

    target_level = {dataset: load_target_level(dataset) for dataset in DATASETS}
    per_policy: dict[str, dict[str, np.ndarray]] = {}
    for key, _label in ALL_POLICIES:
        per_policy[key] = {
            dataset: (
                target_level[dataset][key]
                if key in target_level[dataset]
                else load_target_level_subset(dataset, key)
            )
            for dataset in DATASETS
        }

    fig, axes = create_subplots(
        1,
        2,
        figsize=(5.5, 3.5),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34},
    )
    ax_a, ax_b = axes

    # ---- Panel A: ranking of the nine deployable policies -------------------
    order = sorted(
        ALL_POLICIES,
        key=lambda item: float(np.mean([per_policy[item[0]][d].mean() for d in DATASETS])),
    )
    positions = np.arange(len(order))
    for position, (key, label) in enumerate(order):
        dataset_means = np.asarray([per_policy[key][d].mean() for d in DATASETS])
        ax_a.scatter(
            dataset_means,
            np.full(dataset_means.shape, position),
            s=6,
            color="#9A9A9A",
            alpha=0.75,
            linewidth=0,
            zorder=2,
        )
        color = COLORS.get(key, PALETTE["neutral"])
        mean = float(dataset_means.mean())
        error = sem(dataset_means)
        ax_a.errorbar(
            mean,
            position,
            xerr=error,
            fmt="o",
            markersize=5.0 if key == "ranked_response_q4" else 4.0,
            color=color,
            markeredgecolor="#1A1A1A" if key == "ranked_response_q4" else color,
            markeredgewidth=0.7,
            elinewidth=1.0,
            capsize=2.0,
            zorder=3,
        )
        ax_a.annotate(
            f"{mean:.4f}".lstrip("0"),
            (mean, position),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=5.2,
            color="#4D4D4D",
        )
    ax_a.axvline(0.0, color="#4D4D4D", linewidth=0.9, linestyle="--")
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels([label for _key, label in order], fontsize=5.3)
    ax_a.set_ylim(-0.7, len(order) - 0.3)
    ax_a.set_xlabel("Prediction gain (mean over targets; points are datasets)")
    ax_a.grid(axis="x", alpha=0.18, linewidth=0.5)

    # ---- Panel B: target-level contrasts ------------------------------------
    for offset, key, label, color in (
        (-0.19, "delta_static", "vs Standalone Utility", PALETTE["blue_main"]),
        (0.19, "delta_cached", "vs Cached-only router", PALETTE["teal"]),
    ):
        for index, dataset in enumerate(DATASETS):
            values = target_level[dataset][key]
            box = ax_b.boxplot(
                [values],
                positions=[index + offset],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#1A1A1A", "linewidth": 1.1},
                whiskerprops={"linewidth": 0.9},
                capprops={"linewidth": 0.9},
            )
            for patch in box["boxes"]:
                patch.set(facecolor=color, alpha=0.20, edgecolor=color, linewidth=0.9)
            rng = np.random.default_rng(index)
            jitter = rng.uniform(-0.05, 0.05, size=values.size)
            ax_b.scatter(
                index + offset + jitter,
                values,
                s=4,
                color=color,
                alpha=0.5,
                linewidth=0,
                zorder=3,
                label=label if index == 0 else None,
            )
    ax_b.axhline(0.0, color=PALETTE["red_strong"], linewidth=0.9, linestyle="--")
    ax_b.set_xticks(np.arange(len(DATASETS)))
    ax_b.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=5.8)
    ax_b.set_ylabel("R-MUR gain over baseline")
    ax_b.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax_b.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.2)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
