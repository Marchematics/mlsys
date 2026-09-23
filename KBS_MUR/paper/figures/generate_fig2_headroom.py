"""Figure: structural headroom in the controlled environment and on the six
benchmarks.

Panel A uses the controlled redundancy ladder
(``results/raw/R070_oracle_ladder_*``); panel B uses the per-target oracle
audit (``results/raw/R080a_multitarget_oracle_*/<dataset>_s<seed>/records.json``).
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
CONTROLLED = ROOT / "results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json"
ORACLE_DIRS = [
    ROOT / "results/raw/R080a_multitarget_oracle_s5_20260920_2245",
    ROOT / "results/raw/R080a_multitarget_oracle_s5_20260920_2300",
]
OUT = Path(__file__).resolve().parent / "fig2_headroom"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LABELS = ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def controlled_headroom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = json.loads(CONTROLLED.read_text())["records"]
    ratios = sorted({float(r["redundancy"]) for r in records})
    means, stds = [], []
    for ratio in ratios:
        rows = [r for r in records if float(r["redundancy"]) == ratio]
        values = np.asarray([r["metrics"]["state_conditioning_headroom"] for r in rows], dtype=float)
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)))
    return np.asarray(ratios), np.asarray(means), np.asarray(stds)


def target_headroom() -> list[np.ndarray]:
    per_dataset: list[list[float]] = [[] for _ in DATASETS]
    for root in ORACLE_DIRS:
        if not root.exists():
            continue
        for index, dataset in enumerate(DATASETS):
            for run_dir in sorted(root.glob(f"{dataset}_s*")):
                records_path = run_dir / "records.json"
                if not records_path.exists():
                    continue
                for record in json.loads(records_path.read_text()):
                    per_dataset[index].append(float(record["state_conditioning_headroom"]))
    return [np.asarray(values, dtype=float) for values in per_dataset]


def main() -> None:
    apply_publication_style(FigureStyle(font_size=7.0, axes_linewidth=1.2))
    fig, axes = create_subplots(1, 2, figsize=(5.4, 2.3), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax_a, ax_b = axes

    ratios, means, stds = controlled_headroom()
    make_trend(
        ax_a,
        ratios,
        [means],
        ylabel=r"Structural headroom $H_{\mathrm{state}}$",
        xlabel="Duplicate ratio",
        colors=[PALETTE["green_3"]],
        errors=[stds],
        linewidth=2.0,
        markersize=4.0,
    )
    for x, y in zip(ratios, means):
        ax_a.annotate(f"{y:.3f}".lstrip("0"), (x, y), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=6)
    ax_a.set_xticks(ratios)
    ax_a.set_xticklabels([f"{v:g}" for v in ratios])
    ax_a.set_ylim(-0.02, 0.34)

    values = target_headroom()
    positions = np.arange(len(values))
    box = ax_b.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for patch in box["boxes"]:
        patch.set(facecolor=PALETTE["blue_secondary"], alpha=0.20, edgecolor=PALETTE["blue_main"], linewidth=0.9)
    rng = np.random.default_rng(0)
    for index, series in enumerate(values):
        jitter = rng.uniform(-0.12, 0.12, size=series.size)
        ax_b.scatter(index + jitter, series, s=2.5, color=PALETTE["blue_main"], alpha=0.35, linewidth=0)
        ax_b.annotate(
            f"{series.mean():.3f}".lstrip("0"),
            (index, series.mean()),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=5.8,
            color="#4D4D4D",
        )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(LABELS, rotation=30, ha="right")
    ax_b.set_ylabel(r"Target-level $H_{\mathrm{state}}$")
    ax_b.grid(axis="y", alpha=0.2, linewidth=0.5)

    finalize_figure(fig, OUT, formats=("pdf", "png"), dpi=300, pad=0.10)


if __name__ == "__main__":
    main()
