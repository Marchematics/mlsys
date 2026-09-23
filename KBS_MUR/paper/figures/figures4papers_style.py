"""Publication figure helpers adapted from ChenLiu-1996/figures4papers.

The constants and function names follow the *scientific-figure-making* API in
that repository.  Sizes are tuned for single-column KBS figures while keeping
the house conventions: sans-serif type, semantic blue/green/red palette,
minimal spines, print-safe edges, and paired PDF/PNG export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 9.0
    axes_linewidth: float = 1.6
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Nimbus Sans", "DejaVu Sans")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Apply the repository's rcParams conventions."""

    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": list(style.font_family),
            "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size,
            "xtick.labelsize": style.font_size - 0.5,
            "ytick.labelsize": style.font_size - 0.5,
            "legend.fontsize": style.font_size - 1.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    if style.use_tex:
        plt.rcParams["text.usetex"] = True


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Create subplots and return a flattened axes array."""

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def _clean_ax(ax: Axes) -> None:
    ax.grid(False)
    ax.tick_params(length=3.5, width=1.0)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(plt.rcParams["axes.linewidth"])


def make_trend(
    ax: Axes,
    x,
    y_series,
    labels=None,
    colors=None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    show_shadow: bool = True,
    errors=None,
    marker: str = "o",
    linewidth: float = 2.0,
    markersize: float = 4.0,
):
    """Plot one or more trend lines with optional uncertainty bands."""

    x = np.asarray(x, dtype=float)
    ys = [np.asarray(y, dtype=float) for y in y_series]
    if labels is None:
        labels = [None] * len(ys)
    if colors is None:
        colors = DEFAULT_COLORS[: len(ys)]
    if errors is None:
        errors = [None] * len(ys)

    for y, label, color, err in zip(ys, labels, colors, errors):
        if y.shape != x.shape:
            raise ValueError("each trend series must have the same length as x")
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
            label=label,
        )
        if err is not None and show_shadow:
            err = np.asarray(err, dtype=float)
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.12, linewidth=0)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    _clean_ax(ax)
    return ax


def make_grouped_bar(
    ax: Axes,
    categories,
    series,
    labels,
    ylabel: str | None = None,
    colors=None,
    annotate: bool = False,
    bar_width: float = 0.78,
    edgecolor: str = "black",
    linewidth: float = 1.0,
    capsize: float = 3.0,
    errors=None,
) -> BarContainer:
    """Render a grouped bar chart following the repository conventions."""

    categories = list(categories)
    series = [np.asarray(s, dtype=float) for s in series]
    if len(series) == 0:
        raise ValueError("at least one series is required")
    if any(len(s) != len(categories) for s in series):
        raise ValueError("every series must match the number of categories")
    if len(labels) != len(series):
        raise ValueError("labels must match the number of series")
    if colors is None:
        colors = DEFAULT_COLORS[: len(series)]
    elif isinstance(colors, str):
        colors = [colors] * len(series)
    if errors is None:
        errors = [None] * len(series)

    n = len(series)
    positions = np.arange(len(categories), dtype=float)
    total_width = bar_width
    width = total_width / n
    last_bars = None
    for idx, (values, label, color, err) in enumerate(zip(series, labels, colors, errors)):
        offset = (idx - (n - 1) / 2.0) * width
        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            yerr=err,
            capsize=capsize,
            error_kw={"elinewidth": linewidth, "capthick": linewidth},
        )
        last_bars = bars
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    _clean_ax(ax)
    if annotate and last_bars is not None:
        annotate_bars(ax, last_bars, fmt="{:.3f}")
    return last_bars


def annotate_bars(ax: Axes, bars, fmt: str = "{:.2f}", fontsize: float | None = None,
                  padding: float = 2.5) -> None:
    """Annotate bar heights directly above or below each bar."""

    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        offset = padding if height >= 0 else -padding
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=fontsize,
            clip_on=False,
        )


def make_scatter(
    ax: Axes,
    x,
    y,
    label: str | None = None,
    color: str | None = None,
    size: float = 45,
    alpha: float = 0.85,
    marker: str = "o",
    edgecolor: str | None = "white",
    linewidth: float = 0.6,
):
    """Single-series scatter with repository defaults."""

    ax.scatter(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        s=size,
        color=color or PALETTE["blue_main"],
        alpha=alpha,
        marker=marker,
        edgecolor=edgecolor,
        linewidth=linewidth,
        label=label,
    )
    _clean_ax(ax)
    return ax


def finalize_figure(
    fig: plt.Figure,
    out_path: str | Path,
    formats=("pdf", "png"),
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
    tight_layout: bool = True,
    **kwargs,
) -> list[Path]:
    """Save a figure in the requested vector/raster formats."""

    out_path = Path(out_path)
    if out_path.suffix:
        out_path = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout(pad=pad)
    saved = []
    # A fixed creation timestamp keeps regenerated PDFs byte-identical, so the
    # checksums in the submission package can be verified against a rebuild.
    metadata = {"CreationDate": None, "ModDate": None}
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata=metadata if fmt == "pdf" else None,
            **kwargs,
        )
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


# Backwards-compatible aliases for the earlier KBS figure scripts.
COLORS = {
    "relevance": PALETTE["neutral"],
    "mmr": PALETTE["teal"],
    "static": PALETTE["red_strong"],
    "oracle_static": PALETTE["green_3"],
    "mur": PALETTE["blue_main"],
    "oracle": "#4D4D4D",
    "x1": PALETTE["neutral"],
    "x2": PALETTE["blue_main"],
    "x3": PALETTE["red_strong"],
}


def save(fig: plt.Figure, path) -> None:
    finalize_figure(fig, path, formats=("pdf", "png"), dpi=300)
