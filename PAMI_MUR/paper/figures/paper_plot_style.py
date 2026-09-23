"""Compatibility shim for old KBS figure scripts.

All new figure code should import :mod:`figures4papers_style`, which follows
the conventions in ``ChenLiu-1996/figures4papers``.
"""

from figures4papers_style import (  # noqa: F401
    COLORS,
    DEFAULT_COLORS,
    PALETTE,
    FigureStyle,
    apply_publication_style,
    annotate_bars,
    finalize_figure,
    make_grouped_bar,
    make_scatter,
    make_trend,
    save,
)
