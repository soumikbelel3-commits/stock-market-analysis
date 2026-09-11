"""One matplotlib theme and a small set of chart helpers.

Every notebook calls ``viz.setup()`` in its first cell so the six read as one
document rather than six different-looking reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config

# Colour-blind-safe, ordered so the first few are maximally distinct.
PALETTE = [
    "#2b6cb0", "#c05621", "#2f855a", "#b83280", "#6b46c1",
    "#b7791f", "#2c7a7b", "#9b2c2c", "#4a5568", "#3182ce",
    "#d69e2e", "#38a169", "#805ad5", "#dd6b20", "#319795",
]

POS, NEG, NEUTRAL = "#2f855a", "#c53030", "#718096"
GRID = "#e2e8f0"
INK = "#1a202c"
MUTED = "#718096"


def setup(figsize=(12, 5.5), dpi=110) -> None:
    """Apply the project theme. Idempotent; safe to call in every notebook."""
    mpl.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e0",
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


def save(fig, name: str) -> str:
    """Write a chart to outputs/charts and return its path."""
    path = config.CHART_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    return str(path)


def pct_axis(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def title(ax, main: str, sub: str = "") -> None:
    ax.set_title(main if not sub else f"{main}\n{sub}", loc="left")
    if sub:
        ax.title.set_fontsize(11)


def mark_crises(ax, events=config.CRISIS_CHAIN, y_frac: float = 0.96,
                fontsize: int = 7.5) -> None:
    """Draw the chronological crisis chain on a time-axis chart."""
    lo, hi = ax.get_xlim()
    for i, (date, label) in enumerate(events):
        x = mpl.dates.date2num(pd.Timestamp(date))
        if not (lo <= x <= hi):
            continue
        ax.axvline(x, color=MUTED, ls=":", lw=0.9, alpha=0.7, zorder=0)
        ax.annotate(label, xy=(x, ax.get_ylim()[0]),
                    xytext=(x, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])
                            * (y_frac - 0.07 * (i % 2))),
                    fontsize=fontsize, color=MUTED, rotation=90,
                    ha="right", va="top")


def line(df: pd.DataFrame, ax=None, title_="", sub="", ylabel="",
         highlight: str | None = None, legend_out: bool = False):
    """Multi-series line chart with an optional highlighted series."""
    if ax is None:
        _, ax = plt.subplots()
    for i, c in enumerate(df.columns):
        is_hi = highlight is not None and c == highlight
        ax.plot(df.index, df[c], label=str(c),
                lw=2.4 if is_hi else 1.4,
                color="#1a202c" if is_hi else PALETTE[i % len(PALETTE)],
                alpha=1.0 if is_hi else 0.85, zorder=3 if is_hi else 2)
    title(ax, title_, sub)
    ax.set_ylabel(ylabel)
    if legend_out:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    else:
        ax.legend(ncol=min(4, max(1, len(df.columns))))
    return ax


def ranked_bar(s: pd.Series, ax=None, title_="", sub="", xlabel="",
               positive_is_good: bool = True, fmt: str = "{:+.1f}"):
    """Horizontal ranked bars, coloured by sign."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(3.2, 0.32 * len(s))))
    s = s.dropna().sort_values()
    colors = [(POS if (v >= 0) == positive_is_good else NEG) for v in s.values]
    ax.barh(range(len(s)), s.values, color=colors, alpha=0.85, height=0.72)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([str(i) for i in s.index], fontsize=9)
    ax.axvline(0, color=INK, lw=1)
    ax.grid(axis="y", visible=False)
    span = (s.max() - s.min()) or 1
    for i, v in enumerate(s.values):
        ax.text(v + span * 0.012 * (1 if v >= 0 else -1), i, fmt.format(v),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=MUTED)
    ax.set_xlim(s.min() - span * 0.16, s.max() + span * 0.16)
    title(ax, title_, sub)
    ax.set_xlabel(xlabel)
    return ax


def heatmap(df: pd.DataFrame, ax=None, title_="", sub="", cmap="RdYlGn",
            fmt: str = "{:.1f}", vmin=None, vmax=None, cbar_label=""):
    """Annotated matrix. Used for correlations and the valuation grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(min(16, 1.0 + 0.85 * df.shape[1]),
                                      min(14, 1.2 + 0.42 * df.shape[0])))
    data = df.astype(float).values
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=45,
                       ha="right", fontsize=8.5)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], fontsize=8.5)
    ax.grid(visible=False)
    finite = data[np.isfinite(data)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v - mid) > 0.34 * (finite.max() - finite.min() or 1)
                    else INK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)
    title(ax, title_, sub)
    return ax


def quadrant(x: pd.Series, y: pd.Series, labels=None, ax=None, title_="", sub="",
             xlabel="", ylabel="", x_split: float = 0.0, y_split: float = 0.0,
             quad_names=("Leading", "Improving", "Lagging", "Weakening"),
             annotate: bool = True):
    """Scatter split into four quadrants -- the sector rotation view.

    quad_names order: (top-right, top-left, bottom-left, bottom-right).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 8))
    j = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    labels = list(j.index) if labels is None else labels
    ax.axvline(x_split, color=MUTED, lw=1.1)
    ax.axhline(y_split, color=MUTED, lw=1.1)

    colors = []
    for xv, yv in zip(j["x"], j["y"]):
        if xv >= x_split and yv >= y_split:
            colors.append(POS)
        elif xv < x_split and yv >= y_split:
            colors.append("#3182ce")
        elif xv < x_split and yv < y_split:
            colors.append(NEG)
        else:
            colors.append("#b7791f")
    ax.scatter(j["x"], j["y"], s=70, c=colors, alpha=0.8, edgecolor="white", zorder=3)
    if annotate:
        for (xi, yi), lab in zip(j.values, labels):
            ax.annotate(str(lab), (xi, yi), fontsize=7.5, xytext=(5, 4),
                        textcoords="offset points", color=INK)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    for name, (px, py, ha, va) in zip(
            quad_names,
            [(xhi, yhi, "right", "top"), (xlo, yhi, "left", "top"),
             (xlo, ylo, "left", "bottom"), (xhi, ylo, "right", "bottom")]):
        ax.text(px, py, name, fontsize=9, color=MUTED, alpha=0.85,
                ha=ha, va=va, style="italic")
    title(ax, title_, sub)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def price_panel(df: pd.DataFrame, name: str, rsi_s=None, dma=None,
                levels: dict | None = None, figsize=(12.5, 8)):
    """Annotated price chart: price + DMAs (+ optional RSI and ATR levels)."""
    from . import indicators as ind

    n_rows = 3 if rsi_s is not None else 2
    heights = [3, 1, 1] if n_rows == 3 else [3, 1]
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": heights})
    ax = axes[0]
    ax.plot(df.index, df["Close"], color=INK, lw=1.5, label="Close", zorder=3)
    dma = ind.dma_set(df["Close"]) if dma is None else dma
    for i, c in enumerate(dma.columns):
        ax.plot(dma.index, dma[c], lw=1.1, color=PALETTE[i], alpha=0.9, label=c)

    if levels:
        for key, color, ls in [("stop_2atr", NEG, "--"),
                               ("target_3atr", POS, "--"),
                               ("swing_high_60d", MUTED, ":"),
                               ("swing_low_60d", MUTED, ":")]:
            if key in levels and np.isfinite(levels[key]):
                ax.axhline(levels[key], color=color, ls=ls, lw=1.0, alpha=0.8)
                ax.annotate(f"{key.replace('_', ' ')} {levels[key]:,.0f}",
                            xy=(df.index[-1], levels[key]), fontsize=7.5,
                            color=color, ha="right", va="bottom")
    title(ax, name, "Daily close with 20/50/200 DMA"
          + (" and ATR-derived levels" if levels else ""))
    ax.legend(ncol=4, fontsize=8)

    axv = axes[1]
    if "Volume" in df.columns:
        axv.bar(df.index, df["Volume"], color=MUTED, alpha=0.55, width=1.2)
        axv.plot(df.index, df["Volume"].rolling(20).mean(), color=PALETTE[1], lw=1.1)
        axv.set_ylabel("Volume")
        axv.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v/1e6:,.0f}M" if v >= 1e6 else f"{v:,.0f}"))

    if rsi_s is not None:
        axr = axes[2]
        axr.plot(rsi_s.index, rsi_s.values, color=PALETTE[4], lw=1.2)
        axr.axhline(70, color=NEG, ls="--", lw=0.9)
        axr.axhline(30, color=POS, ls="--", lw=0.9)
        axr.axhline(50, color=MUTED, lw=0.8, alpha=0.6)
        axr.set_ylim(0, 100)
        axr.set_ylabel("RSI(14)")
    fig.align_ylabels(axes)
    return fig, axes


def table_note(text: str) -> None:
    """Consistent one-line caption under a printed table."""
    print(f"   └─ {text}")
