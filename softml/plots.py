# softml/plots.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from typing import Optional

def _style():
    # Light, readable defaults without extra dependencies
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (7, 4.5),
        "grid.alpha": 0.3,
    })

def _show(plt_show: bool):
    if plt_show:
        plt.tight_layout()
        plt.show()

def lineplot(df: pd.DataFrame, x: str, y: str, plt_show: bool=True, title: Optional[str]=None):
    _style()
    ax = df.plot(x=x, y=y, kind="line", marker="o", linewidth=2, alpha=0.9)
    ax.grid(True)
    ax.set_title(title or f"{y} over {x}")
    ax.set_xlabel(x); ax.set_ylabel(y)
    # annotate last point
    try:
        xv = df[x].iloc[-1]; yv = df[y].iloc[-1]
        ax.annotate(f"{yv:.2f}", xy=(xv, yv), xytext=(5,5), textcoords="offset points")
    except Exception:
        pass
    _show(plt_show)

def boxplot(df: pd.DataFrame, column: str, by: str | None=None, plt_show: bool=True, title: Optional[str]=None):
    _style()
    # Filled boxes
    bp = df.boxplot(column=column, by=by, patch_artist=True, return_type='dict')
    # bp is dict-of-dicts when by is None; normalize to list of artists
    artists = []
    if isinstance(bp, dict) and "boxes" in bp:
        artists = bp["boxes"]
    elif isinstance(bp, dict):
        for k in bp:
            artists.extend(bp[k]["boxes"])
    # color fill
    for i, b in enumerate(artists):
        b.set_facecolor(plt.cm.Blues(0.3 + 0.05*i))
        b.set_edgecolor("black")
        b.set_alpha(0.9)

    plt.title(title or (f"Boxplot of {column} by {by}" if by else f"Boxplot of {column}"))
    plt.suptitle("")  # remove pandas default subtitle
    plt.xlabel(by if by else "")
    plt.ylabel(column)
    plt.grid(axis="y")

    # annotate medians
    try:
        medians = []
        if by:
            med = df.groupby(by)[column].median().values
            medians = med
        else:
            medians = [df[column].median()]
        ax = plt.gca()
        xticks = ax.get_xticks()
        for i, m in enumerate(medians):
            ax.text(xticks[i], m, f"{m:.2f}", ha="center", va="bottom", fontsize=9, color="black")
    except Exception:
        pass

    _show(plt_show)

def histogram(df: pd.DataFrame, column: str, bins: int=15, plt_show: bool=True, density: bool=False, title: Optional[str]=None):
    _style()
    vals = df[column].dropna().values
    fig, ax = plt.subplots()
    counts, edges, patches = ax.hist(vals, bins=bins, edgecolor="white", alpha=0.9)
    ax.set_title(title or f"Histogram of {column}")
    ax.set_xlabel(column); ax.set_ylabel("Density" if density else "Count")
    ax.grid(axis="y")

    # annotate counts on bars
    for c, p in zip(counts, patches):
        if c > 0:
            ax.annotate(f"{int(c)}", xy=(p.get_x() + p.get_width()/2, c),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    _show(plt_show)

def violin(df: pd.DataFrame, column: str, by: str | None=None, plt_show: bool=True, title: Optional[str]=None):
    _style()
    fig, ax = plt.subplots()
    if by and by in df.columns:
        order = [k for k, _ in df.groupby(by)]
        groups = [g[column].dropna().values for _, g in df.groupby(by)]
        parts = ax.violinplot(groups, showmeans=True, showmedians=True, showextrema=False)
        # color bodies
        for i, b in enumerate(parts['bodies']):
            b.set_facecolor(plt.cm.Oranges(0.35 + 0.05*i))
            b.set_edgecolor("black")
            b.set_alpha(0.9)
        ax.set_xticks(range(1, len(order)+1))
        ax.set_xticklabels([str(k) for k in order], rotation=15)
        ax.set_title(title or f"Violin: {column} by {by}")
        ax.set_xlabel(by)
    else:
        parts = ax.violinplot(df[column].dropna().values, showmeans=True, showmedians=True, showextrema=False)
        for b in parts['bodies']:
            b.set_facecolor(plt.cm.Oranges(0.6)); b.set_edgecolor("black"); b.set_alpha(0.9)
        ax.set_title(title or f"Violin: {column}")
        ax.set_xlabel("")
    ax.set_ylabel(column)
    ax.grid(axis="y")

    # annotate median(s)
    try:
        if by and by in df.columns:
            meds = [np.median(g) for _, g in df.groupby(by)[column]]
            for i, m in enumerate(meds, start=1):
                ax.annotate(f"{m:.2f}", xy=(i, m), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)
        else:
            m = np.median(df[column].dropna())
            ax.annotate(f"{m:.2f}", xy=(1, m), xytext=(5, 5), textcoords="offset points",
                        ha="left", va="bottom", fontsize=9)
    except Exception:
        pass

    _show(plt_show)

def scatter(df: pd.DataFrame, x: str, y: str, hue: str | None=None, plt_show: bool=True, title: Optional[str]=None):
    _style()
    fig, ax = plt.subplots()
    if hue and hue in df.columns:
        series = df[hue]
        if pd.api.types.is_numeric_dtype(series):
            # continuous hue → use colormap with colorbar
            sc = ax.scatter(df[x], df[y], c=series, cmap="viridis", alpha=0.9, edgecolor="white", linewidth=0.5)
            cbar = fig.colorbar(sc, ax=ax); cbar.set_label(hue)
        else:
            # categorical hue → discrete colors
            cats = series.astype(str).unique()
            cmap = plt.cm.get_cmap("tab10", len(cats))
            for i, c in enumerate(cats):
                g = df[series.astype(str) == c]
                ax.scatter(g[x], g[y], label=str(c), alpha=0.9, edgecolor="white", linewidth=0.5, c=[cmap(i)])
            ax.legend(title=hue, frameon=True)
    else:
        ax.scatter(df[x], df[y], alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.grid(True)
    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(x); ax.set_ylabel(y)

    # optional: annotate a few points at extremes
    try:
        sel = df[[x, y]].dropna()
        for idx in sel.nlargest(1, y).index.tolist() + sel.nsmallest(1, y).index.tolist():
            ax.annotate(str(idx), xy=(df.loc[idx, x], df.loc[idx, y]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    except Exception:
        pass

    _show(plt_show)

def pairplot(df: pd.DataFrame, plt_show: bool=True, title: Optional[str]="Pairwise Scatter (numeric columns)"):
    _style()
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        print("pairplot: need at least two numeric columns")
        return
    axes = scatter_matrix(num, figsize=(8, 8), diagonal='kde')
    # improve diagonals + labels
    for ax in np.ravel(axes):
        ax.grid(True, alpha=0.2)
    plt.suptitle(title)
    _show(plt_show)
