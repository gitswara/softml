
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

def _show(plt_show: bool):
    if plt_show:
        plt.tight_layout()
        plt.show()

def lineplot(df: pd.DataFrame, x: str, y: str, plt_show: bool=True):
    df.plot(x=x, y=y)
    plt.title(f"Line: {y} over {x}")
    plt.xlabel(x); plt.ylabel(y)
    _show(plt_show)

def boxplot(df: pd.DataFrame, column: str, by: str | None=None, plt_show: bool=True):
    if by:
        df.boxplot(column=column, by=by)
        plt.title(f"Boxplot of {column} by {by}"); plt.suptitle("")
    else:
        df.boxplot(column=column)
        plt.title(f"Boxplot of {column}")
    plt.ylabel(column)
    _show(plt_show)

def histogram(df: pd.DataFrame, column: str, bins: int=15, plt_show: bool=True):
    df[column].plot(kind="hist", bins=bins)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column); plt.ylabel("Count")
    _show(plt_show)

def violin(df: pd.DataFrame, column: str, by: str | None=None, plt_show: bool=True):
    if by and by in df.columns:
        groups = [g[column].dropna() for _, g in df.groupby(by)]
        plt.violinplot(groups, showmeans=True)
        plt.xticks(range(1, len(groups)+1), [str(k) for k in df[by].dropna().unique()], rotation=30)
        plt.title(f"Violin: {column} by {by}")
    else:
        plt.violinplot(df[column].dropna(), showmeans=True)
        plt.title(f"Violin: {column}")
    plt.ylabel(column)
    _show(plt_show)

def scatter(df: pd.DataFrame, x: str, y: str, hue: str | None=None, plt_show: bool=True):
    if hue and hue in df.columns:
        for k, g in df.groupby(hue):
            plt.scatter(g[x], g[y], label=str(k))
        plt.legend(title=hue)
    else:
        plt.scatter(df[x], df[y])
    plt.title(f"Scatter: {x} vs {y}")
    plt.xlabel(x); plt.ylabel(y)
    _show(plt_show)

def pairplot(df: pd.DataFrame, plt_show: bool=True):
    scatter_matrix(df.select_dtypes(include='number'), diagonal='kde', figsize=(7,7))
    plt.suptitle("Pairwise Scatter (numeric columns)")
    _show(plt_show)
