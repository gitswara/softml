
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plots import histogram, boxplot, scatter, violin, pairplot

def quick_eda(df: pd.DataFrame, target: str | None=None, show_plots: bool=True) -> dict:
    """
    Return a summary dict and optionally render plots + correlation matrix.
    summary keys: shape, dtypes, missing_pct, head, describe, (optional) corr
    """

    try:
        desc = df.describe(include='all', datetime_is_numeric=True)
    except TypeError:
        desc = df.describe(include='all')
    
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "head": df.head(5),
        "describe": desc,
    }
    
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        summary["corr"] = corr

        if show_plots:
            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)

            # Set axis ticks
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            # Annotate each cell with correlation value
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(
                        j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black"
                    )

            plt.title("Correlation Matrix (numeric)", pad=20)
            plt.tight_layout()
            plt.show()

    if show_plots:
        # Histograms for up to 4 numeric columns
        for col in list(num_df.columns)[:4]:
            histogram(df, col, bins=15, plt_show=True)

        # Target relationships
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                for col in [c for c in num_df.columns if c != target][:3]:
                    scatter(df, col, target, plt_show=True)
            else:
                for col in list(num_df.columns)[:3]:
                    boxplot(df, col, by=target, plt_show=True)

        # Pairplot if enough numeric columns
        if num_df.shape[1] >= 2:
            pairplot(df, plt_show=True)

    return summary
