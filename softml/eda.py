
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plots import histogram, boxplot, scatter, violin, pairplot

def quick_eda(df: pd.DataFrame, target: str | None=None, show_plots: bool=True) -> dict:
    """Return a summary dict and optionally render plots + correlation matrix."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "head": df.head(5),
        "describe": df.describe(include='all', datetime_is_numeric=True)
    }

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        summary["corr"] = corr
        if show_plots:
            plt.figure(figsize=(6,5))
            plt.imshow(corr, interpolation="nearest")
            plt.title("Correlation (numeric)")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.tight_layout(); plt.show()

    if show_plots:
        for col in list(num_df.columns)[:4]:
            histogram(df, col, bins=15, plt_show=True)
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                for col in [c for c in num_df.columns if c != target][:3]:
                    scatter(df, col, target, plt_show=True)
            else:
                for col in list(num_df.columns)[:3]:
                    boxplot(df, col, by=target, plt_show=True)
        if num_df.shape[1] >= 2:
            pairplot(df, plt_show=True)

    return summary
