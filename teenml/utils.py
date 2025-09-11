
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as sk_split

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV into a pandas DataFrame."""
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str | Path, index: bool=False) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=index)

def train_test_split(df: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    """Simple wrapper for sklearn's train_test_split that splits whole dataframes."""
    return sk_split(df, test_size=test_size, random_state=random_state)
