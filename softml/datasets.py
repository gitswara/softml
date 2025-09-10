
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence, Optional, Union, List
import numpy as np
import pandas as pd

def _auto_cast(x: str) -> Any:
    s = x.strip()
    if s == "":
        return np.nan
    if s.lower() in ("true","false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return s

def create_dataset_interactive(
    save_path: Union[str, Path] = "my_dataset.csv",
    fields: Optional[List[str]] = None,
    cast: bool = True,
    stop_word: str = "done",
    echo: bool = True,
) -> pd.DataFrame:
    sp = Path(save_path)
    if not fields:
        print("Enter column names separated by commas (e.g., name,age,city):")
        line = input().strip()
        fields = [c.strip() for c in line.split(",") if c.strip()]
        if not fields:
            raise ValueError("No columns provided.")
    if echo:
        print(f"[ok] Columns: {fields}")
    print(f"""Enter one row per line using commas for fields.\n- Order must match: {fields}\n- Type '{stop_word}' on an empty line to finish.""" )
    rows: List[List[Any]] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line or line.strip().lower() == stop_word:
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            print(f"  (!) Expected {len(fields)} values; got {len(parts)}. Try again.")
            continue
        if cast:
            parts = [_auto_cast(p) for p in parts]
        rows.append(parts)
    df = pd.DataFrame(rows, columns=fields)
    df.to_csv(sp, index=False)
    if echo:
        print(f"[ok] Saved {len(df)} rows to {sp.resolve()}")
    return df

def create_dataset_from_lists(
    fields: Sequence[str],
    data_rows: Iterable[Sequence[Any]],
    save_path: Union[str, Path] = "my_dataset.csv",
    cast: bool = False,
) -> pd.DataFrame:
    if cast:
        def _cast_row(row): return [_auto_cast(str(v)) for v in row]
        rows = [_cast_row(r) for r in data_rows]
    else:
        rows = [list(r) for r in data_rows]
    df = pd.DataFrame(rows, columns=list(fields))
    df.to_csv(save_path, index=False)
    return df
