# load_clean.py
# Usage:
#   from load_clean import load_and_clean
#   df = load_and_clean("data/accidents_raw.csv",
#                       out_csv="data/accidents_clean_1950+.csv")

import pandas as pd
import numpy as np
from typing import Optional

TEXT_COLS = ["type", "registration", "operator", "location", "country", "cat"]

def _fix_text(s: pd.Series) -> pd.Series:
    """
    Attempt to fix common mojibake like 'LiorÃ©' -> 'Lioré'.
    Safe no-op if data is already clean.
    """
    def _fix_one(x):
        if pd.isna(x):
            return x
        x = str(x)
        try:
            # if it was decoded as latin-1 but originally utf-8
            return x.encode("latin1").decode("utf-8")
        except Exception:
            return x
    return s.apply(_fix_one).str.strip()

def load_and_clean(path: str = "aviation-accident.csv",
                   out_csv: Optional[str] = None,
                   start_year: int = 1950,
                   end_year: Optional[int] = None) -> pd.DataFrame:
    # Read CSV (try UTF-8, fallback to latin-1 if needed)
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Normalize column names to exactly what you showed
    expected = ["date","type","registration","operator","fatalities","location","country","cat","year"]
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    missing = [c for c in ["date","type","operator","fatalities","country","year"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    # Parse date like '26-OCT-1932' (day-first, 3-letter month)
    # If parse fails for any rows, they'll be dropped
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df = df[df["date"].notna()].copy()

    # Recompute year from date to avoid inconsistencies, keep original as year_orig
    df["year_orig"] = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Filter to 1950..present (or provided end_year)
    if end_year is None:
        end_year = int(df["year"].max())
    df = df[df["year"].between(start_year, end_year)].copy()

    # Clean text columns, fix mojibake, and strip spaces
    for c in TEXT_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = _fix_text(df[c])

    # Fatalities numeric & target flag
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["is_fatal"] = (df["fatalities"] > 0).astype(int)

    # Helpful time bucket
    df["decade"] = (df["year"] // 10) * 10

    # Reorder/trim columns
    keep = [
        "date", "year", "decade",
        "type", "registration", "operator",
        "location", "country", "cat",
        "fatalities", "is_fatal",
        "year_orig"
    ]
    df = df[keep].sort_values(["year","date"]).reset_index(drop=True)

    if out_csv:
        df.to_csv(out_csv, index=False)

    return df

if __name__ == "__main__":
    cleaned = load_and_clean("aviation-accident.csv",
                             out_csv="clean.csv",
                             start_year=1950)
    print(cleaned.head(10))
    print(f"\nRows kept (>=1950): {len(cleaned)} | Years: {cleaned['year'].min()}–{cleaned['year'].max()}")

