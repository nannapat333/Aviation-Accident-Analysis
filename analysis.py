# analysis.py
# Updated: severity = line chart (median fatalities per accident by decade),
# plus Chi-square tests for Aircraft Types (RQ3) and Operators (RQ4).

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import datetime as dt
import scipy

# ---------------- Config ----------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = Path("clean.csv")
RAW_FILE   = Path("aviation-accident.csv")
RECENT_FROM = 2017  # for aircraft & airlines scope

# -------------- Helpers -----------------
def save_text(p: Path, s: str): p.write_text(s, encoding="utf-8")
def save_json(p: Path, obj):     p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_dataset() -> pd.DataFrame:
    if CLEAN_FILE.exists():
        df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    else:
        from load_clean import load_and_clean
        df = load_and_clean(str(RAW_FILE))
        df.to_csv(CLEAN_FILE, index=False)
    return df

def check_columns(df: pd.DataFrame):
    required = {"date","year","decade","type","operator","fatalities","is_fatal"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

def categorize_operator(op: str) -> str:
    s = str(op).lower()
    if any(k in s for k in ["air force"," af "," af-","af/"," navy","naval"]):
        return "Military"
    if "cargo" in s or "freight" in s or "logistics" in s:
        return "Cargo"
    if "private" in s or "jet" in s or "charter" in s or "bizjet" in s:
        return "Private"
    if "air" in s or "airlines" in s or "airways" in s:
        return "Commercial"
    return "Unknown"

# -------------- RQ1: Trend --------------
def make_trend_tables(df: pd.DataFrame):
    year = (df.groupby("year")
              .agg(accidents=("date","count"),
                   fatal_accidents=("is_fatal","sum"))
              .reset_index())
    year.to_csv(REPORTS_DIR / "trend_year.csv", index=False)

    decade = (df.groupby("decade")
                .agg(accidents=("date","count"),
                     fatal_accidents=("is_fatal","sum"))
                .reset_index()
                .sort_values("decade"))
    decade.to_csv(REPORTS_DIR / "trend_decade.csv", index=False)

    # GLM trend check
    if len(year) >= 3:
        y2 = year.copy()
        y2["year_centered"] = y2["year"] - y2["year"].mean()
        poi = smf.glm("accidents ~ year_centered", data=y2, family=sm.families.Poisson()).fit()
        over = poi.deviance / poi.df_resid if poi.df_resid > 0 else np.nan
        note = [f"Poisson GLM accidents ~ year: coef={poi.params.get('year_centered', np.nan):.4f}, p={poi.pvalues.get('year_centered', np.nan):.4g}",
                f"Overdispersion ratio: {over:.2f}"]
        save_text(REPORTS_DIR / "trend_stats.txt", "\n".join(note))

# -------------- RQ2: Severity --------------
def make_severity_decade(df: pd.DataFrame):
    # keep only median fatalities per accident per decade
    sev = (df.groupby("decade")["fatalities"]
             .agg(median="median")
             .reset_index()
             .sort_values("decade"))
    sev.to_csv(REPORTS_DIR / "severity_decade_summary.csv", index=False)

# -------------- RQ3: Aircraft Types --------------
def make_aircraft_recent(df: pd.DataFrame, start: int = RECENT_FROM):
    recent = df[df["year"] >= start]
    types = (recent.groupby("type")
             .agg(accidents=("date","count"))
             .reset_index()
             .sort_values("accidents", ascending=False))
    types.to_csv(REPORTS_DIR / "aircraft_types_recent.csv", index=False)

    # Chi-square: are some types over-represented?
    top = types[types["accidents"] >= 5]  # avoid tiny categories
    chi2, p = stats.chisquare(top["accidents"].values)
    save_text(REPORTS_DIR / "aircraft_chisq.txt",
              f"Chi-square across aircraft types (>=5 accidents, {start}+): chi2={chi2:.2f}, p={p:.4g}, k={len(top)}")

# -------------- RQ4: Operators --------------
def make_operators_recent(df: pd.DataFrame, start: int = RECENT_FROM):
    recent = df[df["year"] >= start].copy()
    ops = (recent.groupby("operator")
           .agg(accidents=("date","count"))
           .reset_index())
    ops["category"] = ops["operator"].apply(categorize_operator)

    # Drop Military + Unknown
    ops = ops[~ops["category"].isin(["Military","Unknown"])]
    ops.to_csv(REPORTS_DIR / "operators_recent.csv", index=False)

    # Chi-square across categories
    cat_counts = ops.groupby("category")["accidents"].sum().reset_index()
    chi2, p = stats.chisquare(cat_counts["accidents"].values)
    save_text(REPORTS_DIR / "operator_chisq.txt",
              f"Chi-square across operator categories ({start}+): chi2={chi2:.2f}, p={p:.4g}, k={len(cat_counts)}")

# -------------- Main ----------------------
def main():
    df = load_dataset()
    check_columns(df)
    df.to_csv(REPORTS_DIR / "dataset_clean_snapshot.csv", index=False)

    meta = {
        "generated_at": dt.datetime.utcnow().isoformat()+"Z",
        "rows_total": int(len(df)),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "recent_scope_start": RECENT_FROM,
        "python_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
        }
    }
    save_json(REPORTS_DIR / "meta.json", meta)

    make_trend_tables(df)
    make_severity_decade(df)
    make_aircraft_recent(df, RECENT_FROM)
    make_operators_recent(df, RECENT_FROM)

    print("Done. Reports and CSVs ready in ./reports")

if __name__ == "__main__":
    main()




