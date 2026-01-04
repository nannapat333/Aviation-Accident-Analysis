# analysis_final_noplots.py
# RQs:
# 1) Have aviation accidents decreased significantly over the decades?
# 2) Do certain decades show differences in accident severity (fatalities per accident)?
# 3) Are some aircraft types or airlines more frequently involved in accidents?
#
# Outputs (in ./reports):
# - annual_summary.csv, trend_glm.txt
# - decade_summary.csv, decade_severity_kruskal.txt, (optional) decade_pairwise_tests.csv
# - aircraft_type_counts.csv, aircraft_type_kruskal.txt
# - operator_normalized_counts.csv, operator_chi2.txt
# - dataset_clean_snapshot.csv

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------- Config ----------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Your files are in the same folder as this script:
CLEAN_FILE = Path("clean.csv")
RAW_FILE = Path("aviation-accident.csv")

# ---------- Helpers ----------
def save_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def epsilon_squared_kruskal(H, k, n):
    # ε² = (H - k + 1) / (n - k)
    return (H - k + 1) / (n - k) if (n - k) > 0 else np.nan

def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        rank = m - i
        val = min(1.0, pvals[idx] * rank)
        adj[idx] = max(val, prev)
        prev = adj[idx]
    return adj

# ---------- Load data ----------
if CLEAN_FILE.exists():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
else:
    # Uses your cleaner that filters 1950+ and standardizes columns
    from load_clean import load_and_clean
    df = load_and_clean(str(RAW_FILE))
    df.to_csv(CLEAN_FILE, index=False)

required = {"date","year","decade","type","registration","operator","location","country","cat","fatalities","is_fatal"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Columns present: {list(df.columns)}")

# ---------- RQ1: Trend in accident counts ----------
annual = (df.groupby("year")
            .agg(accidents=("date","count"),
                 fatal_accidents=("is_fatal","sum"),
                 fatalities_total=("fatalities","sum"))
            .reset_index())
annual["fatal_rate"] = annual["fatal_accidents"] / annual["accidents"]
annual.to_csv(REPORTS_DIR / "annual_summary.csv", index=False)

# Poisson GLM on accident counts vs year (centered)
annual2 = annual.copy()
annual2["year_centered"] = annual2["year"] - annual2["year"].mean()
poisson = smf.glm("accidents ~ year_centered", data=annual2, family=sm.families.Poisson()).fit()
overdisp_ratio = poisson.deviance / poisson.df_resid if poisson.df_resid > 0 else np.nan

trend_lines = []
trend_lines.append("=== Poisson GLM: accidents ~ year_centered ===")
trend_lines.append(poisson.summary().as_text())
trend_lines.append(f"\nOverdispersion ratio (deviance/df_resid): {overdisp_ratio:.2f}")

# If overdispersed, also fit Negative Binomial
if np.isfinite(overdisp_ratio) and (overdisp_ratio > 1.5):
    nb = smf.glm("accidents ~ year_centered", data=annual2,
                 family=sm.families.NegativeBinomial()).fit()
    trend_lines.append("\n=== Negative Binomial GLM (overdispersion detected) ===")
    trend_lines.append(nb.summary().as_text())


save_text(REPORTS_DIR / "trend_glm.txt", "\n".join(trend_lines))

# ---------- RQ2: Severity across decades ----------
decade = (df.groupby("decade")
            .agg(accidents=("date","count"),
                 fatal_accidents=("is_fatal","sum"),
                 fatalities_total=("fatalities","sum"),
                 median_fatalities=("fatalities","median"),
                 mean_fatalities=("fatalities","mean"),
                 pct_fatal=("is_fatal","mean"))
            .reset_index()
            .sort_values("decade"))
decade.to_csv(REPORTS_DIR / "decade_summary.csv", index=False)

# Kruskal–Wallis on fatalities per accident by decade
groups = [g["fatalities"].values for _, g in df.groupby("decade")]
H, p_kw = stats.kruskal(*groups)
k = len(groups); n = len(df)
eps2 = epsilon_squared_kruskal(H, k, n)

rq2_txt = []
rq2_txt.append("=== Kruskal–Wallis: fatalities per accident by decade ===")
rq2_txt.append(f"H={H:.2f}, p={p_kw:.4g}, epsilon^2={eps2:.3f}, groups={k}, n={n}")

# Optional pairwise (Holm-adjusted) if global test significant
if p_kw < 0.05 and k >= 2:
    dec_vals = {d: g["fatalities"].values for d, g in df.groupby("decade")}
    decades_sorted = sorted(dec_vals.keys())
    comps = []; pvals = []
    for i in range(len(decades_sorted)):
        for j in range(i+1, len(decades_sorted)):
            d1, d2 = decades_sorted[i], decades_sorted[j]
            x, y = dec_vals[d1], dec_vals[d2]
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            comps.append((d1, d2, U, p, np.median(x), np.median(y)))
            pvals.append(p)
    if pvals:
        adj = holm_correction(pvals)
        out = pd.DataFrame(comps, columns=["decade_1","decade_2","U","p_raw","median_1","median_2"])
        out["p_holm"] = adj
        out = out.sort_values("p_holm")
        out.to_csv(REPORTS_DIR / "decade_pairwise_tests.csv", index=False)
        rq2_txt.append("\nPairwise decade tests saved to decade_pairwise_tests.csv (Holm-adjusted).")
else:
    rq2_txt.append("Global test not significant or insufficient groups; no pairwise table produced.")

save_text(REPORTS_DIR / "decade_severity_kruskal.txt", "\n".join(rq2_txt))

# ---------- RQ3: Aircraft types & operators ----------
# Aircraft type counts + severity
type_counts = (df.groupby("type")
                 .agg(accidents=("date","count"),
                      fatal_accidents=("is_fatal","sum"),
                      fatalities_total=("fatalities","sum"),
                      pct_fatal=("is_fatal","mean"),
                      median_fatalities=("fatalities","median"))
                 .reset_index()
                 .sort_values(["accidents","fatal_accidents"], ascending=False))
type_counts.to_csv(REPORTS_DIR / "aircraft_type_counts.csv", index=False)

# Kruskal across common types (avoid tiny groups)
common_types = type_counts[type_counts["accidents"] >= 30]["type"]
subset = df[df["type"].isin(common_types)]
if len(common_types) >= 3:
    groups_t = [g["fatalities"].values for _, g in subset.groupby("type")]
    H_t, p_t = stats.kruskal(*groups_t)
    k_t, n_t = len(groups_t), len(subset)
    eps2_t = epsilon_squared_kruskal(H_t, k_t, n_t)
    save_text(REPORTS_DIR / "aircraft_type_kruskal.txt",
              f"Kruskal–Wallis across common types (>=30 accidents): H={H_t:.2f}, p={p_t:.4g}, epsilon^2={eps2_t:.3f}, k={k_t}, n={n_t}")
else:
    save_text(REPORTS_DIR / "aircraft_type_kruskal.txt",
              "Not enough common aircraft types (need ≥3 with >=30 accidents each).")

# Operators: normalize by years active in the window (distinct years present)
op_years = df.groupby("operator")["year"].nunique().rename("years_active")
op_counts = df.groupby("operator").agg(
    accidents=("date","count"),
    fatal_accidents=("is_fatal","sum"),
    fatalities_total=("fatalities","sum")
)
op_tbl = pd.concat([op_counts, op_years], axis=1).fillna(0)
op_tbl["accidents_per_year_active"] = op_tbl["accidents"] / op_tbl["years_active"].replace({0: np.nan})
op_tbl["fatal_per_year_active"] = op_tbl["fatal_accidents"] / op_tbl["years_active"].replace({0: np.nan})
op_tbl = op_tbl.sort_values(["accidents_per_year_active","accidents"], ascending=False)
op_tbl.to_csv(REPORTS_DIR / "operator_normalized_counts.csv")

# Operator × fatal chi-square (well-represented only)
top_ops = df["operator"].value_counts()
eligible_ops = top_ops[top_ops >= 40].index
df_ops = df[df["operator"].isin(eligible_ops)]
if df_ops["operator"].nunique() >= 2:
    ct_op = pd.crosstab(df_ops["operator"], df_ops["is_fatal"])
    chi2, p, dof, exp = stats.chi2_contingency(ct_op)
    n = ct_op.values.sum()
    r, c = ct_op.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r,c) > 1 else np.nan
    save_text(REPORTS_DIR / "operator_chi2.txt",
              f"Operator × Fatal chi-square on operators with >=40 accidents:\nchi2={chi2:.2f}, p={p:.4g}, dof={dof}, Cramer's V={cramers_v:.3f}")
else:
    save_text(REPORTS_DIR / "operator_chi2.txt",
              "Not enough operators with >=40 accidents for chi-square stability.")

# ---------- Snapshot for teammates ----------
df.to_csv(REPORTS_DIR / "dataset_clean_snapshot.csv", index=False)

# ---------- Friendly summary in terminal ----------
print("\nDone. Files written to ./reports")
produced = sorted([p.name for p in REPORTS_DIR.glob("*")])
for name in produced:
    print("-", name)

print("\nMapping to RQs:")
print("RQ1 → annual_summary.csv, trend_glm.txt")
print("RQ2 → decade_summary.csv, decade_severity_kruskal.txt, (optional) decade_pairwise_tests.csv")
print("RQ3 → aircraft_type_counts.csv, aircraft_type_kruskal.txt, operator_normalized_counts.csv, operator_chi2.txt")
