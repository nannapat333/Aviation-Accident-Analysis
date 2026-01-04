# visual.py
# Streamlit dashboard (clear & simple) for Aviation Accidents
# Works with the CSVs produced by the updated analysis.py

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- Config ----------------
REPORTS_DIR = Path("reports")
st.set_page_config(page_title="Aviation Safety Dashboard", page_icon="✈️", layout="wide")

ORANGE = "#F4A261"
BLACK  = "#000000"
GREEN  = "#2A9D8F"
RED    = "#E76F51"
TEMPLATE = "plotly_white"

st.markdown("""
<style>
.kpi-card { background:#fff; border:1px solid #eee; border-radius:16px; padding:16px 18px; box-shadow:0 2px 12px rgba(0,0,0,.06); }
.kpi-label { font-size:.85rem; color:#666; margin-bottom:6px; }
.kpi-value { font-size:1.6rem; font-weight:700; margin-bottom:0; }
.kpi-sub { font-size:.8rem; color:#888; }
.section-title { font-size:1.05rem; font-weight:700; margin-top:8px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# -------------- Load helpers --------------
def load_csv(name: str) -> pd.DataFrame | None:
    p = REPORTS_DIR / name
    if not p.exists():
        st.warning(f"Missing file: {p}")
        return None
    return pd.read_csv(p)

# -------------- Load data --------------
clean   = load_csv("dataset_clean_snapshot.csv")       # row-level, 1950+
trend_y = load_csv("trend_year.csv")                   # year, accidents, fatal_accidents
trend_d = load_csv("trend_decade.csv")                 # decade, accidents, fatal_accidents
sev_d   = load_csv("severity_decade_summary.csv")      # decade, median   <-- note: ONLY median
types_r = load_csv("aircraft_types_recent.csv")        # type, accidents  (RECENT_FROM+)
ops_r   = load_csv("operators_recent.csv")             # operator, accidents, category (no Military/Unknown)

req = [clean, trend_y, trend_d, sev_d, types_r, ops_r]
if any(x is None for x in req):
    st.error("Missing required CSVs in ./reports. Please run:  python analysis.py")
    st.stop()

clean["date"] = pd.to_datetime(clean["date"], errors="coerce")

# -------------- Header --------------
st.title("✈️ Aviation Safety Dashboard (1950+)")
st.caption("Trends and comparisons focused on clarity. Orange = accidents; Black = fatal accidents.")

# -------------- Filters --------------
with st.expander("🔎 Filters", expanded=True):
    min_year, max_year = int(clean["year"].min()), int(clean["year"].max())
    yr1, yr2 = st.slider("Year range for Trend & Severity", min_year, max_year, (min_year, max_year), step=1)

    c1, c2 = st.columns(2)
    with c1:
        scoped_from = st.number_input("Use data from year (for Aircraft Types & Airlines sections)",
                                      min_value=min_year, max_value=max_year, value=2017, step=1)
    with c2:
        top_n = st.slider("Top-N for Aircraft Types & Airlines", 5, 30, 10, step=1)

# -------------- KPI cards --------------
clean_f = clean[clean["year"].between(yr1, yr2)].copy()
total_acc = int(len(clean_f))
fatal_pct = float(clean_f["is_fatal"].mean()) if total_acc else 0.0
span = f"{yr1}–{yr2}"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Accidents (filtered)</div>'
                f'<div class="kpi-value">{total_acc:,}</div><div class="kpi-sub">Span: {span}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">% Fatal Accidents (filtered)</div>'
                f'<div class="kpi-value">{fatal_pct*100:.1f}%</div><div class="kpi-sub">Fatal / All accidents</div></div>', unsafe_allow_html=True)
lead_type = "—"; lead_op = "—"  # filled after charts

st.markdown("---")

# -------------- 1) Trend: Yearly / Decade --------------
left, right = st.columns([3,1])
with left:
    st.subheader("Accidents over Time")
    gran = st.radio("Granularity", ["Yearly", "Decade"], horizontal=True, label_visibility="collapsed")
    if gran == "Yearly":
        t = trend_y[(trend_y["year"]>=yr1) & (trend_y["year"]<=yr2)].copy()
        if t.empty:
            st.info("No data for selected years.")
        else:
            dfm = t.melt(id_vars="year", value_vars=["accidents","fatal_accidents"], var_name="Series", value_name="Count")
            fig = px.line(dfm, x="year", y="Count", color="Series", markers=True,
                          color_discrete_map={"accidents": ORANGE, "fatal_accidents": BLACK},
                          template=TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
    else:
        d1, d2 = (yr1 // 10) * 10, (yr2 // 10) * 10
        td = trend_d[(trend_d["decade"]>=d1) & (trend_d["decade"]<=d2)].copy()
        if td.empty:
            st.info("No data for selected decades.")
        else:
            dfm = td.melt(id_vars="decade", value_vars=["accidents","fatal_accidents"], var_name="Series", value_name="Count")
            fig = px.line(dfm, x="decade", y="Count", color="Series", markers=True,
                          color_discrete_map={"accidents": ORANGE, "fatal_accidents": BLACK},
                          template=TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown('<div class="section-title">What to notice</div>', unsafe_allow_html=True)
    st.write("- Long-term **decline** in total and fatal accidents.")
    st.write("- Modern aviation is **safer** than early decades.")

st.markdown("---")

# -------------- 2) Severity (Median per Decade) --------------
st.subheader("Severity per Accident by Decade (Median fatalities)")
d1, d2 = (yr1 // 10) * 10, (yr2 // 10) * 10
sev_show = sev_d[(sev_d["decade"]>=d1) & (sev_d["decade"]<=d2)].copy()
if sev_show.empty:
    st.info("No data for selected decades.")
else:
    fig = px.line(sev_show, x="decade", y="median", markers=True,
                  color_discrete_sequence=[ORANGE], template=TEMPLATE,
                  labels={"median":"Median fatalities per accident"})
    fig.update_layout(margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Most accidents in every decade are low-fatality; the **median** has fallen over time → safer & more survivable events.")

st.markdown("---")

# -------------- 3) Aircraft Types: Top-N counts --------------
st.subheader(f"Accidents by Aircraft Type (Top {top_n}, {scoped_from}+)")
# Recompute live from clean_f to respect year filter + scope
t_recent = clean[(clean["year"] >= scoped_from)]
type_counts = (t_recent.groupby("type")
               .agg(accidents=("date","count"))
               .reset_index())
# tidy up
type_counts = type_counts[type_counts["type"].notna()]
type_counts = type_counts[type_counts["type"] != "Unknown"]
type_counts = type_counts.sort_values("accidents", ascending=False).head(top_n)

if type_counts.empty:
    st.info("No aircraft data for this scope.")
else:
    fig = px.bar(type_counts.sort_values("accidents"),
                 x="accidents", y="type", orientation="h",
                 color_discrete_sequence=[ORANGE], template=TEMPLATE,
                 labels={"accidents":"Accident Count","type":"Aircraft Type"})
    st.plotly_chart(fig, use_container_width=True)
    lead_type = type_counts.iloc[0]["type"]

st.markdown("---")

# -------------- 4) Airlines/Operators: Top-N + Proportion --------------
st.subheader(f"Accidents by Airline/Operator (Top {top_n}, {scoped_from}+; Excluding Military & Unknown)")

# categorize locally to be robust
def categorize_operator(op: str) -> str:
    s = str(op).lower()
    if any(k in s for k in ["air force"," af "," af-","af/"," navy","naval"]): return "Military"
    if "cargo" in s or "freight" in s or "logistics" in s: return "Cargo"
    if "private" in s or "jet" in s or "charter" in s or "bizjet" in s: return "Private"
    if "air" in s or "airlines" in s or "airways" in s: return "Commercial"
    return "Unknown"

ops_live = clean[(clean["year"] >= scoped_from)].copy()
ops_live["category"] = ops_live["operator"].apply(categorize_operator)
ops_live = ops_live[~ops_live["category"].isin(["Military","Unknown"])]

ops_counts = (ops_live.groupby(["operator","category"])
              .agg(accidents=("date","count"))
              .reset_index()
              .sort_values("accidents", ascending=False)
              .head(top_n))

colA, colB = st.columns([2,1])

with colA:
    if ops_counts.empty:
        st.info("No operator data for this scope.")
    else:
        fig = px.bar(ops_counts.sort_values("accidents"),
                     x="accidents", y="operator", orientation="h",
                     color="category",
                     color_discrete_map={"Commercial": ORANGE, "Cargo": GREEN, "Private": RED},
                     template=TEMPLATE,
                     labels={"accidents":"Accident Count","operator":"Airline / Operator","category":"Type"})
        st.plotly_chart(fig, use_container_width=True)
        lead_op = ops_counts.iloc[0]["operator"]
        st.caption("Raw counts in the selected window. Exposure matters (major carriers fly more), but most accidents tend to be in **Private** operations, not Commercial.")

with colB:
    # Proportion chart by operator category to emphasize 'Private' vs 'Commercial'
    ops_cats = (ops_live.groupby("category")["date"]
                .count()
                .reset_index()
                .rename(columns={"date":"accidents"}))
    if ops_cats.empty:
        st.info("No category data.")
    else:
        figp = px.pie(ops_cats, values="accidents", names="category",
                      color="category",
                      color_discrete_map={"Commercial": ORANGE, "Cargo": GREEN, "Private": RED},
                      template=TEMPLATE, hole=0.35)
        st.plotly_chart(figp, use_container_width=True)
        st.caption("Share of accidents by operator category (excludes Military/Unknown). **Commercial** is a comparatively small slice.")

# -------------- Fill KPI leaders --------------
k3 = st.empty(); k4 = st.empty()
k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Top Aircraft Type (by accidents)</div>'
            f'<div class="kpi-value">{lead_type}</div><div class="kpi-sub">Scope: {scoped_from}+; Top {top_n}</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Top Operator (by accidents)</div>'
            f'<div class="kpi-value">{lead_op}</div><div class="kpi-sub">Scope: {scoped_from}+; Top {top_n}</div></div>', unsafe_allow_html=True)

# -------------- Footer --------------
st.markdown("---")
st.caption("Design choices: severity shown as median line (clear trend); aircraft & airlines use total accidents; airlines exclude Military/Unknown and include a proportion view to show category mix.")






