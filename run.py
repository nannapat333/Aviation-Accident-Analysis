# ============================================================
# Fatal vs Non-Fatal Classification & Clustering (Latest 10y)
# Models:  Random Forest (Calibrated), MLP (oversampled), GaussianNB
# CV:      5-fold Stratified OOF
# Metrics: Accuracy, Precision, RMSE(√Brier), ROC-AUC (+ ROC/PR plots)
# Extras:  Threshold sweep (best F1) + optimal CMs
# Viz:     CMs, feature importance, decision maps (PCA-2D)
# Cluster: KMeans & Agglomerative (k=2) + Silhouette, ARI, NMI (+ PCA/t-SNE)
# Outputs: ./outputs_latest10y/*
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.base import clone

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    f1_score
)
from sklearn.inspection import permutation_importance

# ------------------------ Config ------------------------
DATA_PATH = Path("clean.csv")
OUT_DIR   = Path("./outputs_latest10y")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS     = 5
TOP_K_COUNTRIES = 15  # bucket top-K countries, others -> "other"

# ------------------------ Load --------------------------
df = pd.read_csv(DATA_PATH)

# ------------------------ Keep / drop --------------------
# Drop year_orig (dup), registration (ID), fatalities (leakage)
keep_cols = ["date","year","decade","type","operator","location","country","cat","is_fatal"]
df = df[[c for c in keep_cols if c in df.columns]].copy()

# ------------------------ Date features ------------------
def safe_to_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce", utc=False)
    except Exception:
        return pd.Series(pd.NaT, index=s.index)

if "date" in df.columns:
    dt = safe_to_datetime(df["date"])
    df["month"]  = dt.dt.month
    df["season"] = pd.Series(np.select(
        [
            df["month"].isin([12,1,2]),
            df["month"].isin([3,4,5]),
            df["month"].isin([6,7,8]),
            df["month"].isin([9,10,11]),
        ],
        [1,2,3,4],
        default=0
    ), index=df.index).astype("Int64")
else:
    df["month"] = np.nan
    df["season"] = np.nan

# Ensure year exists; else derive from date
if "year" not in df.columns:
    df["year"] = safe_to_datetime(df["date"]).dt.year

# Latest 10 years scope
yr = pd.to_numeric(df["year"], errors="coerce")
if yr.notna().any():
    max_year = int(yr.max()); min_year = max_year - 9
    df = df[(yr >= min_year) & (yr <= max_year)].copy()
    print(f"[Scope] Using last 10 years: {min_year}–{max_year} (rows={len(df)})")
else:
    print("[Scope] ⚠️ No usable year; using full dataset.")

# ------------------------ Target -------------------------
if "is_fatal" not in df.columns:
    raise ValueError("Expected a binary target column 'is_fatal'.")

y_raw = df["is_fatal"]
if y_raw.dtype == "O":
    lut = {'yes':1,'no':0,'true':1,'false':0,'fatal':1,'non-fatal':0,'y':1,'n':0,'1':1,'0':0}
    y = y_raw.astype(str).str.lower().map(lut)
    if y.isnull().any():
        try:
            y = y_raw.astype(int)
        except:
            uniq = y_raw.dropna().unique().tolist()
            if len(uniq) != 2:
                raise ValueError("'is_fatal' must be binary.")
            y = y_raw.map({uniq[0]:0, uniq[1]:1})
else:
    y = y_raw.astype(int)

# Harden target
y = pd.Series(y).astype(int)
assert not y.isna().any(), "y has NA after coercion"
y.index = df.index  # keep index aligned with X later

# ------------------------ Text cleaning / bucketing -------
def clean_text(s: pd.Series):
    return s.astype(str).str.strip().str.lower().replace({"nan": np.nan})

def bucket_operator(op: pd.Series) -> pd.Series:
    s = clean_text(op).fillna("")
    out = []
    for v in s:
        if any(k in v for k in ["air force","navy","army","military"]): out.append("military")
        elif any(k in v for k in ["government","govt","minister","police","coast guard"]): out.append("government")
        elif any(k in v for k in ["cargo","freight","express","logistics"]): out.append("cargo")
        elif any(k in v for k in ["airways","airlines","air lines","avia","aero","jet","fly","lineas"]): out.append("airline")
        elif any(k in v for k in ["school","academy","training","flight school"]): out.append("training")
        elif any(k in v for k in ["club","private","charter","aeroclub"]): out.append("general_aviation")
        else: out.append("other")
    return pd.Series(out, index=op.index, dtype="object")

def bucket_country(cty: pd.Series, top_k=TOP_K_COUNTRIES) -> pd.Series:
    s = clean_text(cty)
    counts = Counter(s.dropna())
    top = set([c for c,_ in counts.most_common(top_k)])
    return s.apply(lambda x: x if pd.isna(x) or x in top else "other")

def clean_category(cat: pd.Series) -> pd.Series:
    s = clean_text(cat)
    if s.isna().all(): return s
    def norm(v):
        if pd.isna(v): return v
        if "collision" in v or "midair" in v: return "collision"
        if "loss of control" in v or "loc" in v: return "loss_of_control"
        if "runway" in v or "veer" in v: return "runway_excursion"
        if "cfit" in v or "terrain" in v or "impact" in v: return "cfit/terrain"
        if "engine" in v or "power" in v or "stall" in v: return "engine/stall"
        if "weather" in v or "imc" in v or "icing" in v: return "weather"
        return v
    return s.apply(norm)

def clean_type(typ: pd.Series) -> pd.Series:
    s = clean_text(typ)
    def norm(v):
        if pd.isna(v): return v
        if "accident" in v: return "accident"
        if "incident" in v: return "incident"
        if "hijack" in v: return "hijacking"
        return v
    return s.apply(norm)

df["operator_bucket"] = bucket_operator(df["operator"]) if "operator" in df.columns else np.nan
df["country_bucket"]  = bucket_country(df["country"])   if "country"  in df.columns else np.nan
df["cat_clean"]       = clean_category(df["cat"])       if "cat"      in df.columns else np.nan
df["type_clean"]      = clean_type(df["type"])          if "type"     in df.columns else np.nan

# ------------------------ Features -----------------------
drop_cols    = ["is_fatal","registration","fatalities","operator","country","cat","type"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()

# Leakage checks
assert "fatalities" not in X.columns
assert "is_fatal" not in X.columns

# Numeric / categorical split
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ------------------------ Preprocessing ------------------
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler(with_mean=True))  # center numerics (helps MLP)
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("oh",     ohe)
])
pre = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
], remainder="drop")

# ------------------------ Models -------------------------
rf_base = RandomForestClassifier(
    n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE
)
rf_cal  = CalibratedClassifierCV(rf_base, cv=3, method="isotonic")  # calibrated probabilities

models = {
    "Random Forest (Calibrated)": rf_cal,
    "Neural Net (MLP)": MLPClassifier(
        hidden_layer_sizes=(64,), early_stopping=True, n_iter_no_change=15,
        max_iter=600, learning_rate_init=1e-3, random_state=RANDOM_STATE
    ),
    "Naive Bayes (Gaussian)": GaussianNB(var_smoothing=1e-8)
}

def rmse_from_prob(y_true, y_prob):
    return float(np.sqrt(np.mean((y_prob - y_true)**2)))  # √Brier

def best_threshold(y_true, y_prob, metric="f1"):
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_val = 0.5, -1
    for t in ts:
        pred = (y_prob >= t).astype(int)
        val = f1_score(y_true, pred, zero_division=0) if metric=="f1" else t
        if val > best_val:
            best_val, best_t = val, t
    return best_t

# ------------------------ Oversamplers -------------------
rng = np.random.RandomState(RANDOM_STATE)

# (1) pandas version (used in CV loops on original X/y)
def oversample_minority(Xtr: pd.DataFrame, ytr: pd.Series):
    ytr = pd.Series(ytr).astype(int)
    vc = ytr.value_counts()
    if len(vc) != 2:
        return Xtr, ytr
    maj = vc.idxmax(); minc = vc.idxmin()
    n_maj, n_min = vc[maj], vc[minc]
    if n_min == 0 or n_maj == 0 or n_min == n_maj:
        return Xtr, ytr
    idx_min = ytr[ytr == minc].index.values
    add_idx = rng.choice(idx_min, size=n_maj - n_min, replace=True)
    X_os = pd.concat([Xtr, Xtr.loc[add_idx]], axis=0)
    y_os = pd.concat([ytr, ytr.loc[add_idx]], axis=0)
    return X_os, y_os

# (2) NumPy version (used in PCA decision-surface viz)
def oversample_minority_arrays(Xa: np.ndarray, ya: np.ndarray):
    ya = np.asarray(ya).astype(int)
    if ya.ndim != 1: ya = ya.ravel()
    # replace any NaN with mode (just in case)
    if np.isnan(ya.astype(float)).any():
        vals, counts = np.unique(ya[~np.isnan(ya.astype(float))], return_counts=True)
        mode_val = vals[np.argmax(counts)]
        ya = np.where(np.isnan(ya.astype(float)), mode_val, ya).astype(int)

    n0, n1 = (ya == 0).sum(), (ya == 1).sum()
    if n0 == 0 or n1 == 0 or n0 == n1:
        return Xa, ya
    maj = 1 if n1 > n0 else 0
    minc = 1 - maj
    idx_min = np.where(ya == minc)[0]
    add = rng.choice(idx_min, size=abs(n1 - n0), replace=True)
    Xa_os = np.vstack([Xa, Xa[add]])
    ya_os = np.concatenate([ya, ya[add]])
    return Xa_os, ya_os

# ------------------------ 5-fold OOF ---------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results, confusions = [], []
roc_data, pr_data   = {}, {}
oof_store = {}  # name -> dict(prob, pred@0.5)

for name, base_clf in models.items():
    oof_prob = np.zeros(len(X), dtype=float)
    oof_pred = np.zeros(len(X), dtype=int)

    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr      = y.iloc[tr]

        clf  = clone(base_clf)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Oversample ONLY for MLP
        if "Neural Net (MLP)" in name:
            Xtr_bal, ytr_bal = oversample_minority(Xtr, ytr)
            pipe.fit(Xtr_bal, ytr_bal)
        else:
            pipe.fit(Xtr, ytr)

        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(Xte)[:, 1]
        elif hasattr(pipe, "decision_function"):
            s = pipe.decision_function(Xte)
            p = (s - s.min()) / (s.max() - s.min() + 1e-9)
        else:
            p = pipe.predict(Xte).astype(float)

        oof_prob[te] = p
        oof_pred[te] = (p >= 0.5).astype(int)

    # Metrics @ 0.5
    acc  = accuracy_score(y, oof_pred)
    prec = precision_score(y, oof_pred, zero_division=0)
    rmse = rmse_from_prob(y, oof_prob)
    fpr, tpr, _ = roc_curve(y, oof_prob)
    model_auc   = auc(fpr, tpr)
    rec, pre_curve, _ = precision_recall_curve(y, oof_prob)
    ap = average_precision_score(y, oof_prob)

    cm = confusion_matrix(y, oof_pred); tn, fp, fn, tp = cm.ravel()

    results.append({"model": name, "accuracy": acc, "precision": prec, "rmse_prob": rmse, "auc": model_auc})
    confusions.append({"model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": model_auc}
    pr_data[name]  = {"recall": rec, "precision": pre_curve, "ap": ap}
    oof_store[name]= {"prob": oof_prob, "pred": oof_pred}

eval_df = pd.DataFrame(results).sort_values(["rmse_prob","accuracy"], ascending=[True, False]).reset_index(drop=True)
cms_df  = pd.DataFrame(confusions)
eval_df.to_csv(OUT_DIR/"classification_metrics_oof.csv", index=False)
cms_df.to_csv(OUT_DIR/"confusion_matrices_oof.csv", index=False)

print("\n=== Classification (5-fold OOF @ 0.5) ===")
print(eval_df.round(4).to_string(index=False))
print("\nConfusion matrices (OOF @ 0.5):")
print(cms_df.to_string(index=False))

# ------------------------ ROC & PR -----------------------
plt.figure(figsize=(8,6))
for name, d in roc_data.items():
    plt.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.3f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (5-fold OOF)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUT_DIR/"roc_curves_oof.png", dpi=180); plt.close()

plt.figure(figsize=(8,6))
for name, d in pr_data.items():
    plt.plot(d["recall"], d["precision"], label=f"{name} (AP={d['ap']:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves (5-fold OOF)")
plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(OUT_DIR/"pr_curves_oof.png", dpi=180); plt.close()

# ------------------------ CM images @ 0.5 ----------------
for name, store in oof_store.items():
    cm = confusion_matrix(y, store["pred"])
    fig, ax = plt.subplots(figsize=(4.6,4.6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    ax.set_title(f"Confusion Matrix (OOF @ 0.5) — {name}")
    plt.tight_layout()
    fig.savefig(OUT_DIR/f"cm_{name.replace(' ','_').lower()}_05.png", dpi=180)
    plt.close(fig)

# ------------------------ Threshold sweep (best F1) ------
for name, store in oof_store.items():
    prob = store["prob"]
    t_opt = best_threshold(y, prob, metric="f1")
    pred_opt = (prob >= t_opt).astype(int)
    cm_opt = confusion_matrix(y, pred_opt)
    tn, fp, fn, tp = cm_opt.ravel()
    acc  = accuracy_score(y, pred_opt)
    prec = precision_score(y, pred_opt, zero_division=0)
    f1v  = f1_score(y, pred_opt, zero_division=0)
    print(f"\n{name}: optimal threshold (F1) ≈ {t_opt:.2f} | Acc={acc:.3f} Prec={prec:.3f} F1={f1v:.3f} CM=[TN={tn} FP={fp} FN={fn} TP={tp}]")
    fig, ax = plt.subplots(figsize=(4.6,4.6))
    ConfusionMatrixDisplay(confusion_matrix=cm_opt).plot(ax=ax)
    ax.set_title(f"Confusion Matrix (OOF @ t*={t_opt:.2f}) — {name}")
    plt.tight_layout()
    fig.savefig(OUT_DIR/f"cm_{name.replace(' ','_').lower()}_opt.png", dpi=180)
    plt.close(fig)

# ------------------------ Feature visuals (version-safe) ----------------
pre.fit(X)

def get_transformed_feature_names(pre: ColumnTransformer) -> np.ndarray:
    names = []
    if num_cols:
        names += list(num_cols)
    if cat_cols:
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        names += oh.get_feature_names_out(cat_cols).tolist()
    return np.array(names)

tfeat_names = get_transformed_feature_names(pre)
orig_feat_names = np.array(X.columns.tolist())

# 1) RANDOM FOREST: fit a PLAIN forest (not calibrated) only to extract importances
rf_plain = RandomForestClassifier(
    n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE
)
pipe_rf_plain = Pipeline([("pre", pre), ("clf", rf_plain)])
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
pipe_rf_plain.fit(Xtr, ytr)
imp = pipe_rf_plain.named_steps["clf"].feature_importances_
idx = np.argsort(imp)[-15:]
fig, ax = plt.subplots(figsize=(7,6))
ax.barh(tfeat_names[idx], imp[idx])
ax.set_title("Top 15 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(OUT_DIR/"importance_random_forest.png", dpi=180)
plt.close(fig)

# 2) MLP & NAIVE BAYES: permutation importance on the full Pipeline
other_models = {
    "Neural Net (MLP)": MLPClassifier(
        hidden_layer_sizes=(64,), early_stopping=True, n_iter_no_change=15,
        max_iter=600, learning_rate_init=1e-3, random_state=RANDOM_STATE
    ),
    "Naive Bayes (Gaussian)": GaussianNB(var_smoothing=1e-8)
}
for name, base_clf in other_models.items():
    pipe_model = Pipeline([("pre", pre), ("clf", clone(base_clf))])
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    if "MLP" in name:
        Xtr, ytr = oversample_minority(Xtr, ytr)
    pipe_model.fit(Xtr, ytr)
    pi = permutation_importance(
        pipe_model, Xva, yva, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    pim = pi.importances_mean
    idx = np.argsort(pim)[-15:]
    fig, ax = plt.subplots(figsize=(7,6))
    ax.barh(orig_feat_names[idx], pim[idx])
    ax.set_title(f"Top 15 Permutation Importances — {name}")
    ax.set_xlabel("Importance (mean ΔAUC)")
    plt.tight_layout()
    fig.savefig(OUT_DIR/f"importance_{name.lower().replace(' ','_')}.png", dpi=180)
    plt.close(fig)

# ------------------------ Decision maps (PCA-2D) ---------
Z = pre.fit_transform(X)
try: Z = Z.toarray()
except Exception: pass

pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X2   = pca2.fit_transform(Z)
X2_tr, X2_te, y_tr, y_te = train_test_split(X2, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

viz_models = {
    "Random Forest (PCA2)": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE),
    "MLP (PCA2)":            MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, n_iter_no_change=15, max_iter=600, random_state=RANDOM_STATE),
    "Naive Bayes (PCA2)":    GaussianNB(var_smoothing=1e-8)
}

def plot_decision_surface(X2_tr, y_tr, X2_te, y_te, clf, title, outfile):
    X2_tr = np.asarray(X2_tr); y_tr = np.asarray(y_tr).astype(int)
    X2_te = np.asarray(X2_te); y_te = np.asarray(y_te).astype(int)

    # oversample ONLY for MLP here (to avoid trivial majority fit)
    if isinstance(clf, MLPClassifier):
        X2_tr_os, y2_tr_os = oversample_minority_arrays(X2_tr, y_tr)
        clf.fit(X2_tr_os, y2_tr_os)
    else:
        clf.fit(X2_tr, y_tr)

    x_min, x_max = X2[:,0].min()-0.5, X2[:,0].max()+0.5
    y_min, y_max = X2[:,1].min()-0.5, X2[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(clf, "predict_proba"):
        zz = clf.predict_proba(grid)[:,1].reshape(xx.shape)
    elif hasattr(clf, "decision_function"):
        s  = clf.decision_function(grid)
        s  = (s - s.min())/(s.max()-s.min()+1e-9)
        zz = s.reshape(xx.shape)
    else:
        zz = clf.predict(grid).astype(float).reshape(xx.shape)

    yhat_te = clf.predict(X2_te)
    correct = yhat_te == y_te

    plt.figure(figsize=(7,6))
    cs = plt.contourf(xx, yy, zz, levels=20, alpha=0.6)
    plt.colorbar(cs, label="P(class=1)")
    plt.scatter(X2_te[correct,0], X2_te[correct,1], s=20, marker='o', label="Correct", alpha=0.9)
    plt.scatter(X2_te[~correct,0], X2_te[~correct,1], s=20, marker='x', label="Incorrect", alpha=0.9)
    plt.title(title); plt.xlabel("PCA-1"); plt.ylabel("PCA-2"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR/outfile, dpi=180); plt.close()

for name, clf in viz_models.items():
    safe = name.lower().replace(" ","_").replace("(","").replace(")","")
    plot_decision_surface(X2_tr, y_tr, X2_te, y_te, clf, f"Decision Map — {name}", f"decision_map_{safe}.png")

# ------------------------ Clustering ---------------------
rows = []
km  = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
agg = AgglomerativeClustering(n_clusters=2, linkage="ward")

lab_km  = km.fit_predict(Z)
lab_agg = agg.fit_predict(Z)

sil_km  = silhouette_score(Z, lab_km)  if len(np.unique(lab_km)) > 1  else np.nan
sil_agg = silhouette_score(Z, lab_agg) if len(np.unique(lab_agg)) > 1 else np.nan
ari_km  = adjusted_rand_score(y, lab_km)
ari_agg = adjusted_rand_score(y, lab_agg)
nmi_km  = normalized_mutual_info_score(y, lab_km)
nmi_agg = normalized_mutual_info_score(y, lab_agg)

rows.append({"algorithm":"KMeans (k=2)","silhouette":round(sil_km,4),"ARI_vs_is_fatal":round(ari_km,4),"NMI_vs_is_fatal":round(nmi_km,4)})
rows.append({"algorithm":"Agglomerative (Ward, k=2)","silhouette":round(sil_agg,4),"ARI_vs_is_fatal":round(ari_agg,4),"NMI_vs_is_fatal":round(nmi_agg,4)})
cluster_df = pd.DataFrame(rows)
cluster_df.to_csv(OUT_DIR/"clustering_metrics.csv", index=False)

def scatter_clusters(X2, labels, title, outfile):
    plt.figure(figsize=(7,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=8)
    plt.title(title); plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
    plt.tight_layout(); plt.savefig(OUT_DIR/outfile, dpi=180); plt.close()

# PCA & t-SNE plots
scatter_clusters(X2, lab_km,  "KMeans clusters (PCA-2D)", "kmeans_pca.png")
scatter_clusters(X2, lab_agg, "Agglomerative clusters (PCA-2D)", "agg_pca.png")

try:
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto", perplexity=30)
    X2_tsne = tsne.fit_transform(Z)
    scatter_clusters(X2_tsne, lab_km,  "KMeans clusters (t-SNE 2D)", "kmeans_tsne.png")
    scatter_clusters(X2_tsne, lab_agg, "Agglomerative clusters (t-SNE 2D)", "agg_tsne.png")
except Exception as e:
    print(f"[t-SNE] Skipped due to: {e}")

# True label PCA scatter
scatter_clusters(X2, y.values, "True labels (PCA-2D)", "true_pca.png")

# Clustering table figure
fig, ax = plt.subplots(figsize=(7, 2 + 0.5*len(cluster_df)))
ax.axis("off")
tbl = ax.table(cellText=cluster_df.values, colLabels=cluster_df.columns, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.2)
plt.title("Clustering Metrics (k=2)")
plt.tight_layout(); plt.savefig(OUT_DIR/"clustering_table.png", dpi=180); plt.close()

# ------------------------ Done ---------------------------
print(f"\nArtifacts saved to: {OUT_DIR.resolve()}")
print(" - classification_metrics_oof.csv, confusion_matrices_oof.csv")
print(" - roc_curves_oof.png, pr_curves_oof.png")
print(" - cm_*_05.png (threshold=0.5), cm_*_opt.png (best F1)")
print(" - importance_*.png")
print(" - decision_map_random_forest_pca2.png, decision_map_mlp_pca2.png, decision_map_naive_bayes_pca2.png")
print(" - clustering_metrics.csv, clustering_table.png")
print(" - kmeans_pca.png, kmeans_tsne.png (if available), agg_pca.png, agg_tsne.png (if available), true_pca.png")