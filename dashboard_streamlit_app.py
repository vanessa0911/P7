# Streamlit Credit Scoring Dashboard — "Prêt à dépenser" (v0.7.3)
# ----------------------------------------------------------------
# Run:
#   python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
#
# Root files auto-detected (first found wins):
# - Data:  application_train_clean.csv  |  clients_demo.csv  |  clients_demo.parquet
# - Model: model_calibrated_isotonic.joblib | model_calibrated_sigmoid.joblib | model_baseline_logreg.joblib
# - Features (optional): feature_names.npy
# - Global importance (optional): global_importance.csv  (columns: feature, importance)
# - Interpretability (optional): interpretability_summary.json

APP_VERSION = "0.7.3"

import os
import json
import subprocess
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional, Tuple

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

st.set_page_config(page_title="Prêt à dépenser — Credit Scoring", page_icon="💳", layout="wide")

# -------------------------------
# Runtime diagnostics (to verify file actually running)
# -------------------------------
def _runtime_info():
    try:
        path = os.path.abspath(__file__)
    except NameError:
        path = "(unknown)"
    try:
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        mtime_str = "n/a"
    try:
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:8]
    except Exception:
        sha = "n/a"
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git = "n/a"
    return path, mtime_str, sha, git

# -------------------------------
# Helpers
# -------------------------------
def _pick_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def safe_load_model(path: str):
    """Loader avec message clair si une dépendance manque (ex. catboost)."""
    try:
        return load_model(path)
    except ModuleNotFoundError as e:
        st.error(f"Le modèle nécessite le paquet manquant: `{e.name}`. Installez-le avec `pip install {e.name}` puis relancez l'app.")
        raise
    except Exception as e:
        st.error(f"Échec du chargement du modèle `{os.path.basename(path)}`: {e}")
        raise

@st.cache_data(show_spinner=False)
def load_feature_names(path: Optional[str], df_cols: List[str]) -> List[str]:
    if path and os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        names = list(arr.tolist())
        return [c for c in names if c in df_cols]
    # fallback: toutes les colonnes brutes sauf ID/Target
    return [c for c in df_cols if c not in {"TARGET", "SK_ID_CURR"}]

@st.cache_data(show_spinner=False)
def load_global_importance(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        fcol = cols.get("feature") or cols.get("features") or list(df.columns)[0]
        icol = cols.get("importance") or cols.get("importance_mean") or cols.get("importance_mean_abs") or list(df.columns)[1]
        out = df[[fcol, icol]].copy()
        out.columns = ["feature", "importance"]
        return out.sort_values("importance", ascending=False)
    return None

@st.cache_data(show_spinner=False)
def load_interpretability_summary(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_expected_input_columns(model) -> Optional[List[str]]:
    """Colonnes d'entrée (brutes) attendues par le préprocesseur du modèle."""
    try:
        m = model
        if isinstance(m, SkPipeline):
            for _, step in m.steps:
                if isinstance(step, SkColumnTransformer) and hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
    except Exception:
        pass
    return None

def compute_local_shap(estimator, X_background: pd.DataFrame, x_row: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explication locale robuste, modèle-agnostique, sur la pipeline complète.
    Utilise shap.Explainer + masker indépendant sur la fonction predict_proba.
    """
    import shap, pandas as pd, numpy as np

    # limiter le fond pour garder de bonnes perfs
    bg = X_background
    if len(bg) > 200:
        bg = bg.sample(200, random_state=42)

    def f(Xdf):
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf, columns=list(bg.columns))
        return estimator.predict_proba(Xdf)[:, 1]

    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(f, masker, feature_names=list(bg.columns))
    ex = explainer(x_row)
    return np.array(ex.values).reshape(-1), np.array(ex.base_values).reshape(-1)

# ---- quantiles robustes (évite les KeyError) ----
def get_quantile_series(feature: str, pool_df: pd.DataFrame, X: pd.DataFrame) -> Optional[pd.Series]:
    """
    Série utilisée pour les quantiles :
    - priorité au dataset brut pool_df si la colonne est présente et numérique
    - sinon bascule sur X (colonnes alignées au modèle)
    - sinon None
    """
    if feature in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[feature]):
        return pool_df[feature]
    if feature in X.columns and pd.api.types.is_numeric_dtype(X[feature]):
        return X[feature]
    return None

def get_cohort_series(feature: str, cohort_df: pd.DataFrame, X: pd.DataFrame, ID_COL: Optional[str]) -> Optional[pd.Series]:
    """
    Série pour la cohorte (mêmes règles que ci-dessus). Si la colonne n’est pas dans cohort_df,
    on la récupère via X en filtrant sur les IDs de la cohorte.
    """
    if feature in cohort_df.columns and pd.api.types.is_numeric_dtype(cohort_df[feature]):
        return cohort_df[feature]
    if ID_COL and ID_COL in cohort_df.columns and feature in X.columns:
        idx = cohort_df[ID_COL].values
        s = X.loc[X.index.intersection(idx), feature]
        return s if pd.api.types.is_numeric_dtype(s) else None
    return None

def prob_to_band(p: float, low=0.05, high=0.15) -> Tuple[str, str]:
    if p < low:
        return ("Faible", "#3CB371")
    if p < high:
        return ("Modérée", "#E6B800")
    return ("Élevée", "#E74C3C")

# ---- coût métier ----
def cost_at_threshold(y_true: np.ndarray, p: np.ndarray, t: float, cost_fp: float, cost_fn: float):
    """Calcule coût total + métriques au seuil t."""
    y_pred = (p >= t).astype(int)  # 1 = défaut prédit (refus)
    # Labels: 1 = défaut réel
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    cost = fp * cost_fp + fn * cost_fn
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return dict(cost=float(cost), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=float(prec), recall=float(rec), f1=float(f1))

def cost_curve(y_true: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float, step: float=0.001):
    """Balaye les seuils [0,1] et retourne DataFrame coût vs seuil + seuil optimal."""
    step = float(step)
    if step <= 0:
        step = 0.001
    ts = np.arange(0.0, 1.0 + step, step, dtype=float)
    rows = []
    for t in ts:
        m = cost_at_threshold(y_true, p, float(t), cost_fp, cost_fn)
        m["threshold"] = float(t)
        rows.append(m)
    df = pd.DataFrame(rows)
    # Robust min selection (no Series.idx_* calls)
    cost_arr = pd.to_numeric(df["cost"], errors="coerce").to_numpy()
    cost_arr = np.where(np.isfinite(cost_arr), cost_arr, np.inf)  # replace NaN by +inf
    best_pos = int(np.argmin(cost_arr))
    best = df.iloc[best_pos].to_dict()
    return df, best

# -------------------------------
# Locate artifacts at repo root (no folders required)
# -------------------------------
DATA_TRAIN = _pick_first_existing([
    "application_train_clean.csv",
    "clients_demo.csv",
    "clients_demo.parquet",
])
DATA_TEST  = _pick_first_existing(["application_test_clean.csv"])  # optional
MODEL_ISO  = _pick_first_existing(["model_calibrated_isotonic.joblib"])
MODEL_SIG  = _pick_first_existing(["model_calibrated_sigmoid.joblib"])
MODEL_BASE = _pick_first_existing(["model_baseline_logreg.joblib"])
FEATS_PATH = _pick_first_existing(["feature_names.npy"])
GLOBIMP    = _pick_first_existing(["global_importance.csv"])
INTERP_SUM = _pick_first_existing(["interpretability_summary.json"])

with st.sidebar:
    st.title("💳 Scoring Crédit — Dashboard")
    st.caption("Prêt à dépenser — transparence & explicabilité")

    # Diagnostics de version/fichier
    path, mtime_str, sha8, git = _runtime_info()
    st.caption(f"App version: {APP_VERSION}")
    st.caption(f"Fichier: {os.path.basename(path)}")
    st.caption(f"Dernière modif: {mtime_str}")
    st.caption(f"SHA fichier: {sha8} | Git: {git}")

    if st.button("🔄 Forcer rechargement (vider cache)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    if not DATA_TRAIN:
        st.error("⚠️ Données non trouvées (placez `clients_demo.csv` ou `clients_demo.parquet` à la racine)")

# Load datasets
train_df = load_table(DATA_TRAIN) if DATA_TRAIN else pd.DataFrame()
holdout_df = load_table(DATA_TEST) if DATA_TEST else pd.DataFrame()

# Prepare feature list
pool_df = train_df if not train_df.empty else holdout_df
feature_names = load_feature_names(FEATS_PATH, list(pool_df.columns)) if not pool_df.empty else []

# Models (lazy load later)
model_paths = {}
if MODEL_ISO: model_paths["Calibré (Isotonic)"] = MODEL_ISO
if MODEL_SIG: model_paths["Calibré (Sigmoid)"]  = MODEL_SIG
if MODEL_BASE: model_paths["Baseline"]          = MODEL_BASE

# Interpretability artifacts
global_imp_df = load_global_importance(GLOBIMP)
interp_summary = load_interpretability_summary(INTERP_SUM)

# IDs and target
ID_COL = "SK_ID_CURR" if (not pool_df.empty and "SK_ID_CURR" in pool_df.columns) else (pool_df.columns[0] if not pool_df.empty else None)
TARGET_COL = "TARGET" if (not pool_df.empty and "TARGET" in pool_df.columns) else None

# Sidebar controls (with session_state to allow programmatic update)
with st.sidebar:
    st.subheader("Paramètres du modèle (choix)")
    model_name = st.selectbox("Choisir le modèle", list(model_paths.keys()) if model_paths else ["—"])
    default_thresh = st.session_state.get("threshold", 0.08)
    threshold = st.slider("Seuil d'acceptation (proba défaut)", 0.0, 0.5, float(default_thresh), 0.005, key="threshold",
                          help="Au-delà du seuil = risque élevé ⇒ refus")
    st.divider()
    st.subheader("Sélection du client")
    id_options = pool_df[ID_COL].tolist() if (ID_COL and not pool_df.empty) else []
    selected_id = st.selectbox("SK_ID_CURR", id_options, index=0 if id_options else None)
    st.caption("Astuce : utilisez le champ de recherche pour filtrer par ID.")

# Load selected model lazily
model = None
if model_name in (model_paths or {}):
    try:
        model = safe_load_model(model_paths[model_name])
    except Exception:
        model = None

# Build X aligned to model expected columns
if not pool_df.empty and selected_id is not None:
    df_idx = pool_df.set_index(ID_COL)
    temp_expected = get_expected_input_columns(model) or feature_names or list(df_idx.columns)
    # Créer les colonnes manquantes à NaN et ordonner
    for c in temp_expected:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    X = df_idx[temp_expected]
    # x_row + background (limité pour SHAP)
    x_row = X.loc[[selected_id]]
    background = X.sample(min(200, len(X)), random_state=42)
else:
    X = pd.DataFrame(columns=feature_names)
    x_row = X.head(0)
    background = X

# Prediction
proba = None
if model is not None and not x_row.empty:
    proba = float(model.predict_proba(x_row)[0, 1])

# -------------------------------
# Tabs — single row (no duplicates)
# -------------------------------
TABS = [
    "📈 Score & explication",
    "🧑‍💼 Fiche client",
    "⚖️ Comparaison",
    "🌍 Insights globaux",
    "🧪 Qualité des données",
    "🆕 Nouveau client",
    "💰 Seuil & coût métier",
]
main_tabs = st.tabs(TABS)

# -------------------------------
# Tab 1 — Score & local explanation
# -------------------------------
with main_tabs[0]:
    st.subheader("Score individuel & interprétation")
    if proba is None:
        st.warning("Modèle ou données indisponibles pour calculer une probabilité.")
    else:
        def _band(p: float):
            if p < 0.05: return ("Faible", "#3CB371")
            if p < 0.15: return ("Modérée", "#E6B800")
            return ("Élevée", "#E74C3C")
        band, color = _band(proba)
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color},
                       "steps": [
                           {"range": [0, threshold * 100], "color": "#ecf8f3"},
                           {"range": [threshold * 100, 100], "color": "#fdecea"},
                       ]},
                title={"text": "Probabilité de défaut"},
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Décision (seuil {threshold:.3f})** : **{'Refus' if proba >= threshold else 'Accord'}**")
            st.markdown(f"Risque : **{band}**")

        with col2:
            st.markdown("**Contributions locales (SHAP)** — top 10")
            shap_enabled = st.toggle("Activer SHAP (expérimental)", value=False,
                                     help="Active l'explication locale. Peut être lent selon le modèle.")
            if shap_enabled and not background.empty and model is not None:
                try:
                    vals, base_vals = compute_local_shap(model, background, x_row)
                    local_df = pd.DataFrame({
                        "feature": list(X.columns),
                        "shap_value": vals,
                        "abs_val": np.abs(vals),
                        "value": x_row.iloc[0].values,
                    }).sort_values("abs_val", ascending=False).head(10)
                    bar = px.bar(local_df[::-1], x="shap_value", y="feature", orientation="h",
                                 hover_data={"value": True, "abs_val": False},
                                 title="Impact sur le score (positif = ↑ risque)")
                    st.plotly_chart(bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP indisponible: {e}")
            else:
                if global_imp_df is not None:
                    st.dataframe(global_imp_df.head(10))
                else:
                    st.info("Importance globale indisponible.")

# -------------------------------
# Tab 2 — Client sheet
# -------------------------------
with main_tabs[1]:
    st.subheader("Fiche client")
    if x_row.empty:
        st.info("Sélectionnez un client dans la barre latérale.")
    else:
        if global_imp_df is not None:
            key_feats = [f for f in global_imp_df.head(20)["feature"].tolist() if f in X.columns]
        else:
            key_feats = list(X.columns)[:20]
        pretty = x_row[key_feats].T.reset_index()
        pretty.columns = ["Variable", "Valeur"]
        st.dataframe(pretty, use_container_width=True)

# -------------------------------
# Tab 3 — Comparison
# -------------------------------
with main_tabs[2]:
    st.subheader("Comparaison du client")
    if X.empty:
        st.info("Données indisponibles pour la comparaison.")
    else:
        st.markdown("**Définir le groupe de comparaison**")
        candidate_cohorts = [c for c in [
            "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE",
            "REGION_RATING_CLIENT", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"
        ] if c in pool_df.columns]
        selected_cohorts = st.multiselect("Filtrer par attributs (cohorte similaire)", candidate_cohorts, default=[c for c in candidate_cohorts[:2]])
        cohort_df = pool_df.copy()
        for c in selected_cohorts:
            cohort_df = cohort_df[cohort_df[c] == pool_df.loc[pool_df[ID_COL] == selected_id, c].iloc[0]]
        st.caption(f"Taille de la cohorte similaire : **{len(cohort_df):,}**")

        if global_imp_df is not None and not global_imp_df.empty:
            cand = [f for f in global_imp_df["feature"].tolist() if (f in X.columns or f in pool_df.columns)]
        else:
            cand = [f for f in list(X.columns) if (f in X.columns or f in pool_df.columns)]

        comp_feats = []
        for f in cand:
            if get_quantile_series(f, pool_df, X) is not None:
                comp_feats.append(f)
            if len(comp_feats) >= 8:
                break

        if not comp_feats:
            st.info("Aucune variable numérique comparable disponible.")
        else:
            long_rows = []
            for f in comp_feats:
                s_pop = get_quantile_series(f, pool_df, X)
                if s_pop is None or s_pop.dropna().empty:
                    continue
                s_coh = get_cohort_series(f, cohort_df, X, ID_COL)
                client_val = float(x_row[f].iloc[0]) if f in X.columns and pd.notnull(x_row[f].iloc[0]) else np.nan

                pop_q = s_pop.quantile([0.1, 0.5, 0.9]).values
                if s_coh is not None and not s_coh.dropna().empty:
                    coh_q = s_coh.quantile([0.1, 0.5, 0.9]).values
                else:
                    coh_q = [np.nan, np.nan, np.nan]

                long_rows += [
                    {"feature": f, "group": "Population", "p10": pop_q[0], "p50": pop_q[1], "p90": pop_q[2], "client": client_val},
                    {"feature": f, "group": "Cohorte similaire", "p10": coh_q[0], "p50": coh_q[1], "p90": coh_q[2], "client": client_val},
                ]

            if not long_rows:
                st.info("Aucune variable numérique comparable disponible dans vos données.")
            else:
                long_df = pd.DataFrame(long_rows)
                for grp in ["Population", "Cohorte similaire"]:
                    sub = long_df[long_df.group == grp]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sub["p10"], y=sub["feature"], mode="markers", name="P10"))
                    fig.add_trace(go.Scatter(x=sub["p50"], y=sub["feature"], mode="markers", name="P50"))
                    fig.add_trace(go.Scatter(x=sub["p90"], y=sub["feature"], mode="markers", name="P90"))
                    fig.add_trace(go.Scatter(x=sub["client"], y=sub["feature"], mode="markers", name="Client", marker=dict(symbol="diamond", size=12)))
                    fig.update_layout(title=f"{grp} — Positionnement du client (P10/P50/P90)", height=400)
                    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 4 — Global insights
# -------------------------------
with main_tabs[3]:
    st.subheader("Importance globale & calibration")
    if global_imp_df is not None:
        fig_imp = px.bar(global_imp_df.head(20), x="importance", y="feature", orientation="h", title="Top 20 — Importance globale")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Importance globale non fournie (`global_importance.csv`).")
    st.markdown("**Figures disponibles**")
    figs = [
        _pick_first_existing(["__results___14_1.png", "calibration.png"]),
        _pick_first_existing(["__results___8_0.png", "target_balance.png"]),
    ]
    for p in figs:
        if p:
            st.image(p, use_column_width=True)

# -------------------------------
# Tab 5 — Data quality
# -------------------------------
with main_tabs[4]:
    st.subheader("Qualité des données & valeurs manquantes")
    miss_fig = _pick_first_existing(["__results___5_1.png", "missing_train.png"])
    if miss_fig:
        st.image(miss_fig, caption="Top taux de valeurs manquantes (train)", use_column_width=True)
    else:
        st.info("Figure de valeurs manquantes non trouvée.")
    st.markdown("""
    **Notes**
    - Variables avec >70% de valeurs manquantes nécessitent une attention particulière.
    - Considérer la suppression, l'imputation ciblée ou des modèles robustes au manquant (ex. CatBoost).
    """)

# -------------------------------
# Tab 6 — Nouveau client (what-if)
# -------------------------------
with main_tabs[5]:
    st.subheader("Comparer un nouveau client (what-if)")
    if model is None or X.empty:
        st.info("Modèle ou données indisponibles. Sélectionnez un modèle et chargez un dataset.")
    else:
        st.markdown("Chargez un **CSV** (1 ligne) ou saisissez quelques variables clés pour simuler un nouveau client.")
        up = st.file_uploader("Fichier CSV (1 ligne)", type=["csv"], accept_multiple_files=False)
        topk = st.slider("Nombre de variables clés à saisir (importance globale)", min_value=5, max_value=40, value=15, step=1)
        manual = st.checkbox("Saisie manuelle des variables clés", value=False)

        new_x = None
        if up is not None:
            try:
                df_new = pd.read_csv(up)
                if len(df_new) != 1:
                    st.error("Le CSV doit contenir **exactement 1 ligne**.")
                else:
                    new_x = df_new
            except Exception as e:
                st.error(f"Impossible de lire le CSV : {e}")

        if manual and new_x is None:
            if global_imp_df is not None and not global_imp_df.empty:
                keys = [f for f in global_imp_df["feature"].tolist() if f in X.columns][:topk]
            else:
                keys = list(X.columns)[:topk]
            num_cand = [f for f in keys if pd.api.types.is_numeric_dtype(X[f])]
            cat_cand = [f for f in keys if f not in num_cand]

            st.markdown("**Saisie manuelle** — valeurs par défaut = médiane (num) / modalité la plus fréquente (cat)")
            cols = st.columns(2)
            inputs = {}
            with cols[0]:
                for f in num_cand:
                    series = X[f]
                    default = float(np.nanmedian(series.values)) if np.isfinite(np.nanmedian(series.values)) else 0.0
                    inputs[f] = st.number_input(f, value=float(default))
            with cols[1]:
                for f in cat_cand:
                    series = pd.Series(X[f].dropna().astype(str))
                    mode = series.mode().iloc[0] if not series.empty else "NA"
                    opts = sorted(series.unique().tolist()[:50]) or ["NA"]
                    inputs[f] = st.selectbox(f, options=opts, index=(opts.index(mode) if mode in opts else 0))

            with st.expander("Pré-remplir depuis le client sélectionné"):
                if not x_row.empty:
                    if st.button("Copier les valeurs du client sélectionné"):
                        for f in keys:
                            val = x_row.iloc[0][f] if f in x_row.columns else np.nan
                            if f in num_cand and pd.notnull(val):
                                inputs[f] = float(val)
                            elif f in cat_cand and pd.notnull(val):
                                inputs[f] = str(val)
                        st.experimental_rerun()

            if st.button("Simuler"):
                new_x = pd.DataFrame([inputs])

        if new_x is not None:
            exp_cols = list(X.columns)
            for c in exp_cols:
                if c not in new_x.columns:
                    new_x[c] = np.nan
            new_x = new_x[exp_cols]
            try:
                new_p = float(model.predict_proba(new_x)[0, 1])
                def _band(p: float):
                    if p < 0.05: return ("Faible", "#3CB371")
                    if p < 0.15: return ("Modérée", "#E6B800")
                    return ("Élevée", "#E74C3C")
                band, color = _band(new_p)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=new_p * 100,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
                    title={"text": "Probabilité de défaut (nouveau client)"},
                ))
                st.plotly_chart(fig, use_container_width=True)
                try:
                    vals, base_vals = compute_local_shap(model, background, new_x)
                    ld = pd.DataFrame({
                        "feature": list(X.columns),
                        "shap_value": vals,
                        "abs_val": np.abs(vals),
                        "value": new_x.iloc[0].values,
                    }).sort_values("abs_val", ascending=False).head(10)
                    st.markdown("**Contributions locales (SHAP)** — top 10")
                    st.plotly_chart(px.bar(ld[::-1], x="shap_value", y="feature", orientation="h"), use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP non disponible: {e}")
                if global_imp_df is not None and not global_imp_df.empty:
                    comp_feats = [f for f in global_imp_df.head(6)["feature"].tolist() if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
                else:
                    comp_feats = [f for f in list(X.columns) if pd.api.types.is_numeric_dtype(X[f])][:6]
                long_rows = []
                for f in comp_feats:
                    client_val = float(new_x[f].iloc[0]) if pd.notnull(new_x[f].iloc[0]) else np.nan
                    pop_q = X[f].quantile([0.1, 0.5, 0.9]).values
                    long_rows.append({"feature": f, "group": "Population", "p10": pop_q[0], "p50": pop_q[1], "p90": pop_q[2], "client": client_val})
                if long_rows:
                    long_df = pd.DataFrame(long_rows)
                    figc = go.Figure()
                    figc.add_trace(go.Scatter(x=long_df["p10"], y=long_df["feature"], mode="markers", name="P10"))
                    figc.add_trace(go.Scatter(x=long_df["p50"], y=long_df["feature"], mode="markers", name="P50"))
                    figc.add_trace(go.Scatter(x=long_df["p90"], y=long_df["feature"], mode="markers", name="P90"))
                    figc.add_trace(go.Scatter(x=long_df["client"], y=long_df["feature"], mode="markers", name="Client", marker=dict(symbol="diamond", size=12)))
                    figc.update_layout(title="Positionnement du nouveau client (P10/P50/P90)", height=400)
                    st.plotly_chart(figc, use_container_width=True)
                else:
                    st.info("Aucune variable numérique comparable disponible pour le nouveau client.")
            except Exception as e:
                st.error(f"Échec de la prédiction: {e}")

# -------------------------------
# Tab 7 — Seuil & coût métier
# -------------------------------
with main_tabs[6]:
    st.subheader("Seuil & coût métier (optimisation)")
    if model is None or X.empty or TARGET_COL is None:
        st.info("Pour optimiser le seuil, il faut : un modèle chargé, des données et la colonne TARGET.")
    else:
        cols = st.columns(4)
        with cols[0]:
            unit = st.selectbox("Unité monétaire", ["€", "CHF", "USD"], index=0)
        with cols[1]:
            cost_fp = st.number_input("Coût d'un FP (refus à tort)", min_value=0.0, value=100.0, step=10.0)
        with cols[2]:
            cost_fn = st.number_input("Coût d'un FN (acceptation risquée)", min_value=0.0, value=1000.0, step=10.0)
        with cols[3]:
            max_sample = st.number_input("Taille échantillon (max)", min_value=1000, value=20000, step=1000)

        labeled = pool_df.dropna(subset=[TARGET_COL]).copy()
        if len(labeled) == 0:
            st.warning("Aucune ligne labellisée trouvée (TARGET manquant).")
        else:
            df_lab = labeled.set_index(ID_COL)
            expected = list(X.columns)
            for c in expected:
                if c not in df_lab.columns:
                    df_lab[c] = np.nan
            X_all = df_lab[expected]
            y_all = df_lab[TARGET_COL].astype(int)  # aligné sur le même index

            if len(X_all) > max_sample:
                X_all = X_all.sample(int(max_sample), random_state=42)
                y_all = y_all.loc[X_all.index]

            try:
                p_all = model.predict_proba(X_all)[:, 1]
            except Exception as e:
                st.error(f"Impossible de scorer l'échantillon : {e}")
                p_all = None

            if p_all is not None:
                try:
                    auc = roc_auc_score(y_all.values, p_all)
                except Exception:
                    auc = float("nan")

                try:
                    fpr, tpr, roc_th = roc_curve(y_all.values, p_all)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
                    fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR", height=350)
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception:
                    pass

                try:
                    prec, rec, pr_th = precision_recall_curve(y_all.values, p_all)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Precision-Recall"))
                    fig_pr.update_layout(title="Précision–Rappel", xaxis_title="Recall", yaxis_title="Precision", height=350)
                    st.plotly_chart(fig_pr, use_container_width=True)
                except Exception:
                    pass

                df_cost, best = cost_curve(y_all.values, p_all, cost_fp, cost_fn, step=0.001)
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(x=df_cost["threshold"], y=df_cost["cost"], mode="lines", name="Coût total"))
                fig_cost.add_vline(x=float(best["threshold"]), line_width=2, line_dash="dash", line_color="green",
                                   annotation_text=f"Seuil optimal = {best['threshold']:.3f}",
                                   annotation_position="top left")
                fig_cost.add_vline(x=float(st.session_state["threshold"]), line_width=2, line_dash="dot", line_color="red",
                                   annotation_text=f"Seuil courant = {st.session_state['threshold']:.3f}",
                                   annotation_position="top right")
                fig_cost.update_layout(title=f"Coût vs Seuil ({unit})", xaxis_title="Seuil", yaxis_title=f"Coût total ({unit})", height=350)
                st.plotly_chart(fig_cost, use_container_width=True)

                cur = cost_at_threshold(y_all.values, p_all, float(st.session_state["threshold"]), cost_fp, cost_fn)
                best_row = {
                    "Seuil": f"{best['threshold']:.3f}",
                    "Coût total": f"{best['cost']:.0f} {unit}",
                    "TP": int(best["tp"]), "FP": int(best["fp"]), "FN": int(best["fn"]), "TN": int(best["tn"]),
                    "Précision": f"{best['precision']:.3f}", "Rappel": f"{best['recall']:.3f}", "F1": f"{best['f1']:.3f}",
                }
                cur_row = {
                    "Seuil": f"{st.session_state['threshold']:.3f}",
                    "Coût total": f"{cur['cost']:.0f} {unit}",
                    "TP": int(cur["tp"]), "FP": int(cur["fp"]), "FN": int(cur["fn"]), "TN": int(cur["tn"]),
                    "Précision": f"{cur['precision']:.3f}", "Rappel": f"{cur['recall']:.3f}", "F1": f"{cur['f1']:.3f}",
                }
                st.markdown("**Synthèse**")
                st.dataframe(pd.DataFrame([best_row, cur_row], index=["Seuil optimal", "Seuil courant"]))

                c1, c2 = st.columns([1,2])
                with c1:
                    if st.button("✅ Appliquer le seuil optimal au dashboard"):
                        st.session_state["threshold"] = float(best["threshold"])
                        st.success(f"Seuil mis à jour à {best['threshold']:.3f}.")
                        st.experimental_rerun()
                with c2:
                    st.caption("Le seuil optimal minimise le coût total attendu : `coût = FP × coût_FP + FN × coût_FN`.")

# Footer
st.divider()
left, mid, right = st.columns([2,2,1])
with left:
    st.caption("© Prêt à dépenser — Dashboard pédagogique. Transparence & explicabilité des décisions d'octroi.")
with mid:
    st.caption("App version: " + APP_VERSION)
with right:
    st.caption("Build: Streamlit + SHAP + scikit-learn")
