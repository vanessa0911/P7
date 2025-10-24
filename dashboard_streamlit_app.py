# Streamlit Credit Scoring Dashboard — "Prêt à dépenser" (v0.4.1)
# ---------------------------------------------------------------
# Run:  streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
# One-file app (no folders required). Place artifacts at the repo root.
#
# Root files auto-detected (first found wins):
# - Data:  application_train_clean.csv  |  clients_demo.csv  |  clients_demo.parquet
# - Model: model_calibrated_isotonic.joblib | model_calibrated_sigmoid.joblib | model_baseline_logreg.joblib
# - Features: feature_names.npy (optional)
# - Global importance: global_importance.csv (optional)
# - Interpretability: interpretability_summary.json (optional)

APP_VERSION = "0.4.1"

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer

st.set_page_config(page_title="Prêt à dépenser — Credit Scoring", page_icon="💳", layout="wide")

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

# Safe loader that tells which dependency is missing (e.g. catboost)

def safe_load_model(path: str):
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

# Get raw expected input columns from a Pipeline/ColumnTransformer

def get_expected_input_columns(model) -> Optional[List[str]]:
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

# Unwrap calibrated models to the base estimator if possible

def unwrap_estimator(m):
    base = m
    try:
        # Pipeline → take last step (estimator side)
        if isinstance(base, SkPipeline):
            base = base.steps[-1][1]
        # CalibratedClassifierCV variants
        for attr in ("base_estimator", "estimator", "calibrated_classifiers_"):
            if hasattr(base, attr):
                base = getattr(base, attr)
                if isinstance(base, list) and len(base) > 0:
                    inner = base[0]
                    if hasattr(inner, "base_estimator"):
                        base = inner.base_estimator
                    elif hasattr(inner, "estimator"):
                        base = inner.estimator
                break
    except Exception:
        pass
    return base

# Local SHAP with two strategies:
# 1) If we can isolate a tree-based estimator *after* preprocessing, use TreeExplainer on transformed inputs.
# 2) Fallback: Kernel-based explainer on the full pipeline predict_proba (robust but slower).

def compute_local_shap(estimator, X_background: pd.DataFrame, x_row: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as np
    import pandas as pd
    import shap

    # Try to locate a ColumnTransformer in a pipeline for transformation
    ct = None
    est = estimator
    if isinstance(estimator, SkPipeline):
        for name, step in estimator.steps:
            if isinstance(step, SkColumnTransformer):
                ct = step
        # last estimator is at the end of the pipeline
    base = unwrap_estimator(estimator)

    # If we have a transformer, transform background and row to numeric space
    Xbg = X_background
    x1 = x_row
    feat_names_out = None
    try:
        if ct is not None:
            Xbg_t = ct.transform(Xbg)
            x1_t = ct.transform(x1)
            # feature names out if available (sklearn >=1.0)
            try:
                feat_names_out = ct.get_feature_names_out()
            except Exception:
                feat_names_out = None
            # Try tree explainer on the base estimator with transformed inputs
            try:
                expl = shap.TreeExplainer(base)
                sv = expl.shap_values(x1_t)
                # CatBoost returns list for multiclass; binary returns array
                vals = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
                base_vals = expl.expected_value[1] if isinstance(expl.expected_value, (list, tuple, np.ndarray)) else expl.expected_value
                return np.array(vals).reshape(-1), np.array([base_vals]).reshape(-1)
            except Exception:
                # fall back to kernel on transformed numeric inputs
                def f_num(Xnp):
                    return base.predict_proba(Xnp)[:, 1]
                masker = shap.maskers.Independent(Xbg_t)
                expl = shap.Explainer(f_num, masker)
                ex = expl(x1_t)
                return np.array(ex.values).reshape(-1), np.array(ex.base_values).reshape(-1)
    except Exception:
        pass

    # Fallback: kernel on the full estimator with raw inputs (may be slower with categoricals)
    def f_raw(Xdf):
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf, columns=list(X_background.columns))
        return estimator.predict_proba(Xdf)[:, 1]
    try:
        masker = shap.maskers.Independent(X_background)
        expl = shap.Explainer(f_raw, masker, feature_names=list(X_background.columns))
        ex = expl(x_row)
        return np.array(ex.values).reshape(-1), np.array(ex.base_values).reshape(-1)
    except Exception as e:
        raise RuntimeError(f"SHAP indisponible: {e}")

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
if MODEL_BASE: model_paths["Baseline"]           = MODEL_BASE

# Interpretability artifacts
global_imp_df = load_global_importance(GLOBIMP)
interp_summary = load_interpretability_summary(INTERP_SUM)

# IDs and target
ID_COL = "SK_ID_CURR" if (not pool_df.empty and "SK_ID_CURR" in pool_df.columns) else (pool_df.columns[0] if not pool_df.empty else None)
TARGET_COL = "TARGET" if (not pool_df.empty and "TARGET" in pool_df.columns) else None

# Inferred types
num_cols = [c for c in feature_names if c in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[c])]
cat_cols = [c for c in feature_names if c in pool_df.columns and c not in num_cols]

# Sidebar controls
with st.sidebar:
    st.subheader("Paramètres du modèle")
    model_name = st.selectbox("Choisir le modèle", list(model_paths.keys()) if model_paths else ["—"])
    threshold = st.slider("Seuil d'acceptation (proba défaut)", 0.0, 0.5, 0.08, 0.005,
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
    for c in temp_expected:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    X = df_idx[temp_expected]
    x_row = X.loc[[selected_id]]
    background = X.sample(min(500, len(X)), random_state=42)
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
        band = "Faible" if proba < 0.05 else ("Modérée" if proba < 0.15 else "Élevée")
        color = {"Faible": "#3CB371", "Modérée": "#E6B800", "Élevée": "#E74C3C"}[band]
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
            if not background.empty and model is not None:
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
                except Exception:
                    if global_imp_df is not None:
                        st.info("SHAP indisponible pour ce modèle/format. Affichage des 10 variables les plus importantes (globales).")
                        st.dataframe(global_imp_df.head(10))
                    else:
                        st.info("SHAP et importance globale indisponibles.")

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
# Tab 3 — Comparison (client vs population or similar group)
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

        comp_feats = [f for f in (global_imp_df.head(8)["feature"].tolist() if global_imp_df is not None else list(X.columns)[:8]) if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
        if not comp_feats:
            st.info("Pas de variables numériques à comparer.")
        else:
            long_rows = []
            for f in comp_feats:
                client_val = float(x_row[f].iloc[0]) if f in X.columns and pd.notnull(x_row[f].iloc[0]) else np.nan
                # Use pool_df when the feature exists in the raw dataset; otherwise skip quantiles for that feature
                if f in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[f]):
                    pop_q = pool_df[f].quantile([0.1, 0.5, 0.9]).values
                    coh_q = cohort_df[f].quantile([0.1, 0.5, 0.9]).values if len(cohort_df) > 1 else [np.nan, np.nan, np.nan]
                    long_rows += [
                        {"feature": f, "group": "Population", "p10": pop_q[0], "p50": pop_q[1], "p90": pop_q[2], "client": client_val},
                        {"feature": f, "group": "Cohorte similaire", "p10": coh_q[0], "p50": coh_q[1], "p90": coh_q[2], "client": client_val},
                    ]
            if not long_rows:
                st.info("Aucune variable numérique comparable disponible dans le dataset brut.")
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
    st.subheader("Comparer un nouveau client (what‑if)")
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
            # Variables clés = topK par importance globale si dispo, sinon premières colonnes
            if global_imp_df is not None and not global_imp_df.empty:
                keys = [f for f in global_imp_df["feature"].tolist() if f in X.columns][:topk]
            else:
                keys = list(X.columns)[:topk]
            # Séparer numériques / catégorielles parmi ces clés
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

            # Option: pré-remplir depuis le client sélectionné
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
                new_x = pd.DataFrame([inputs])

        if new_x is not None:
            exp_cols = list(X.columns)
            for c in exp_cols:
                if c not in new_x.columns:
                    new_x[c] = np.nan
            new_x = new_x[exp_cols]
            try:
                new_p = float(model.predict_proba(new_x)[0, 1])
                band = "Faible" if new_p < 0.05 else ("Modérée" if new_p < 0.15 else "Élevée")
                color = {"Faible": "#3CB371", "Modérée": "#E6B800", "Élevée": "#E74C3C"}[band]
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
                except Exception:
                    st.info("SHAP non disponible pour ce modèle/format.")
                # quick comparison to population quantiles
                comp_feats = [f for f in (global_imp_df.head(6)["feature"].tolist() if global_imp_df is not None else list(X.columns)[:6]) if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
                long_rows = []
                for f in comp_feats:
                    client_val = float(new_x[f].iloc[0]) if pd.notnull(new_x[f].iloc[0]) else np.nan
                    pop_q = X[f].quantile([0.1, 0.5, 0.9]).values
                    long_rows.append({"feature": f, "group": "Population", "p10": pop_q[0], "p50": pop_q[1], "p90": pop_q[2], "client": client_val})
                long_df = pd.DataFrame(long_rows)
                figc = go.Figure()
                figc.add_trace(go.Scatter(x=long_df["p10"], y=long_df["feature"], mode="markers", name="P10"))
                figc.add_trace(go.Scatter(x=long_df["p50"], y=long_df["feature"], mode="markers", name="P50"))
                figc.add_trace(go.Scatter(x=long_df["p90"], y=long_df["feature"], mode="markers", name="P90"))
                figc.add_trace(go.Scatter(x=long_df["client"], y=long_df["feature"], mode="markers", name="Client", marker=dict(symbol="diamond", size=12)))
                figc.update_layout(title="Positionnement du nouveau client (P10/P50/P90)", height=400)
                st.plotly_chart(figc, use_container_width=True)
            except Exception as e:
                st.error(f"Échec de la prédiction: {e}")

# Footer
st.divider()
left, mid, right = st.columns([2,2,1])
with left:
    st.caption("© Prêt à dépenser — Dashboard pédagogique. Transparence & explicabilité des décisions d'octroi.")
with mid:
    st.caption("App version: " + APP_VERSION)
with right:
    st.caption("Build: Streamlit + SHAP + scikit-learn")
