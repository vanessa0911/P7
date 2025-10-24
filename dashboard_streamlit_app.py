# Streamlit Credit Scoring Dashboard — "Prêt à dépenser" (v0.8.1)
# ----------------------------------------------------------------
# Run:
#   python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
#
# Modes de scoring:
#   - Local (modèle .joblib embarqué) — comme avant
#   - API FastAPI (credit_api.py) — appelle /predict pour une prédiction et des explications
#
# Fichiers détectés à la racine:
# - Data:  application_train_clean.csv  |  clients_demo.csv  |  clients_demo.parquet
# - Model: model_calibrated_isotonic.joblib | model_calibrated_sigmoid.joblib | model_baseline_logreg.joblib
# - Features (optional): feature_names.npy
# - Global importance (optional): global_importance.csv
# - Interpretability (optional): interpretability_summary.json

APP_VERSION = "0.8.1"

import os
import json
import subprocess
import hashlib
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# --- PDF (ReportLab) ---
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Prêt à dépenser — Credit Scoring", page_icon="💳", layout="wide")

# -------------------------------
# Runtime diagnostics
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
    try:
        return load_model(path)
    except ModuleNotFoundError as e:
        st.error(f"Le modèle nécessite le paquet manquant: `{e.name}`. `pip install {e.name}` puis relancez l'app.")
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

def compute_local_shap(estimator, X_background: pd.DataFrame, x_row: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    import shap, pandas as pd, numpy as np
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

def get_quantile_series(feature: str, pool_df: pd.DataFrame, X: pd.DataFrame) -> Optional[pd.Series]:
    if feature in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[feature]):
        return pool_df[feature]
    if feature in X.columns and pd.api.types.is_numeric_dtype(X[feature]):
        return X[feature]
    return None

def get_cohort_series(feature: str, cohort_df: pd.DataFrame, X: pd.DataFrame, ID_COL: Optional[str]) -> Optional[pd.Series]:
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

def cost_at_threshold(y_true: np.ndarray, p: np.ndarray, t: float, cost_fp: float, cost_fn: float):
    y_pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    cost = fp * cost_fp + fn * cost_fn
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return dict(cost=float(cost), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=float(prec), recall=float(rec), f1=float(f1))

def cost_curve(y_true: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float, step: float=0.001):
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
    cost_arr = pd.to_numeric(df["cost"], errors="coerce").to_numpy()
    cost_arr = np.where(np.isfinite(cost_arr), cost_arr, np.inf)
    best_pos = int(np.argmin(cost_arr))
    best = df.iloc[best_pos].to_dict()
    return df, best

# API helpers
def api_health(base_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=4)
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def api_predict(base_url: str, payload: dict) -> dict:
    r = requests.post(base_url.rstrip("/") + "/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# ---- PDF builder ----
def build_client_report_pdf(
    client_id: str,
    model_name: str,
    threshold: float,
    proba: Optional[float],
    x_row: pd.DataFrame,
    X: pd.DataFrame,
    pool_df: pd.DataFrame,
    global_imp_df: Optional[pd.DataFrame],
    shap_vals: Optional[pd.DataFrame] = None,
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=18, leading=22, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor="#555555"))

    story, now = [], datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Prêt à dépenser — Fiche client", styles["TitleBig"]))
    header = f"Date: {now} • App: {APP_VERSION} • Modèle/Mode: {model_name} • Client: {client_id}"
    story.append(Paragraph(header, styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Score & décision", styles["H2"]))
    if proba is not None:
        decision = "Refus" if proba >= threshold else "Accord"
        band, _ = prob_to_band(proba)
        tbl = [
            ["Probabilité de défaut", f"{proba*100:.2f} %"],
            ["Seuil (proba défaut)", f"{threshold:.3f}"],
            ["Décision", decision],
            ["Niveau de risque", band],
        ]
    else:
        tbl = [["Probabilité de défaut", "—"], ["Seuil", f"{threshold:.3f}"], ["Décision", "—"], ["Niveau de risque", "—"]]
    t = Table(tbl, hAlign="LEFT", colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))
    story.append(t); story.append(Spacer(1, 8))

    story.append(Paragraph("Contributions locales (top 10)", styles["H2"]))
    if shap_vals is not None and not shap_vals.empty:
        dfc = shap_vals.sort_values("abs_val", ascending=False).head(10).copy()
        dfc["effet"] = dfc["shap_value"].apply(lambda v: "↑ risque" if v > 0 else ("↓ risque" if v < 0 else "neutre"))
        data = [["Variable", "Valeur", "Contribution", "Effet"]] + \
               [[str(r["feature"]), str(r["value"]), f'{r["shap_value"]:+.4f}', r["effet"]] for _, r in dfc.iterrows()]
        t2 = Table(data, hAlign="LEFT", colWidths=[7*cm, 3.5*cm, 3.5*cm, 2*cm])
    elif global_imp_df is not None and not global_imp_df.empty:
        dfc = global_imp_df.head(10)
        data = [["Variable", "Importance"]] + [[str(r["feature"]), f'{r["importance"]:.4f}'] for _, r in dfc.iterrows()]
        t2 = Table(data, hAlign="LEFT", colWidths=[10*cm, 4*cm])
    else:
        t2 = Table([["Information", "Détail"], ["Explicabilité", "Indisponible"]], hAlign="LEFT", colWidths=[10*cm, 4*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (2,1), (2,-1), "RIGHT"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(t2); story.append(Spacer(1, 8))

    story.append(Paragraph("Variables clés", styles["H2"]))
    if global_imp_df is not None and not global_imp_df.empty:
        keys = [f for f in global_imp_df["feature"].tolist() if f in X.columns][:20]
    else:
        keys = list(X.columns)[:20]
    kv = [["Variable", "Valeur"]]
    row = x_row.iloc[0] if not x_row.empty else pd.Series(dtype=object)
    for f in keys:
        v = row[f] if f in row.index else np.nan
        kv.append([str(f), "" if pd.isna(v) else str(v)])
    t3 = Table(kv, hAlign="LEFT", colWidths=[9*cm, 5*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(t3); story.append(Spacer(1, 8))

    story.append(Paragraph("Positionnement vs population (P10 / P50 / P90)", styles["H2"]))
    if global_imp_df is not None and not global_imp_df.empty:
        comp_feats = [f for f in global_imp_df["feature"].tolist() if f in X.columns and pd.api.types.is_numeric_dtype(X[f])][:6]
    else:
        comp_feats = [f for f in list(X.columns) if pd.api.types.is_numeric_dtype(X[f])][:6]
    comp_tbl = [["Variable", "P10", "P50", "P90", "Client"]]
    for f in comp_feats:
        s = X[f].dropna()
        if s.empty:
            continue
        p10, p50, p90 = np.nanpercentile(s.values, [10, 50, 90])
        client_val = row[f] if f in row.index else np.nan
        comp_tbl.append([str(f),
                         f"{p10:.4g}" if pd.notnull(p10) else "—",
                         f"{p50:.4g}" if pd.notnull(p50) else "—",
                         f"{p90:.4g}" if pd.notnull(p90) else "—",
                         f"{client_val:.4g}" if pd.notnull(client_val) else "—"])
    if len(comp_tbl) == 1:
        comp_tbl.append(["—", "—", "—", "—", "—"])
    t4 = Table(comp_tbl, hAlign="LEFT", colWidths=[6*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(t4)

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# -------------------------------
# Locate artifacts at repo root
# -------------------------------
DATA_TRAIN = _pick_first_existing([
    "application_train_clean.csv",
    "clients_demo.csv",
    "clients_demo.parquet",
])
DATA_TEST  = _pick_first_existing(["application_test_clean.csv"])
MODEL_ISO  = _pick_first_existing(["model_calibrated_isotonic.joblib"])
MODEL_SIG  = _pick_first_existing(["model_calibrated_sigmoid.joblib"])
MODEL_BASE = _pick_first_existing(["model_baseline_logreg.joblib"])
FEATS_PATH = _pick_first_existing(["feature_names.npy"])
GLOBIMP    = _pick_first_existing(["global_importance.csv"])
INTERP_SUM = _pick_first_existing(["interpretability_summary.json"])

with st.sidebar:
    st.title("💳 Scoring Crédit — Dashboard")
    st.caption("Prêt à dépenser — transparence & explicabilité")

    # Diagnostics
    path, mtime_str, sha8, git = _runtime_info()
    st.caption(f"App version: {APP_VERSION}")
    st.caption(f"Fichier: {os.path.basename(path)}")
    st.caption(f"Dernière modif: {mtime_str}")
    st.caption(f"SHA fichier: {sha8} | Git: {git}")

    if st.button("🔄 Forcer rechargement (vider cache)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    # Mode de scoring
    st.subheader("Mode de scoring")
    mode = st.radio("Choisir le mode", ["Local (modèle embarqué)", "API FastAPI"], index=0)
    api_base = None
    api_ok = False
    if mode == "API FastAPI":
        api_base = st.text_input("API base URL", value="http://localhost:8000", help="Lance `uvicorn credit_api:app --host 0.0.0.0 --port 8000 --reload`")
        if api_base:
            ok, msg = api_health(api_base)
            api_ok = ok
            st.caption(f"Statut API: {'✅ OK' if ok else '❌ KO'} — {msg}")

    if not DATA_TRAIN:
        st.error("⚠️ Données non trouvées (placez `clients_demo.csv` ou `clients_demo.parquet` à la racine)")

# Load datasets
train_df = load_table(DATA_TRAIN) if DATA_TRAIN else pd.DataFrame()
holdout_df = load_table(DATA_TEST) if DATA_TEST else pd.DataFrame()
pool_df = train_df if not train_df.empty else holdout_df
feature_names = load_feature_names(FEATS_PATH, list(pool_df.columns)) if not pool_df.empty else []
global_imp_df = load_global_importance(GLOBIMP)
interp_summary = load_interpretability_summary(INTERP_SUM)

ID_COL = "SK_ID_CURR" if (not pool_df.empty and "SK_ID_CURR" in pool_df.columns) else (pool_df.columns[0] if not pool_df.empty else None)
TARGET_COL = "TARGET" if (not pool_df.empty and "TARGET" in pool_df.columns) else None

# Sidebar: paramètres + client
with st.sidebar:
    st.subheader("Paramètres du modèle / seuil")
    default_thresh = st.session_state.get("threshold", 0.08)
    threshold = st.slider("Seuil d'acceptation (proba défaut)", 0.0, 0.5, float(default_thresh), 0.005, key="threshold",
                          help="Au-delà du seuil = risque élevé ⇒ refus")

    st.subheader("Sélection du client")
    id_options = pool_df[ID_COL].tolist() if (ID_COL and not pool_df.empty) else []
    selected_id = st.selectbox("SK_ID_CURR", id_options, index=0 if id_options else None)
    st.caption("Astuce : utilisez le champ de recherche pour filtrer par ID.")

# Charger modèle local si nécessaire
model_paths = {}
if MODEL_ISO: model_paths["Calibré (Isotonic)"] = MODEL_ISO
if MODEL_SIG: model_paths["Calibré (Sigmoid)"]  = MODEL_SIG
if MODEL_BASE: model_paths["Baseline"]          = MODEL_BASE

model_name_local = list(model_paths.keys())[0] if model_paths else "—"
model_local = None
if mode == "Local (modèle embarqué)":
    if model_paths:
        try:
            model_local = safe_load_model(model_paths[model_name_local])
        except Exception:
            model_local = None

# Préparer X aligné (utile pour affichages & PDF, même en mode API)
if not pool_df.empty and selected_id is not None:
    df_idx = pool_df.set_index(ID_COL)
    expected_cols_local = get_expected_input_columns(model_local) if model_local is not None else (feature_names or list(df_idx.columns))
    for c in expected_cols_local:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    X = df_idx[expected_cols_local]
    x_row = X.loc[[selected_id]]
    background = X.sample(min(200, len(X)), random_state=42)
else:
    X = pd.DataFrame(columns=feature_names)
    x_row = X.head(0)
    background = X

# -------------------------------
# Tabs
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
# Tab 1 — Score & explication
# -------------------------------
with main_tabs[0]:
    st.subheader("Score individuel & interprétation")

    proba = None
    shap_df = None
    source_label = "Local"
    # ==== PREDICTION ====
    if mode == "API FastAPI":
        if not api_base or not api_ok:
            st.error("API indisponible. Lance `uvicorn credit_api:app --host 0.0.0.0 --port 8000 --reload` puis rafraîchis.")
        else:
            try:
                payload = {"client_id": selected_id, "threshold": float(threshold), "shap": True, "topk": 10}
                resp = api_predict(api_base, payload)
                proba = float(resp["proba_default"])
                source_label = "API"
                # Top contrib renvoyées par l'API (si shap=True)
                if resp.get("top_contrib"):
                    rows = []
                    for r in resp["top_contrib"]:
                        rows.append({
                            "feature": r["feature"],
                            "shap_value": float(r["shap_value"]),
                            "abs_val": abs(float(r["shap_value"])),
                            "value": r["value"],
                        })
                    shap_df = pd.DataFrame(rows)
            except Exception as e:
                st.warning(f"API KO ({e}). Bascule en mode Local si un modèle est disponible.")
                if model_local is None:
                    st.stop()
                # fallback local
                proba = float(model_local.predict_proba(x_row)[0, 1])
    else:
        # Local
        if model_local is None or x_row.empty:
            st.warning("Modèle local ou données indisponibles.")
        else:
            proba = float(model_local.predict_proba(x_row)[0, 1])

    if proba is None:
        st.stop()

    # rendu
    def _band(p: float):
        if p < 0.05: return ("Faible", "#3CB371")
        if p < 0.15: return ("Modérée", "#E6B800")
        return ("Élevée", "#E74C3C")
    band, color = _band(proba)

    col1, col2 = st.columns([1, 2])
    with col
