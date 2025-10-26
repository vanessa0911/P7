# Streamlit Credit Scoring Dashboard — "Prêt à dépenser" (v1.2.2)
# ----------------------------------------------------------------
# Run:
#   python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

APP_VERSION = "1.2.2"

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
st.title("💳 Prêt à dépenser — Credit Scoring")
st.caption("Transparence & explicabilité des décisions d’octroi")

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

def get_expected_input_columns(model) -> Optional[List[str]]:
    try:
        m = model
        if isinstance(m, SkPipeline):
            # ColumnTransformer inside Pipeline?
            for _, step in m.steps:
                if isinstance(step, SkColumnTransformer) and hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
    except Exception:
        pass
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

# ---------- Local interpretability ----------
def _is_all_numeric(df: pd.DataFrame) -> bool:
    try:
        return all(pd.api.types.is_numeric_dtype(t) for _, t in df.dtypes.items())
    except Exception:
        return False

def compute_local_shap_or_delta(estimator, background: pd.DataFrame, x_row: pd.DataFrame,
                                max_features: int = 50) -> Tuple[pd.DataFrame, str]:
    """
    Tente SHAP (agnostique) ; si ça échoue, fallback 'delta-contributions':
    on remplace chaque feature par une valeur de référence (médiane ou modalité majoritaire),
    on recalcule la proba, et la 'contribution' = base_proba - proba_ref.
    Retourne (df_contrib, method_label)
    """
    # 1) Tentative SHAP
    try:
        import shap
        # Eviter l'erreur 'isfinite' -> si non numérique, on bascule direct en delta
        if _is_all_numeric(background) and _is_all_numeric(x_row):
            bg = background.copy()
            if len(bg) > 200:
                bg = bg.sample(200, random_state=42)
            def f(Xdf):
                if not isinstance(Xdf, pd.DataFrame):
                    Xdf = pd.DataFrame(Xdf, columns=list(bg.columns))
                return estimator.predict_proba(Xdf)[:, 1]
            masker = shap.maskers.Independent(bg)
            explainer = shap.Explainer(f, masker, feature_names=list(bg.columns))
            ex = explainer(x_row)
            vals = np.array(ex.values).reshape(-1)
            out = pd.DataFrame({
                "feature": list(x_row.columns),
                "value": x_row.iloc[0].values,
                "contrib": vals,
                "abs_val": np.abs(vals),
            }).sort_values("abs_val", ascending=False)
            return out.head(max_features), "SHAP"
        # sinon -> delta
    except Exception:
        pass

    # 2) Fallback delta-contributions (robuste à tout)
    try:
        # Base proba
        base_p = float(estimator.predict_proba(x_row)[0, 1])

        # Valeurs de référence: médiane pour numériques, modalité la plus fréquente pour catégorielles
        ref_vals: Dict[str, Any] = {}
        for c in x_row.columns:
            col_pop = background[c] if c in background.columns else None
            if col_pop is not None and pd.api.types.is_numeric_dtype(col_pop):
                ref_vals[c] = float(np.nanmedian(col_pop.values)) if not col_pop.dropna().empty else 0.0
            else:
                # Catégorie: prendre la modalité la plus fréquente (si dispo), sinon valeur actuelle
                if col_pop is not None and not col_pop.dropna().empty:
                    mode = pd.Series(col_pop.dropna().astype(str)).mode()
                    ref_vals[c] = mode.iloc[0] if len(mode) else x_row.iloc[0][c]
                else:
                    ref_vals[c] = x_row.iloc[0][c]

        rows = []
        # On ne prend pas 200 features : limite à max_features (en suivant global importance si dispo ailleurs)
        for c in list(x_row.columns)[:max_features]:
            x_tmp = x_row.copy()
            x_tmp.iloc[0, x_tmp.columns.get_loc(c)] = ref_vals[c]
            try:
                p_ref = float(estimator.predict_proba(x_tmp)[0, 1])
                contrib = base_p - p_ref  # >0 => la valeur actuelle ↑ risque
                rows.append({
                    "feature": c,
                    "value": x_row.iloc[0][c],
                    "contrib": contrib,
                    "abs_val": abs(contrib),
                })
            except Exception:
                # ignorer si le modèle n'accepte pas la perturbation (rare)
                continue

        df = pd.DataFrame(rows).sort_values("abs_val", ascending=False)
        return df, "DELTA"
    except Exception as e:
        # Dernier filet
        return pd.DataFrame(columns=["feature", "value", "contrib", "abs_val"]), "NONE"

# ---- PDF builders ----
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
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=18, leading=22, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor="#555555"))

    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Contributions locales (top 10)", styles["H2"]))
    if shap_vals is not None and not shap_vals.empty:
        dfc = shap_vals.sort_values("abs_val", ascending=False).head(10).copy()
        dfc["effet"] = dfc["contrib"].apply(lambda v: "↑ risque" if v > 0 else ("↓ risque" if v < 0 else "neutre"))
        data = [["Variable", "Valeur", "Contribution", "Effet"]] + \
               [[str(r["feature"]), str(r["value"]), f'{r["contrib"]:+.4f}', r["effet"]] for _, r in dfc.iterrows()]
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
    story.append(t2)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Variables clés", styles["H2"]))
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
    story.append(t3)
    story.append(Spacer(1, 8))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

def build_new_client_report_pdf(
    proba: Optional[float],
    threshold: float,
    decision: str,
    band_label: str,
    new_x: pd.DataFrame,
    X: pd.DataFrame,
    pool_df: pd.DataFrame,
    global_imp_df: Optional[pd.DataFrame],
    shap_df: Optional[pd.DataFrame],
) -> bytes:
    """PDF dédié 'Nouveau client' + axes d’amélioration & points forts."""
    if not REPORTLAB_AVAILABLE:
        return b""

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=18, leading=22, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor="#555555"))

    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Prêt à dépenser — Résultats (Nouveau client)", styles["TitleBig"]))
    story.append(Paragraph(f"Date: {now} • App: {APP_VERSION}", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Score & décision", styles["H2"]))
    if proba is not None:
        tbl = [
            ["Probabilité de défaut", f"{proba*100:.2f} %"],
            ["Seuil (proba défaut)", f"{threshold:.3f}"],
            ["Décision", decision],
            ["Niveau de risque", band_label],
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
    story.append(t)
    story.append(Spacer(1, 8))

    # Axes/Points forts via shap_df (qui peut venir du fallback delta)
    story.append(Paragraph("Contributions locales (top 10)", styles["H2"]))
    if shap_df is not None and not shap_df.empty:
        dfc = shap_df.sort_values("abs_val", ascending=False).head(10).copy()
        dfc["effet"] = dfc["contrib"].apply(lambda v: "↑ risque" if v > 0 else ("↓ risque" if v < 0 else "neutre"))
        data = [["Variable", "Valeur", "Contribution", "Effet"]] + \
               [[str(r["feature"]), str(r["value"]), f'{r["contrib"]:+.4f}', r["effet"]] for _, r in dfc.iterrows()]
        t4 = Table(data, hAlign="LEFT", colWidths=[6.0*cm, 3.5*cm, 3.0*cm, 2.0*cm])
    else:
        t4 = Table([["Information", "Détail"], ["Explicabilité", "Indisponible"]], hAlign="LEFT", colWidths=[10*cm, 4*cm])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("BOX", (0,0), (-1,-1), 0.25, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (2,1), (2,-1), "RIGHT"),
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

with st.sidebar:
    # Diagnostics
    path, mtime_str, sha8, git = _runtime_info()
    st.caption(f"Fichier: {os.path.basename(path)}")
    st.caption(f"Dernière modif: {mtime_str}")
    st.caption(f"SHA fichier: {sha8} | Git: {git}")

    if st.button("🔄 Forcer rechargement (vider cache)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Mode de scoring
    st.subheader("Mode de scoring")
    mode = st.radio("Choisir le mode", ["Local (modèle embarqué)", "API FastAPI"], index=0)
    api_base = None
    api_ok = False
    if mode == "API FastAPI":
        api_base = st.text_input("API base URL", value="http://localhost:8000")
        def api_health(base_url: str) -> tuple[bool, str]:
            try:
                r = requests.get(base_url.rstrip("/") + "/health", timeout=4)
                if r.status_code == 200:
                    return True, "ok"
                return False, f"HTTP {r.status_code}"
            except Exception as e:
                return False, str(e)
        if api_base:
            ok, msg = api_health(api_base)
            api_ok = ok
            st.caption(f"Statut API: {'✅ OK' if ok else '❌ KO'} — {msg}")

    # Paramètres
    st.subheader("Paramètres du modèle / seuil")
    default_thresh = float(np.clip(st.session_state.get("threshold", 0.67), 0.0, 1.0))
    threshold = st.slider(
        "Seuil d'acceptation (proba défaut)",
        0.0, 1.0, float(default_thresh), 0.001, key="threshold",
        help="Au-delà du seuil = risque élevé ⇒ refus")

# Load datasets
train_df = load_table(DATA_TRAIN) if DATA_TRAIN else pd.DataFrame()
holdout_df = load_table(DATA_TEST) if DATA_TEST else pd.DataFrame()
pool_df = train_df if not train_df.empty else holdout_df
feature_names = load_feature_names(FEATS_PATH, list(pool_df.columns)) if not pool_df.empty else []
global_imp_df = load_global_importance(GLOBIMP)

# Identify ID/TARGET
ID_COL = "SK_ID_CURR" if (not pool_df.empty and "SK_ID_CURR" in pool_df.columns) else (pool_df.columns[0] if not pool_df.empty else None)
TARGET_COL = "TARGET" if (not pool_df.empty and "TARGET" in pool_df.columns) else None

# Sidebar client picker (after data known)
with st.sidebar:
    st.subheader("Sélection du client")
    id_options = pool_df[ID_COL].tolist() if (ID_COL and not pool_df.empty) else []
    selected_id = st.selectbox("SK_ID_CURR", id_options, index=0 if id_options else None)

# Load local model if needed
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

# Align X
if not pool_df.empty and selected_id is not None:
    df_idx = pool_df.set_index(ID_COL)
    expected_cols_local = get_expected_input_columns(model_local) if model_local is not None else (feature_names or list(df_idx.columns))
    # Ensure all expected cols exist
    for c in expected_cols_local:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    X = df_idx[expected_cols_local].copy()
    x_row = X.loc[[selected_id]]
    # petit background pour local explications
    background = X.sample(min(300, len(X)), random_state=42).copy()
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
    "🧪 Qualité des données",
    "🆕 Nouveau client",
    "💰 Seuil & coût métier",
    "📚 Dictionnaire des variables",
    "🧮 Ratios (feature engineering)",
]
main_tabs = st.tabs(TABS)

# -------------------------------
# Tab 1 — Score & explication
# -------------------------------
with main_tabs[0]:
    st.subheader("Score individuel & interprétation")
    proba = None
    contrib_df = None
    method = "—"
    source_label = "Local"

    # API helpers (inline to avoid clutter)
    def api_predict(base_url: str, payload: dict) -> dict:
        r = requests.post(base_url.rstrip("/") + "/predict", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    # Predict
    if mode == "API FastAPI":
        if not api_base or not api_ok:
            st.error("API indisponible.")
        else:
            try:
                payload = {"client_id": selected_id, "threshold": float(threshold), "shap": False}
                resp = api_predict(api_base, payload)
                proba = float(resp["proba_default"])
                source_label = "API"
            except Exception as e:
                st.warning(f"API KO ({e}). Bascule en Local si possible.")
                if model_local is None:
                    st.stop()
                proba = float(model_local.predict_proba(x_row)[0, 1])
    else:
        if model_local is None or x_row.empty:
            st.warning("Modèle local ou données indisponibles.")
        else:
            proba = float(model_local.predict_proba(x_row)[0, 1])

    if proba is None:
        st.stop()
    band, color = prob_to_band(float(proba), low=0.05, high=0.15)

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(proba) * 100.0,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": color},
                   "steps": [
                       {"range": [0, float(threshold) * 100.0], "color": "#ecf8f3"},
                       {"range": [float(threshold) * 100.0, 100], "color": "#fdecea"},
                   ]},
            title={"text": "Probabilité de défaut"},
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Décision (seuil {threshold:.3f})** : **{'Refus' if proba >= threshold else 'Accord'}**")
        st.markdown(f"Risque : **{band}**")
        st.caption(f"Source: **{source_label}**")

    with col2:
        st.markdown("**Contributions locales (Top 10)**")

        use_local_exp = st.toggle("Activer contributions locales (SHAP si possible, sinon delta)", value=True)

        def _bar_from_df(df: pd.DataFrame, title: str) -> go.Figure:
            tmp = df.copy().sort_values("abs_val").tail(10)
            x_vals = np.asarray(tmp["contrib"].values, dtype=float)
            y_vals = tmp["feature"].astype(str).tolist()
            hvr = [f"valeur: {v}" for v in tmp["value"]]
            figb = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h", hovertext=hvr, hoverinfo="text+x+y"))
            figb.update_layout(title=title, xaxis_title="Contribution ( + = ↑ risque )")
            # ligne verticale à 0
            figb.add_vline(x=0, line_dash="dot", line_width=1)
            return figb

        if use_local_exp and model_local is not None and not x_row.empty and not background.empty:
            contrib_df, method = compute_local_shap_or_delta(model_local, background, x_row, max_features=50)
            if contrib_df is not None and not contrib_df.empty:
                st.plotly_chart(_bar_from_df(contrib_df, f"Impact sur le score (positif = ↑ risque) — {method}"), use_container_width=True)
            else:
                st.info("Explicabilité locale indisponible ; affichage de l’importance globale.")
        else:
            st.info("Explicabilité locale désactivée ; affichage de l’importance globale.")

        if (not use_local_exp) or (contrib_df is None or contrib_df.empty):
            if global_imp_df is not None and not global_imp_df.empty:
                tmp = global_imp_df.head(10).copy()
                x_vals = np.asarray(tmp["importance"].values, dtype=float)
                y_vals = tmp["feature"].astype(str).tolist()
                figb = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h"))
                figb.update_layout(title="Top 10 — Importance globale")
                st.plotly_chart(figb, use_container_width=True)
            else:
                st.info("Importance globale indisponible.")

    st.divider()
    st.subheader("📄 Export")
    if not REPORTLAB_AVAILABLE:
        st.warning("Le module **reportlab** n'est pas installé. `pip install reportlab` puis relancez l'app.")
    else:
        try:
            client_id_str = str(selected_id) if selected_id is not None else "NA"
            pdf_bytes = build_client_report_pdf(
                client_id=client_id_str,
                model_name=f"{source_label}/{method}",
                threshold=float(threshold),
                proba=proba,
                x_row=x_row,
                X=X,
                pool_df=pool_df,
                global_imp_df=global_imp_df,
                shap_vals=(contrib_df if contrib_df is not None and not contrib_df.empty else None),
            )
            filename = f"fiche_client_{client_id_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("📄 Exporter la fiche client (PDF)", data=pdf_bytes, file_name=filename, mime="application/pdf",
                               use_container_width=True)
        except Exception as e:
            st.error(f"Échec de la génération du PDF : {e}")

# -------------------------------
# Tab 2 — Fiche client
# -------------------------------
with main_tabs[1]:
    st.subheader("Fiche client")
    if x_row.empty:
        st.info("Sélectionnez un client dans la barre latérale.")
    else:
        if global_imp_df is not None and not global_imp_df.empty:
            key_feats = [f for f in global_imp_df.head(20)["feature"].tolist() if f in X.columns]
        else:
            key_feats = list(X.columns)[:20]
        pretty = x_row[key_feats].T.reset_index()
        pretty.columns = ["Variable", "Valeur"]
        st.dataframe(pretty, use_container_width=True)

# -------------------------------
# Tab 3 — Comparaison
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
        default_coh = [c for c in candidate_cohorts[:2]]
        selected_cohorts = st.multiselect("Filtrer par attributs (cohorte similaire)", candidate_cohorts, default=default_coh)
        cohort_df = pool_df.copy()
        for c in selected_cohorts:
            cohort_df = cohort_df[cohort_df[c] == pool_df.loc[pool_df[ID_COL] == selected_id, c].iloc[0]]
        st.caption(f"Taille de la cohorte similaire : **{len(cohort_df):,}**")

        # features comparées: prendre jusqu'à 8 numériques parmi les plus importantes
        if global_imp_df is not None and not global_imp_df.empty:
            cand = [f for f in global_imp_df["feature"].tolist() if f in pool_df.columns]
        else:
            cand = [f for f in list(pool_df.columns)]
        comp_feats = []
        for f in cand:
            if pd.api.types.is_numeric_dtype(pool_df[f]) and pool_df[f].dropna().size > 0:
                comp_feats.append(f)
            if len(comp_feats) >= 8:
                break

        if not comp_feats:
            st.info("Aucune variable numérique comparable disponible.")
        else:
            long_rows = []
            for f in comp_feats:
                s_pop = pool_df[f].dropna()
                if s_pop.empty:
                    continue
                s_coh = cohort_df[f].dropna() if f in cohort_df.columns else None
                client_val = float(x_row[f].iloc[0]) if f in X.columns and pd.notnull(x_row[f].iloc[0]) else np.nan

                pop_q = s_pop.quantile([0.1, 0.5, 0.9]).values
                if s_coh is not None and not s_coh.empty:
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
                    x_p10 = np.asarray(sub["p10"].values, dtype=float)
                    x_p50 = np.asarray(sub["p50"].values, dtype=float)
                    x_p90 = np.asarray(sub["p90"].values, dtype=float)
                    x_cli = np.asarray(sub["client"].values, dtype=float)
                    y_feat = sub["feature"].astype(str).tolist()

                    figc = go.Figure()
                    figc.add_trace(go.Scatter(x=x_p10, y=y_feat, mode="markers", name="P10"))
                    figc.add_trace(go.Scatter(x=x_p50, y=y_feat, mode="markers", name="P50"))
                    figc.add_trace(go.Scatter(x=x_p90, y=y_feat, mode="markers", name="P90"))
                    figc.add_trace(go.Scatter(x=x_cli, y=y_feat, mode="markers", name="Client",
                                              marker=dict(symbol="diamond", size=12)))
                    figc.update_layout(title=f"{grp} — Positionnement du client (P10/P50/P90)", height=420)
                    st.plotly_chart(figc, use_container_width=True)

# -------------------------------
# Tab 4 — Qualité des données
# -------------------------------
with main_tabs[3]:
    st.subheader("Qualité des données & valeurs manquantes")
    # Image (si fournie)
    miss_fig = _pick_first_existing(["__results___5_1.png", "missing_train.png"])
    if miss_fig:
        st.image(miss_fig, caption="Top taux de valeurs manquantes (train)", use_column_width=True)
    else:
        st.info("Figure de valeurs manquantes non trouvée.")

    st.markdown("### Ratio : MISSING_COUNT_ROW")
    if not pool_df.empty:
        # On calcule sur les colonnes réellement présentes
        cols_present = list(pool_df.columns)
        # Ne pas inclure TARGET/ID
        cols_present = [c for c in cols_present if c not in {ID_COL, TARGET_COL}]
        s_missing = pool_df[cols_present].isna().sum(axis=1)
        # valeur client
        client_missing = int(s_missing.loc[pool_df[pool_df[ID_COL] == selected_id].index].iloc[0]) if selected_id is not None else None
        figm = go.Figure()
        hist = np.histogram(s_missing.values, bins=20)
        figm.add_trace(go.Bar(x=hist[1][:-1], y=hist[0], name="Nb lignes"))
        figm.update_layout(title="Distribution du nombre de valeurs manquantes par ligne",
                           xaxis_title="Nombre de NA par ligne", yaxis_title="Effectif", height=350)
        if client_missing is not None:
            figm.add_vline(x=client_missing, line_width=2, line_dash="dash", line_color="red",
                           annotation_text=f"Client: {client_missing} NA", annotation_position="top right")
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("Données indisponibles pour calculer MISSING_COUNT_ROW.")

# -------------------------------
# Tab 5 — Nouveau client
# -------------------------------
with main_tabs[4]:
    st.subheader("Comparer un nouveau client")
    st.markdown("Chargez un **CSV** (1 ligne) ou saisissez quelques variables clés pour simuler un nouveau client.")
    up = st.file_uploader("Fichier CSV (1 ligne)", type=["csv"], accept_multiple_files=False)
    topk = st.slider("Nombre de variables clés à saisir", min_value=5, max_value=40, value=15, step=1)
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
        cols_in = st.columns(2)
        inputs: Dict[str, Any] = {}
        with cols_in[0]:
            for f in num_cand:
                series = X[f]
                default = float(np.nanmedian(series.values)) if np.isfinite(np.nanmedian(series.values)) else 0.0
                inputs[f] = st.number_input(f, value=float(default))
        with cols_in[1]:
            for f in cat_cand:
                series = pd.Series(X[f].dropna().astype(str))
                mode = series.mode().iloc[0] if not series.empty else "NA"
                opts = sorted(series.unique().tolist()[:50]) or ["NA"]
                idx = opts.index(mode) if mode in opts else 0
                inputs[f] = st.selectbox(f, options=opts, index=idx)

        if st.button("Simuler"):
            new_x = pd.DataFrame([inputs])

    if new_x is not None:
        exp_cols = list(X.columns)
        for c in exp_cols:
            if c not in new_x.columns:
                new_x[c] = np.nan
        new_x = new_x[exp_cols].copy()
        st.session_state["last_new_x"] = new_x.copy()

        # Score local/API
        new_p, contrib_df2, method2 = None, None, "—"
        if mode == "API FastAPI" and api_ok:
            try:
                payload = {
                    "features": {k: (None if pd.isna(v) else v) for k, v in new_x.iloc[0].to_dict().items()},
                    "threshold": float(threshold), "shap": False
                }
                resp = requests.post(api_base.rstrip("/") + "/predict", json=payload, timeout=20)
                resp.raise_for_status()
                j = resp.json()
                new_p = float(j["proba_default"])
            except Exception as e:
                st.error(f"API KO: {e}")
                new_p = None
        else:
            if model_local is None:
                st.error("Modèle local indisponible.")
            else:
                try:
                    new_p = float(model_local.predict_proba(new_x)[0, 1])
                except Exception as e:
                    st.error(f"Échec de la prédiction: {e}")
                    new_p = None

        if new_p is not None:
            band2, color2 = prob_to_band(float(new_p), low=0.05, high=0.15)
            decision = "Refus" if float(new_p) >= float(threshold) else "Accord"

            fign = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(new_p) * 100.0,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color2},
                    "steps": [
                        {"range": [0, float(threshold) * 100.0], "color": "#ecf8f3"},
                        {"range": [float(threshold) * 100.0, 100], "color": "#fdecea"},
                    ],
                },
                title={"text": f"Probabilité de défaut (nouveau client) — {mode.split()[0]}"},
            ))
            st.plotly_chart(fign, use_container_width=True)
            st.markdown(f"**Décision (seuil {float(threshold):.3f})** : **{decision}**")
            st.markdown(f"Niveau de risque : **{band2}**")

            # Contributions locales (SHAP si possible ; sinon delta)
            if model_local is not None:
                contrib_df2, method2 = compute_local_shap_or_delta(model_local, background, new_x, max_features=50)

            with st.expander("🛠️ Axes d’amélioration (si décision = Refus)"):
                if contrib_df2 is not None and not contrib_df2.empty:
                    df_axes = contrib_df2[contrib_df2["contrib"] > 0].copy().head(10)
                    df_axes["Effet"] = "↑ risque"
                    df_axes = df_axes.rename(columns={"feature": "Variable", "value": "Valeur", "contrib": "Contribution"})
                    st.dataframe(df_axes[["Variable", "Valeur", "Contribution", "Effet"]], use_container_width=True)
                else:
                    st.info("Aucune recommandation spécifique (explicabilité locale indisponible).")

            with st.expander("🌟 Points forts"):
                if contrib_df2 is not None and not contrib_df2.empty:
                    df_pf = contrib_df2[contrib_df2["contrib"] < 0].copy().head(10)
                    df_pf["Effet"] = "↓ risque"
                    df_pf = df_pf.rename(columns={"feature": "Variable", "value": "Valeur", "contrib": "Contribution"})
                    st.dataframe(df_pf[["Variable", "Valeur", "Contribution", "Effet"]], use_container_width=True)
                else:
                    st.info("Non disponible (explicabilité locale indisponible).")

            st.divider()
            st.subheader("📄 Export (nouveau client)")
            if not REPORTLAB_AVAILABLE:
                st.warning("Le module **reportlab** n'est pas installé. `pip install reportlab` puis relancez l'app.")
            else:
                try:
                    pdf_new = build_new_client_report_pdf(
                        proba=float(new_p),
                        threshold=float(threshold),
                        decision=decision,
                        band_label=band2,
                        new_x=new_x,
                        X=X,
                        pool_df=pool_df,
                        global_imp_df=global_imp_df,
                        shap_df=contrib_df2
                    )
                    fname = f"nouveau_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button("📄 Télécharger le PDF (nouveau client)",
                                       data=pdf_new, file_name=fname, mime="application/pdf",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Échec génération PDF nouveau client : {e}")

# -------------------------------
# Tab 6 — Seuil & coût métier
# -------------------------------
with main_tabs[5]:
    st.subheader("Seuil & coût métier (optimisation)")
    cols_opt = st.columns(4)
    with cols_opt[0]:
        unit = st.selectbox("Unité monétaire", ["€", "CHF", "USD"], index=0)
    with cols_opt[1]:
        cost_fp = st.number_input("Coût d'un FP (refus à tort)", min_value=0.0, value=100.0, step=10.0)
    with cols_opt[2]:
        cost_fn = st.number_input("Coût d'un FN (acceptation risquée)", min_value=0.0, value=1000.0, step=10.0)
    with cols_opt[3]:
        max_sample = st.number_input("Taille échantillon (max)", min_value=1000, value=20000, step=1000)

    if model_local is None or X.empty or TARGET_COL is None:
        st.info("Pour optimiser le seuil en local, il faut : un **modèle local**, des **données** et la colonne **TARGET**.")
    else:
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
            y_all = df_lab[TARGET_COL].astype(int)

            if len(X_all) > max_sample:
                X_all = X_all.sample(int(max_sample), random_state=42)
                y_all = y_all.loc[X_all.index]

            try:
                p_all = model_local.predict_proba(X_all)[:, 1]
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
                    fig_roc.add_trace(go.Scatter(x=np.asarray(fpr, dtype=float),
                                                 y=np.asarray(tpr, dtype=float),
                                                 mode="lines", name=f"ROC (AUC={auc:.3f})"))
                    fig_roc.add_trace(go.Scatter(x=np.asarray([0,1], dtype=float),
                                                 y=np.asarray([0,1], dtype=float),
                                                 mode="lines", name="Random", line=dict(dash="dash")))
                    fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR", height=350)
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception:
                    pass

                try:
                    prec, rec, pr_th = precision_recall_curve(y_all.values, p_all)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=np.asarray(rec, dtype=float),
                                                y=np.asarray(prec, dtype=float),
                                                mode="lines", name="Precision-Recall"))
                    fig_pr.update_layout(title="Précision–Rappel", xaxis_title="Recall", yaxis_title="Precision", height=350)
                    st.plotly_chart(fig_pr, use_container_width=True)
                except Exception:
                    pass

                df_cost, best = cost_curve(y_all.values, p_all, cost_fp, cost_fn, step=0.001)
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(x=np.asarray(df_cost["threshold"].values, dtype=float),
                                              y=np.asarray(df_cost["cost"].values, dtype=float),
                                              mode="lines", name="Coût total"))
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

                apply_cols = st.columns([1,2])
                with apply_cols[0]:
                    if st.button("✅ Appliquer le seuil optimal au dashboard"):
                        st.session_state["threshold"] = float(best["threshold"])
                        st.success(f"Seuil mis à jour à {best['threshold']:.3f}.")
                        st.rerun()
                with apply_cols[1]:
                    st.caption("Le seuil optimal minimise le coût total attendu : `coût = FP × coût_FP + FN × coût_FN`.")

# -------------------------------
# Tab 7 — Dictionnaire des variables
# -------------------------------
with main_tabs[6]:
    st.subheader("Dictionnaire des variables (auto)")
    if pool_df.empty:
        st.info("Données indisponibles.")
    else:
        def french_label(name: str) -> str:
            # Traduction simple par règles (couvre tous les termes d'une manière lisible)
            rep = (
                ("AMT_", "Montant "),
                ("AMT", "Montant "),
                ("INCOME", "Revenu"),
                ("ANNUITY", "Mensualité"),
                ("GOODS", "Biens"),
                ("CREDIT", "Crédit"),
                ("DAYS", "Jours"),
                ("YEARS", "Années"),
                ("EXT_SOURCE", "Score externe"),
                ("EXT SOURCES", "Scores externes"),
                ("CNT", "Nombre"),
                ("FAM", "Famille"),
                ("REGION", "Région"),
                ("NAME", "Nom"),
                ("FLAG", "Drapeau"),
                ("OWN", "Possession"),
                ("REALTY", "Immobilier"),
                ("CAR", "Voiture"),
                ("EMPLOY", "Emploi"),
                ("OCCUPATION", "Métier"),
                ("ORGANIZATION", "Organisation"),
                ("WEEKDAY", "JourSemaine"),
                ("HOUR", "Heure"),
                ("HOUSETYPE", "TypeHabitation"),
                ("FONDKAPREMONT", "Fonds de réparation"),
                ("WALLSMATERIAL", "Matériaux murs"),
                ("EMERGENCYSTATE", "État d'urgence"),
                ("RATIO", "Ratio"),
                ("_MEAN", " (moyenne)"),
                ("_SUM", " (somme)"),
                ("_NA", " (nb NA)"),
                ("_BIN", " (bin)"),
            )
            s = name.upper()
            for a, b in rep:
                s = s.replace(a, b.upper())
            # espaces et casse douce
            s = s.replace("_", " ").strip().title()
            return s

        cols = [c for c in pool_df.columns if c not in {TARGET_COL}]
        df_dict = pd.DataFrame({
            "Variable": cols,
            "Libellé (FR)": [french_label(c) for c in cols],
        })
        st.dataframe(df_dict, use_container_width=True)

# -------------------------------
# Tab 8 — Ratios (feature engineering)
# -------------------------------
with main_tabs[7]:
    st.subheader("Ratios (calculs & interprétation)")
    if pool_df.empty:
        st.info("Données indisponibles.")
    else:
        # calculs pour le client sélectionné
        def compute_ratios(df_row: pd.Series) -> Dict[str, Any]:
            r: Dict[str, Any] = {}
            g = df_row.get

            # Âge
            if pd.notnull(g("AGE_YEARS")):
                r["AGE_YEARS"] = g("AGE_YEARS")
            elif pd.notnull(g("DAYS_BIRTH")):
                r["AGE_YEARS"] = round(abs(float(g("DAYS_BIRTH"))) / 365.25, 2)

            # Ancienneté emploi
            if pd.notnull(g("EMPLOY_YEARS")):
                r["EMPLOY_YEARS"] = g("EMPLOY_YEARS")
            elif pd.notnull(g("DAYS_EMPLOYED")):
                r["EMPLOY_YEARS"] = round(abs(float(g("DAYS_EMPLOYED"))) / 365.25, 2)

            # Ancienneté enregistrement
            if pd.notnull(g("REG_YEARS")):
                r["REG_YEARS"] = g("REG_YEARS")
            elif pd.notnull(g("DAYS_REGISTRATION")):
                r["REG_YEARS"] = round(abs(float(g("DAYS_REGISTRATION"))) / 365.25, 2)

            # Ratios financiers
            try:
                r["CREDIT_INCOME_RATIO"] = float(g("AMT_CREDIT")) / float(g("AMT_INCOME_TOTAL"))
            except Exception:
                pass
            try:
                r["ANNUITY_INCOME_RATIO"] = float(g("AMT_ANNUITY")) / float(g("AMT_INCOME_TOTAL"))
            except Exception:
                pass
            try:
                r["CREDIT_TERM_MONTHS"] = float(g("AMT_CREDIT")) / float(g("AMT_ANNUITY"))
            except Exception:
                pass
            try:
                r["PAYMENT_RATE"] = float(g("AMT_ANNUITY")) / float(g("AMT_CREDIT"))
            except Exception:
                pass
            try:
                r["CREDIT_GOODS_RATIO"] = float(g("AMT_CREDIT")) / float(g("AMT_GOODS_PRICE"))
            except Exception:
                pass

            # Scores externes groupés
            xs = []
            na_count = 0
            for k in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                val = g(k)
                if pd.notnull(val):
                    xs.append(float(val))
                else:
                    na_count += 1
            if xs:
                r["EXT_SOURCES_MEAN"] = float(np.mean(xs))
                r["EXT_SOURCES_SUM"]  = float(np.sum(xs))
            r["EXT_SOURCES_NA"] = int(na_count)

            # Ratios famille
            try:
                r["INCOME_PER_PERSON"] = float(g("AMT_INCOME_TOTAL")) / float(g("CNT_FAM_MEMBERS"))
            except Exception:
                pass
            try:
                r["CHILDREN_RATIO"] = float(g("CNT_CHILDREN")) / float(g("CNT_FAM_MEMBERS"))
            except Exception:
                pass

            # Bool possession
            vcar = g("FLAG_OWN_CAR")
            if pd.notnull(vcar):
                r["OWN_CAR_BOOL"] = 1 if str(vcar).upper() in {"Y", "TRUE", "1"} else 0
            vrea = g("FLAG_OWN_REALTY")
            if pd.notnull(vrea):
                r["OWN_REALTY_BOOL"] = 1 if str(vrea).upper() in {"Y", "TRUE", "1"} else 0

            # Emploi/Âge
            if "EMPLOY_YEARS" in r and "AGE_YEARS" in r and r["AGE_YEARS"]:
                try:
                    r["EMPLOY_TO_AGE_RATIO"] = float(r["EMPLOY_YEARS"]) / float(r["AGE_YEARS"])
                except Exception:
                    pass

            # Missing row count (sur colonnes présentes)
            cols_present = [c for c in pool_df.columns if c not in {ID_COL, TARGET_COL}]
            r["MISSING_COUNT_ROW"] = int(pd.isna(df_row[cols_present]).sum())

            return r

        # affichage client
        if selected_id is None:
            st.info("Sélectionnez d’abord un client.")
        else:
            base_row = pool_df.set_index(ID_COL).loc[selected_id]
            ratios_cli = compute_ratios(base_row)
            df_cli = pd.DataFrame([ratios_cli]).T.reset_index()
            df_cli.columns = ["Ratio", "Valeur (client)"]
            st.markdown("**Client sélectionné**")
            st.dataframe(df_cli, use_container_width=True)

        # affichage nouveau client si dispo
        st.markdown("---")
        st.markdown("**Nouveau client (si saisi sur l’onglet dédié)**")
        if "last_new_x" in st.session_state:
            new_row = st.session_state["last_new_x"].iloc[0]
            ratios_new = compute_ratios(new_row)
            df_new = pd.DataFrame([ratios_new]).T.reset_index()
            df_new.columns = ["Ratio", "Valeur (nouveau client)"]
            st.dataframe(df_new, use_container_width=True)
        else:
            st.info("Aucun nouveau client saisi pour le moment.")

# Footer
st.divider()
footer_cols = st.columns([2,2,1])
with footer_cols[0]:
    st.caption("© Prêt à dépenser — Dashboard pédagogique. Transparence & explicabilité des décisions d'octroi.")
with footer_cols[1]:
    st.caption("App version: " + APP_VERSION)
with footer_cols[2]:
    st.caption("Build: Streamlit + SHAP + scikit-learn")
