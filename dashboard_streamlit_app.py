# Streamlit Credit Scoring Dashboard — "Prêt à dépenser" (v1.4.0)
# ----------------------------------------------------------------
# Run:
#   python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

APP_VERSION = "1.4.0"

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

def fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)

def fmt_num(x, decimals=2) -> str:
    try:
        if pd.isna(x):
            return "—"
        s = f"{float(x):,.{decimals}f}"
        return s.replace(",", " ").replace(".", ",")
    except Exception:
        return str(x)

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

# ---------- SHAP local (robuste avec fallback permutation) ----------
def compute_local_shap(estimator, X_background: pd.DataFrame, x_row: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    import shap
    # Échantillon de fond limité
    bg = X_background.copy()
    if len(bg) > 200:
        bg = bg.sample(200, random_state=42)

    # Fonction modèle : garantit l'ordre & les noms de colonnes attendues
    expected_cols = list(bg.columns)

    def f(Xdf):
        import pandas as pd
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf, columns=expected_cols)
        else:
            Xdf = Xdf.reindex(columns=expected_cols)
        return estimator.predict_proba(Xdf)[:, 1]

    # 1) Tentative standard (masker Independent)
    try:
        masker = shap.maskers.Independent(bg)
        explainer = shap.Explainer(f, masker, feature_names=expected_cols)
        ex = explainer(x_row.reindex(columns=expected_cols))
        return np.array(ex.values).reshape(-1), np.array(ex.base_values).reshape(-1)
    except Exception:
        pass

    # 2) Fallback robuste : Permutation explainer (peu sensible aux dtypes)
    try:
        explainer_perm = shap.explainers.Permutation(f, bg)
        ex = explainer_perm(x_row.reindex(columns=expected_cols))
        return np.array(ex.values).reshape(-1), np.array(ex.base_values).reshape(-1)
    except Exception as e2:
        raise RuntimeError(f"SHAP indisponible (y compris permutation): {e2}")

# ---------- Utils quantiles / cohortes ----------
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

# -------------------------------
# API helpers
# -------------------------------
def api_health(base_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=4)
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def api_predict(base_url: str, payload: dict) -> dict:
    r = requests.post(base_url.rstrip("/") + "/predict", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def api_metrics(base_url: str, payload: dict) -> dict:
    r = requests.post(base_url.rstrip("/") + "/metrics", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# -------------------------------
# PDF builders
# -------------------------------
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
    story.append(t2)
    story.append(Spacer(1, 8))

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
    story.append(t3)
    story.append(Spacer(1, 8))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# ---- Suggestions "nouveau client" ----
def suggest_actions(
    shap_df: Optional[pd.DataFrame],
    new_x: pd.DataFrame,
    X: pd.DataFrame,
    pool_df: pd.DataFrame,
    top_n:int = 5,
    global_imp_df: Optional[pd.DataFrame] = None,
) -> tuple[list[dict], list[dict]]:
    def _quantiles_for(f: str):
        s = None
        if f in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[f]):
            s = pool_df[f]
        elif f in X.columns and pd.api.types.is_numeric_dtype(X[f]):
            s = X[f]
        if s is None or s.dropna().empty:
            return None
        q = s.quantile([0.1, 0.5, 0.9])
        return {"p10": float(q.loc[0.1]), "p50": float(q.loc[0.5]), "p90": float(q.loc[0.9])}

    RULES = {
        "CREDIT_GOODS_RATIO": "Un ratio crédit/biens plus faible (apport initial plus élevé) réduit le risque.",
        "ANNUITY_INCOME_RATIO": "Réduire la mensualité par rapport au revenu (renégocier durée/montant) améliore la solvabilité.",
        "CREDIT_INCOME_RATIO": "Un endettement plus faible (crédit/revenu) est attendu chez les bons dossiers.",
        "PAYMENT_RATE": "Un taux de mensualité plus faible par rapport au crédit peut alléger la pression budgétaire.",
        "EXT_SOURCE_1": "Scores externes non actionnables directement ; maintenir un historique de paiement sain.",
        "EXT_SOURCE_2": "Renforcer l’historique de paiement (zéro retard) améliore ce signal externe.",
        "EXT_SOURCE_3": "Même idée : régularité des paiements, pas d’incidents, soldes à temps.",
        "DAYS_EMPLOYED": "Une ancienneté plus élevée stabilise le profil (éviter les ruptures d’emploi si possible).",
        "EMPLOY_TO_AGE_RATIO": "Un ratio emploi/âge plus élevé reflète une carrière plus stable.",
    }

    def _mk_note(feat:str, val:Any, shap_pos:bool, q:Optional[dict]):
        base = RULES.get(str(feat))
        if base:
            return base
        if q is None:
            return "Contribution estimée vs profil moyen."
        try:
            v = float(val)
        except Exception:
            v = None
        if v is None:
            return "Contribution liée à la modalité observée."
        if shap_pos:
            if v >= q["p50"]:
                return f"Valeur au-dessus de la médiane ({q['p50']:.2f}). Se rapprocher de la médiane peut réduire le risque."
            else:
                return f"Valeur en-dessous de la médiane ({q['p50']:.2f}). Se rapprocher de la médiane peut réduire le risque."
        else:
            pos_txt = "au-dessus" if v >= q["p50"] else "en-dessous"
            return f"Point fort : valeur {pos_txt} de la médiane ({q['p50']:.2f})."

    if shap_df is not None and not shap_df.empty and not new_x.empty:
        tmp = shap_df.copy().sort_values("abs_val", ascending=False)
        pos = tmp[tmp["shap_value"] > 0].head(top_n)
        neg = tmp[tmp["shap_value"] < 0].head(top_n)
        axes, strong = [], []
        row = new_x.iloc[0]
        for _, r in pos.iterrows():
            f = str(r["feature"]); v = row[f] if f in row.index else np.nan; q = _quantiles_for(f)
            axes.append({"feature": f, "value": v, "note": _mk_note(f, v, True,  q)})
        for _, r in neg.iterrows():
            f = str(r["feature"]); v = row[f] if f in row.index else np.nan; q = _quantiles_for(f)
            strong.append({"feature": f, "value": v, "note": _mk_note(f, v, False, q)})
        return axes, strong

    axes, strong = [], []
    if global_imp_df is None or global_imp_df.empty or new_x.empty:
        return axes, strong

    row = new_x.iloc[0]
    HIGH_IS_RISK = {"CREDIT_GOODS_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_INCOME_RATIO",
                    "AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON",
                    "AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR","EXT_SOURCES_NA","DOC_COUNT"}
    LOW_IS_RISK  = {"EXT_SOURCES_MEAN","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","EMPLOY_TO_AGE_RATIO","PAYMENT_RATE"}

    cand = [f for f in global_imp_df["feature"].tolist() if (f in X.columns or f in pool_df.columns)]
    cand = cand[: max(top_n*3, 15)]

    scored = []
    for f in cand:
        s = None
        if f in pool_df.columns and pd.api.types.is_numeric_dtype(pool_df[f]):
            s = pool_df[f]
        elif f in X.columns and pd.api.types.is_numeric_dtype(X[f]):
            s = X[f]
        if s is None or s.dropna().empty:
            continue
        q = s.quantile([0.1, 0.5, 0.9])
        qd = {"p10": float(q.loc[0.1]), "p50": float(q.loc[0.5]), "p90": float(q.loc[0.9])}
        v = row[f] if f in row.index else np.nan
        try:
            v_float = float(v)
        except Exception:
            continue
        spread = max(qd["p90"] - qd["p10"], 1e-9)
        deviation = abs(v_float - qd["p50"]) / spread
        if f in HIGH_IS_RISK:
            shap_pos_guess = (v_float > qd["p50"])
        elif f in LOW_IS_RISK:
            shap_pos_guess = (v_float < qd["p50"])
        else:
            shap_pos_guess = (v_float > qd["p90"] or v_float < qd["p10"])
        scored.append((f, v, qd, deviation, shap_pos_guess))

    scored.sort(key=lambda t: t[3], reverse=True)
    for f, v, qd, _, shap_pos_guess in scored[:top_n]:
        if shap_pos_guess:
            axes.append({"feature": f, "value": v, "note": _mk_note(f, v, True,  qd)})
        else:
            strong.append({"feature": f, "value": v, "note": _mk_note(f, v, False, qd)})

    if not axes:
        forced = []
        for f, v, qd, _, _ in scored:
            try:
                vf = float(v)
            except Exception:
                continue
            if (f in HIGH_IS_RISK and vf >= qd["p50"]) or (f in LOW_IS_RISK and vf <= qd["p50"]):
                forced.append({"feature": f, "value": v, "note": _mk_note(f, v, True, qd)})
                if len(forced) >= min(3, top_n):
                    break
        if not forced:
            for f, v, qd, _, _ in scored[:min(3, top_n)]:
                forced.append({"feature": f, "value": v, "note": _mk_note(f, v, True, qd)})
        axes = forced

    return axes, strong

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
    path, mtime_str, sha8, git = _runtime_info()
    st.caption(f"Fichier: {os.path.basename(path)}")
    st.caption(f"Dernière modif: {mtime_str}")
    st.caption(f"SHA fichier: {sha8} | Git: {git}")

    if st.button("🔄 Forcer rechargement (vider cache)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.subheader("Mode de scoring")
    mode = st.radio("Choisir le mode", ["Local (modèle embarqué)", "API FastAPI"], index=0)
    api_base = None
    api_ok = False
    if mode == "API FastAPI":
        api_base = st.text_input("API base URL", value="http://localhost:8000")
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
    default_thresh = float(np.clip(st.session_state.get("threshold", 0.67), 0.0, 1.0))
    threshold = st.slider(
        "Seuil d'acceptation (proba défaut)",
        0.0, 1.0, float(default_thresh), 0.001, key="threshold",
        help="Au-delà du seuil = risque élevé ⇒ refus")

    st.subheader("Sélection du client")
    id_options = pool_df[ID_COL].tolist() if (ID_COL and not pool_df.empty) else []
    selected_id = st.selectbox("SK_ID_CURR", id_options, index=0 if id_options else None)

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

# Préparer X aligné pour affichages
if not pool_df.empty and selected_id is not None:
    df_idx = pool_df.set_index(ID_COL)
    expected_cols_local = get_expected_input_columns(model_local) if model_local is not None else (feature_names or list(df_idx.columns))
    for c in expected_cols_local:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    X = df_idx[expected_cols_local]
    if selected_id not in X.index:
        st.stop()
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
    "🧪 Qualité des données",
    "🆕 Nouveau client",
    "📚 Dictionnaire des variables",
    "⚙️ Ratios (feature engineering)",
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

    # PREDICT
    if mode == "API FastAPI":
        if not api_base or not api_ok:
            st.error("API indisponible.")
        else:
            try:
                payload = {"client_id": selected_id, "threshold": float(threshold), "shap": True, "topk": 10}
                resp = api_predict(api_base, payload)
                proba = float(resp["proba_default"])
                source_label = "API"
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
                st.warning(f"API KO ({e}). Bascule en Local si possible.")
                if model_local is None or x_row.empty:
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

        def _bar_from_df(df: pd.DataFrame, title: str) -> go.Figure:
            tmp = df.copy().sort_values("abs_val").tail(10)
            x_vals = np.asarray(tmp["shap_value"].values, dtype=float)
            y_vals = tmp["feature"].astype(str).tolist()
            hover = [f"valeur: {v}" for v in tmp["value"]]
            figb = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h", hovertext=hover, hoverinfo="text+x+y"))
            figb.update_layout(title=title)
            figb.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
            return figb

        if mode == "API FastAPI" and shap_df is not None and not shap_df.empty:
            st.plotly_chart(_bar_from_df(shap_df, "Impact sur le score (positif = ↑ risque)"), use_container_width=True)
        else:
            shap_enabled = st.toggle("Activer SHAP local (expérimental)", value=False)
            if shap_enabled and not background.empty and model_local is not None:
                try:
                    vals, base_vals = compute_local_shap(model_local, background, x_row)
                    shap_df_local = pd.DataFrame({
                        "feature": list(X.columns),
                        "shap_value": vals,
                        "abs_val": np.abs(vals),
                        "value": x_row.iloc[0].values,
                    }).sort_values("abs_val", ascending=False)
                    st.plotly_chart(_bar_from_df(shap_df_local, "Impact sur le score (positif = ↑ risque)"),
                                    use_container_width=True)
                except Exception as e:
                    if global_imp_df is not None and not global_imp_df.empty:
                        st.info("Explicabilité locale indisponible. Affichage de l'importance globale.")
                        tmp = global_imp_df.head(10).copy()
                        x_vals = np.asarray(tmp["importance"].values, dtype=float)
                        y_vals = tmp["feature"].astype(str).tolist()
                        figb = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h"))
                        figb.update_layout(title="Top 10 — Importance globale")
                        st.plotly_chart(figb, use_container_width=True)
                    else:
                        st.info("Importance globale indisponible.")
            else:
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
                model_name=f"{source_label}",
                threshold=float(threshold),
                proba=proba,
                x_row=x_row,
                X=X,
                pool_df=pool_df,
                global_imp_df=global_imp_df,
                shap_vals=(shap_df if shap_df is not None and not shap_df.empty else None),
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
        if global_imp_df is not None:
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
            try:
                cohort_df = cohort_df[cohort_df[c] == pool_df.loc[pool_df[ID_COL] == selected_id, c].iloc[0]]
            except Exception:
                pass

        st.caption(f"Taille de la cohorte similaire : **{fmt_int(len(cohort_df))}**")

        # Choix de 8 variables numériques montrables
        if global_imp_df is not None and not global_imp_df.empty:
            cand = [f for f in global_imp_df["feature"].tolist() if f in pool_df.columns]
        else:
            cand = [f for f in pool_df.columns]
        num_feats = [f for f in cand if pd.api.types.is_numeric_dtype(pool_df[f])][:8]

        if not num_feats:
            st.info("Aucune variable numérique comparable disponible.")
        else:
            long_rows = []
            for f in num_feats:
                s_pop = pd.to_numeric(pool_df[f], errors="coerce")
                if s_pop.dropna().empty:
                    continue
                s_coh = pd.to_numeric(cohort_df[f], errors="coerce") if f in cohort_df.columns else pd.Series(dtype=float)
                try:
                    client_val = float(X.at[selected_id, f]) if f in X.columns and pd.notnull(X.at[selected_id, f]) else np.nan
                except Exception:
                    client_val = np.nan
                pop_q = s_pop.quantile([0.1, 0.5, 0.9]).values
                coh_q = s_coh.quantile([0.1, 0.5, 0.9]).values if not s_coh.dropna().empty else [np.nan, np.nan, np.nan]
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
                    figc.add_trace(go.Scatter(x=x_p10, y=y_feat, mode="markers", name="P10",
                                              hovertemplate="Variable: %{y}<br>P10: %{x:.2f}<extra></extra>"))
                    figc.add_trace(go.Scatter(x=x_p50, y=y_feat, mode="markers", name="P50",
                                              hovertemplate="Variable: %{y}<br>P50: %{x:.2f}<extra></extra>"))
                    figc.add_trace(go.Scatter(x=x_p90, y=y_feat, mode="markers", name="P90",
                                              hovertemplate="Variable: %{y}<br>P90: %{x:.2f}<extra></extra>"))
                    figc.add_trace(go.Scatter(x=x_cli, y=y_feat, mode="markers", name="Client",
                                              marker=dict(symbol="diamond", size=12),
                                              hovertemplate="Variable: %{y}<br>Client: %{x:.2f}<extra></extra>"))
                    figc.update_layout(title=f"{grp} — Positionnement du client (P10/P50/P90)",
                                       height=420, separators=", ")
                    st.plotly_chart(figc, use_container_width=True)

# -------------------------------
# Tab 4 — Qualité des données
# -------------------------------
with main_tabs[3]:
    st.subheader("Qualité des données & valeurs manquantes")
    miss_fig = _pick_first_existing(["__results___5_1.png", "missing_train.png"])
    if miss_fig:
        st.image(miss_fig, caption="Top taux de valeurs manquantes (train)")
    else:
        st.info("Figure de valeurs manquantes non trouvée.")

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
        cols = st.columns(2)
        inputs: Dict[str, Any] = {}
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
                idx = opts.index(mode) if mode in opts else 0
                inputs[f] = st.selectbox(f, options=opts, index=idx)

        if st.button("Simuler"):
            new_x = pd.DataFrame([inputs])

    if new_x is not None:
        exp_cols = list(X.columns)
        for c in exp_cols:
            if c not in new_x.columns:
                new_x[c] = np.nan
        new_x = new_x[exp_cols]

        # On mémorise la dernière ligne "nouveau client" pour la page Ratios
        st.session_state["last_new_client_row"] = new_x.copy()

        if mode == "API FastAPI":
            if not api_base or not api_ok:
                st.error("API indisponible pour scorer le nouveau client.")
                new_p, shap_df2 = None, None
            else:
                try:
                    payload = {
                        "features": {k: (None if pd.isna(v) else v) for k, v in new_x.iloc[0].to_dict().items()},
                        "threshold": float(threshold), "shap": True, "topk": 10
                    }
                    resp = api_predict(api_base, payload)
                    new_p = float(resp["proba_default"])
                    shap_rows = resp.get("top_contrib") or []
                    shap_df2 = (pd.DataFrame([{
                        "feature": r["feature"],
                        "shap_value": float(r["shap_value"]),
                        "abs_val": abs(float(r["shap_value"])),
                        "value": r["value"],
                    } for r in shap_rows]) if shap_rows else None)
                except Exception as e:
                    st.error(f"API KO: {e}")
                    new_p, shap_df2 = None, None
        else:
            if model_local is None:
                st.error("Modèle local indisponible.")
                new_p, shap_df2 = None, None
            else:
                try:
                    new_p = float(model_local.predict_proba(new_x)[0, 1])
                    try:
                        vals, _ = compute_local_shap(model_local, background, new_x)
                        shap_df2 = pd.DataFrame({
                            "feature": list(X.columns),
                            "shap_value": vals,
                            "abs_val": np.abs(vals),
                            "value": new_x.iloc[0].values,
                        }).sort_values("abs_val", ascending=False).head(10)
                    except Exception:
                        shap_df2 = None
                except Exception as e:
                    st.error(f"Échec de la prédiction: {e}")
                    new_p, shap_df2 = None, None

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

            used_fallback = (shap_df2 is None) or shap_df2.empty
            axes, strong = suggest_actions(shap_df2, new_x, X, pool_df, top_n=5, global_imp_df=global_imp_df)

            if used_fallback:
                st.info("Explicabilité locale indisponible ; recommandations basées sur l’importance globale.")

            with st.expander("🛠️ Axes d’amélioration (si décision = Refus)"):
                if axes:
                    st.dataframe(pd.DataFrame([{
                        "Variable": a["feature"],
                        "Valeur": ("" if pd.isna(a["value"]) else a["value"]),
                        "Recommandation": a["note"],
                    } for a in axes]), use_container_width=True)
                else:
                    st.info("Aucune recommandation spécifique (explicabilité locale indisponible).")

            with st.expander("🌟 Points forts"):
                if strong:
                    st.dataframe(pd.DataFrame([{
                        "Variable": s["feature"],
                        "Valeur": ("" if pd.isna(s["value"]) else s["value"]),
                        "Commentaire": s["note"],
                    } for s in strong]), use_container_width=True)
                else:
                    st.info("Non disponible (explicabilité locale indisponible).")

            if shap_df2 is not None and not shap_df2.empty:
                tmp = shap_df2.copy().sort_values("abs_val", ascending=False).head(10)
                tmp = tmp.assign(
                    Effet=tmp["shap_value"].apply(lambda v: "↑ risque" if v > 0 else ("↓ risque" if v < 0 else "neutre")),
                    SHAP=tmp["shap_value"].map(lambda v: f"{v:+.4f}"),
                    Valeur=tmp["value"]
                )[["feature", "Valeur", "SHAP", "Effet"]]
                tmp.columns = ["Variable", "Valeur", "SHAP", "Effet"]
                st.markdown("**Contributions locales (SHAP) — top 10**")
                st.dataframe(tmp, use_container_width=True)

                tmp2 = shap_df2.copy().sort_values("abs_val").tail(10)
                x_vals = np.asarray(tmp2["shap_value"].values, dtype=float)
                y_vals = tmp2["feature"].astype(str).tolist()
                figb2 = go.Figure(go.Bar(x=x_vals, y=y_vals, orientation="h"))
                figb2.update_layout(title="Impact sur le score (positif = ↑ risque)")
                figb2.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                st.plotly_chart(figb2, use_container_width=True)

            st.divider()
            st.subheader("📄 Export (nouveau client)")
            if not REPORTLAB_AVAILABLE:
                st.warning("Le module **reportlab** n'est pas installé. `pip install reportlab` puis relancez l'app.")
            else:
                try:
                    pdf_new = build_client_report_pdf(
                        client_id="Nouveau client",
                        model_name=f"{mode.split()[0]}",
                        threshold=float(threshold),
                        proba=float(new_p),
                        x_row=new_x,
                        X=X,
                        pool_df=pool_df,
                        global_imp_df=global_imp_df,
                        shap_vals=(shap_df2 if shap_df2 is not None and not shap_df2.empty else None),
                    )
                    fname = f"nouveau_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button("📄 Télécharger le PDF (nouveau client)",
                                       data=pdf_new, file_name=fname, mime="application/pdf",
                                       use_container_width=True)
                except Exception as e:
                    st.error(f"Échec génération PDF nouveau client : {e}")

# -------------------------------
# Tab 6 — Dictionnaire des variables (traduction TOUTES colonnes)
# -------------------------------
# Dicos spécifiques (prioritaires)
FEATURE_LABELS_SPEC = {
    "SK_ID_CURR": "Identifiant client",
    "CODE_GENDER": "Sexe",
    "FLAG_OWN_CAR": "Possède une voiture",
    "FLAG_OWN_REALTY": "Possède un bien immobilier",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "NAME_HOUSING_TYPE": "Type de logement",
    "WEEKDAY_APPR_PROCESS_START": "Jour de la demande",
    "ORGANIZATION_TYPE": "Secteur employeur",
    "AGE_YEARS": "Âge (années)",
    "EMPLOY_YEARS": "Ancienneté (années)",
    "REG_YEARS": "Ancienneté à l'adresse (années)",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Mensualité du crédit",
    "AMT_GOODS_PRICE": "Prix des biens financés",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "EXT_SOURCES_MEAN": "Moyenne des scores externes",
    "EXT_SOURCES_SUM": "Somme des scores externes",
    "EXT_SOURCES_NA": "Nb. scores externes manquants",
    "PAYMENT_RATE": "Mensualité / Crédit",
    "CREDIT_INCOME_RATIO": "Crédit / Revenu",
    "ANNUITY_INCOME_RATIO": "Mensualité / Revenu",
    "CREDIT_GOODS_RATIO": "Crédit / Biens",
    "EMPLOY_TO_AGE_RATIO": "Ancienneté / Âge",
    "INCOME_PER_PERSON": "Revenu par personne",
    "CHILDREN_RATIO": "Enfants / Ménage",
    "DOC_COUNT": "Nb. documents fournis",
    "MISSING_COUNT_ROW": "Nb. champs manquants (ligne)",
    "HOUSETYPE_MODE": "Type de logement (mode)",
    "WALLSMATERIAL_MODE": "Matériaux des murs (mode)",
    "FONDKAPREMONT_MODE": "Fonds de rénovation (mode)",
    "NAME_TYPE_SUITE": "Accompagnant",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "REGION_RATING_CLIENT": "Indice région (rating)",
    "AGE_BIN": "Tranche d'âge",
}

# Heuristiques génériques (appliquées si non dans SPEC)
def french_label(col: str) -> str:
    if col in FEATURE_LABELS_SPEC:
        return FEATURE_LABELS_SPEC[col]
    c = col.upper()

    # Patterns courants
    repl = [
        ("AMT_", "Montant "),
        ("CNT_", "Nombre "),
        ("DAYS_", "Jours "),
        ("YEARS", "Années"),
        ("HOUR", "Heure"),
        ("MIN", "Minute"),
        ("SEC", "Seconde"),
        ("FLAG_", "Indicateur "),
        ("NAME_", "Libellé "),
        ("EXT_SOURCE_", "Score externe "),
        ("EXT_SOURCES_", "Scores externes "),
        ("REGION_", "Région "),
        ("ORGANIZATION_TYPE", "Secteur employeur"),
        ("WEEKDAY_APPR_PROCESS_START", "Jour de la demande"),
        ("SK_ID", "Identifiant "),
    ]
    label = c
    for a, b in repl:
        label = label.replace(a, b)
    # Nettoyage
    label = label.replace("__", " ").replace("_", " ").strip()
    # Titrage
    label = label.capitalize()
    # Ajustements sémantiques
    label = label.replace("Amt ", "Montant ")
    label = label.replace("Cnt ", "Nombre ")
    label = label.replace("Indicateur own car", "Possède une voiture")
    label = label.replace("Indicateur own realty", "Possède un bien immobilier")
    label = label.replace("Libellé education type", "Niveau d'éducation")
    label = label.replace("Libellé income type", "Type de revenu")
    label = label.replace("Libellé family status", "Situation familiale")
    label = label.replace("Libellé housing type", "Type de logement")
    label = label.replace("Score externe 1", "Score externe 1")
    label = label.replace("Score externe 2", "Score externe 2")
    label = label.replace("Score externe 3", "Score externe 3")
    return label

with main_tabs[5]:
    st.subheader("Dictionnaire des variables")
    if pool_df.empty:
        st.info("Données indisponibles.")
    else:
        rows = []
        for c in pool_df.columns:
            rows.append({
                "Variable": c,
                "Nom (FR)": french_label(c),
                "Description": "—"  # à enrichir si besoin
            })
        dict_df = pd.DataFrame(rows)
        st.dataframe(dict_df.sort_values("Variable"), use_container_width=True)

# -------------------------------
# Tab 7 — Ratios (feature engineering) avec calculs
# -------------------------------
def _safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
        if b == 0 or pd.isna(a) or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _to_years(days):
    try:
        return max(0.0, -float(days)) / 365.25
    except Exception:
        return np.nan

def compute_ratios_for_row(row: pd.Series) -> pd.DataFrame:
    # On tolère valeurs manquantes; on renvoie un tableau clair
    vals = {}
    g = lambda k: row.get(k, np.nan)

    vals["PAYMENT_RATE"]        = _safe_div(g("AMT_ANNUITY"), g("AMT_CREDIT"))
    vals["CREDIT_INCOME_RATIO"] = _safe_div(g("AMT_CREDIT"), g("AMT_INCOME_TOTAL"))
    vals["ANNUITY_INCOME_RATIO"]= _safe_div(g("AMT_ANNUITY"), g("AMT_INCOME_TOTAL"))
    vals["CREDIT_GOODS_RATIO"]  = _safe_div(g("AMT_CREDIT"), g("AMT_GOODS_PRICE"))

    # Scores externes
    s1, s2, s3 = g("EXT_SOURCE_1"), g("EXT_SOURCE_2"), g("EXT_SOURCE_3")
    ext_list = [x for x in [s1, s2, s3] if pd.notna(x)]
    vals["EXT_SOURCES_MEAN"] = (np.mean(ext_list) if ext_list else np.nan)
    vals["EXT_SOURCES_SUM"]  = (np.sum(ext_list)  if ext_list else np.nan)
    vals["EXT_SOURCES_NA"]   = 3 - len(ext_list)

    # Âge et ancienneté
    age_years   = g("AGE_YEARS")
    if pd.isna(age_years) and "DAYS_BIRTH" in row.index:
        age_years = _to_years(g("DAYS_BIRTH"))
    employ_years = g("EMPLOY_YEARS")
    if pd.isna(employ_years) and "DAYS_EMPLOYED" in row.index:
        employ_years = _to_years(g("DAYS_EMPLOYED"))
    vals["AGE_YEARS"]   = age_years
    vals["EMPLOY_YEARS"]= employ_years
    vals["EMPLOY_TO_AGE_RATIO"] = _safe_div(employ_years, age_years)

    # Revenu par personne & ratio enfants
    vals["INCOME_PER_PERSON"] = _safe_div(g("AMT_INCOME_TOTAL"), g("CNT_FAM_MEMBERS"))
    vals["CHILDREN_RATIO"]    = _safe_div(g("CNT_CHILDREN"), g("CNT_FAM_MEMBERS"))

    # DOC & MISSING s'ils existent
    vals["DOC_COUNT"]         = g("DOC_COUNT") if "DOC_COUNT" in row.index else np.nan
    vals["MISSING_COUNT_ROW"] = g("MISSING_COUNT_ROW") if "MISSING_COUNT_ROW" in row.index else np.nan

    # Mise en forme
    out = []
    for k, v in vals.items():
        out.append({
            "Variable": k,
            "Valeur": v,
            "Interprétation": {
                "PAYMENT_RATE": "Part de la mensualité dans le crédit (faible = moindre pression).",
                "CREDIT_INCOME_RATIO": "Charge du crédit vs revenu.",
                "ANNUITY_INCOME_RATIO": "Mensualité vs revenu.",
                "CREDIT_GOODS_RATIO": "≈1 sans apport ; <1 avec apport.",
                "EXT_SOURCES_MEAN": "Moyenne des scores externes.",
                "EXT_SOURCES_SUM": "Somme des scores externes.",
                "EXT_SOURCES_NA": "Nombre de scores externes manquants.",
                "AGE_YEARS": "Âge en années.",
                "EMPLOY_YEARS": "Ancienneté pro en années.",
                "EMPLOY_TO_AGE_RATIO": "Part de la vie passée en emploi.",
                "INCOME_PER_PERSON": "Revenu par personne.",
                "CHILDREN_RATIO": "Charge enfants / ménage.",
                "DOC_COUNT": "Nombre de documents fournis.",
                "MISSING_COUNT_ROW": "Champs manquants dans le dossier.",
            }.get(k, "—")
        })
    df = pd.DataFrame(out)
    # Formatage lisible (valeurs petites en décimales, grosses avec espace)
    def _fmt(v):
        if pd.isna(v):
            return "—"
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if abs(fv) >= 1000:
            return fmt_num(fv, 0)
        return fmt_num(fv, 4)
    df["Valeur"] = df["Valeur"].map(_fmt)
    return df

with main_tabs[6]:
    st.subheader("Ratios & variables dérivées (calculs réels)")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Client sélectionné**")
        if x_row.empty:
            st.info("Sélectionnez un client dans la barre latérale.")
        else:
            df_rat_cli = compute_ratios_for_row(x_row.iloc[0])
            st.dataframe(df_rat_cli, use_container_width=True)
    with colB:
        st.markdown("**Dernier “Nouveau client” saisi**")
        if "last_new_client_row" not in st.session_state:
            st.info("Aucun nouveau client saisi pour l’instant (utilisez l’onglet *Nouveau client*).")
        else:
            try:
                df_newrow = st.session_state["last_new_client_row"]
                df_rat_new = compute_ratios_for_row(df_newrow.iloc[0])
                st.dataframe(df_rat_new, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossible d'afficher les ratios du nouveau client : {e}")

# -------------------------------
# Tab 8 — Seuil & coût métier (inchangé)
# -------------------------------
with main_tabs[7]:
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

    if mode == "API FastAPI" and api_ok:
        try:
            payload = {"cost_fp": float(cost_fp), "cost_fn": float(cost_fn), "max_sample": int(max_sample), "step": 0.001}
            m = api_metrics(api_base, payload)
            auc = float(m.get("auc", float("nan")))
            roc = m.get("roc", {})
            pr  = m.get("pr", {})
            cc  = m.get("cost_curve", {})
            st.caption(f"Échantillon scoré côté API : **{fmt_int(m.get('n_scored', 0))}** lignes")

            if roc.get("fpr") and roc.get("tpr"):
                fpr = np.asarray(roc["fpr"], dtype=float)
                tpr = np.asarray(roc["tpr"], dtype=float)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
                fig_roc.add_trace(go.Scatter(x=np.asarray([0,1], dtype=float), y=np.asarray([0,1], dtype=float),
                                             mode="lines", name="Random", line=dict(dash="dash")))
                fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR", height=350, separators=", ")
                st.plotly_chart(fig_roc, use_container_width=True)

            if pr.get("precision") and pr.get("recall"):
                prec = np.asarray(pr["precision"], dtype=float)
                rec  = np.asarray(pr["recall"], dtype=float)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Precision-Recall"))
                fig_pr.update_layout(title="Précision–Rappel", xaxis_title="Recall", yaxis_title="Precision", height=350, separators=", ")
                st.plotly_chart(fig_pr, use_container_width=True)

            thr = np.asarray(cc.get("threshold", []), dtype=float)
            cst = np.asarray(cc.get("cost", []), dtype=float)
            best = cc.get("best", {})
            if len(thr) and len(cst) and best:
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Scatter(x=thr, y=cst, mode="lines", name="Coût total"))
                fig_cost.add_vline(x=float(best.get("threshold", 0.0)), line_width=2, line_dash="dash", line_color="green",
                                   annotation_text=f"Seuil optimal = {float(best.get('threshold', 0.0)):.3f}",
                                   annotation_position="top left")
                fig_cost.add_vline(x=float(st.session_state["threshold"]), line_width=2, line_dash="dot", line_color="red",
                                   annotation_text=f"Seuil courant = {st.session_state['threshold']:.3f}",
                                   annotation_position="top right")
                fig_cost.update_layout(title=f"Coût vs Seuil ({unit})", xaxis_title="Seuil",
                                       yaxis_title=f"Coût total ({unit})", height=350, separators=", ")
                st.plotly_chart(fig_cost, use_container_width=True)

                st.markdown("**Synthèse (API)**")
                best_row = {
                    "Seuil": f"{float(best.get('threshold', 0.0)):.3f}",
                    "Coût total": f"{float(best.get('cost', 0.0)):.0f} {unit}",
                    "TP": int(best.get("tp", 0)), "FP": int(best.get("fp", 0)),
                    "FN": int(best.get("fn", 0)), "TN": int(best.get("tn", 0)),
                    "Précision": f"{float(best.get('precision', 0.0)):.3f}",
                    "Rappel": f"{float(best.get('recall', 0.0)):.3f}",
                    "F1": f"{float(best.get('f1', 0.0)):.3f}",
                }
                cur_row = {
                    "Seuil": f"{st.session_state['threshold']:.3f}",
                    "Coût total": "—", "TP": "—", "FP": "—", "FN": "—", "TN": "—",
                    "Précision": "—", "Rappel": "—", "F1": "—",
                }
                st.dataframe(pd.DataFrame([best_row, cur_row], index=["Seuil optimal (API)", "Seuil courant"]), use_container_width=True)
            else:
                st.info("Courbe de coût non disponible depuis l'API.")
        except Exception as e:
            st.error(f"Échec récupération métriques API : {e}")

    else:
        mdl_local = model_local
        if mdl_local is None or X.empty or TARGET_COL is None:
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

                if len(X_all) > 20000:
                    X_all = X_all.sample(20000, random_state=42)
                    y_all = y_all.loc[X_all.index]

                try:
                    p_all = mdl_local.predict_proba(X_all)[:, 1]
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
                        fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR", height=350, separators=", ")
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception:
                        pass

                    try:
                        prec, rec, pr_th = precision_recall_curve(y_all.values, p_all)
                        fig_pr = go.Figure()
                        fig_pr.add_trace(go.Scatter(x=np.asarray(rec, dtype=float),
                                                    y=np.asarray(prec, dtype=float),
                                                    mode="lines", name="Precision-Recall"))
                        fig_pr.update_layout(title="Précision–Rappel", xaxis_title="Recall", yaxis_title="Precision", height=350, separators=", ")
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
                    fig_cost.update_layout(title=f"Coût vs Seuil ({unit})", xaxis_title="Seuil", yaxis_title=f"Coût total ({unit})", height=350, separators=", ")
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
                    st.dataframe(pd.DataFrame([best_row, cur_row], index=["Seuil optimal", "Seuil courant"]), use_container_width=True)

                    apply_cols = st.columns([1,2])
                    with apply_cols[0]:
                        if st.button("✅ Appliquer le seuil optimal au dashboard"):
                            st.session_state["threshold"] = float(best["threshold"])
                            st.success(f"Seuil mis à jour à {best['threshold']:.3f}.")
                            st.rerun()
                    with apply_cols[1]:
                        st.caption("Le seuil optimal minimise le coût total attendu : `coût = FP × coût_FP + FN × coût_FN`.")

# Footer
st.divider()
footer_cols = st.columns([2,2,1])
with footer_cols[0]:
    st.caption("© Prêt à dépenser — Dashboard pédagogique. Transparence & explicabilité des décisions d'octroi.")
with footer_cols[1]:
    st.caption("App version: " + APP_VERSION)
with footer_cols[2]:
    st.caption("Build: Streamlit + SHAP + scikit-learn")
