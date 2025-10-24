# credit_api.py — FastAPI pour scoring crédit + métriques de masse (v1.0.2)
# Run local (debug): uvicorn credit_api:app --host 0.0.0.0 --port 8000 --reload

import os
import traceback
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# ----------------- Utils chargement -----------------
def _pick_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def safe_load_model(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Modèle nécessite le paquet manquant: {e.name}")
    except Exception as e:
        raise RuntimeError(f"Échec chargement modèle: {e}")

def get_expected_input_columns(model) -> Optional[List[str]]:
    try:
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.compose import ColumnTransformer as SkColumnTransformer
    except Exception:
        SkPipeline = tuple()
        SkColumnTransformer = tuple()
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

# ----------------- Fichiers à la racine -----------------
DATA_TRAIN = _pick_first_existing([
    "application_train_clean.csv",
    "clients_demo.csv",
    "clients_demo.parquet",
])
MODEL_ISO  = _pick_first_existing(["model_calibrated_isotonic.joblib"])
MODEL_SIG  = _pick_first_existing(["model_calibrated_sigmoid.joblib"])
MODEL_BASE = _pick_first_existing(["model_baseline_logreg.joblib"])

if not DATA_TRAIN:
    raise RuntimeError("Données non trouvées (placez clients_demo.csv ou .parquet à la racine).")

pool_df = load_table(DATA_TRAIN)
ID_COL = "SK_ID_CURR" if "SK_ID_CURR" in pool_df.columns else pool_df.columns[0]
TARGET_COL = "TARGET" if "TARGET" in pool_df.columns else None

model_path = MODEL_ISO or MODEL_SIG or MODEL_BASE
if not model_path:
    raise RuntimeError("Aucun modèle joblib trouvé à la racine.")
model = safe_load_model(model_path)

df_idx = pool_df.set_index(ID_COL)
expected_cols = get_expected_input_columns(model) or [c for c in df_idx.columns if c not in {"TARGET"}]
for c in expected_cols:
    if c not in df_idx.columns:
        df_idx[c] = np.nan
X_all = df_idx[expected_cols]

# Background par défaut pour SHAP (réduit pour fiabiliser en cloud free)
_BG_BASE = X_all.sample(min(100, len(X_all)), random_state=42)

# ----------------- Métriques & coût -----------------
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

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
    best_pos = int(np.argmin(pd.to_numeric(df["cost"], errors="coerce").to_numpy()))
    best = df.iloc[best_pos].to_dict()
    return df, best

# ----------------- FastAPI app -----------------
app = FastAPI(title="Credit Scoring API", version="1.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Route racine: redirige vers /docs
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

class PredictBody(BaseModel):
    client_id: Optional[int | str] = None
    features: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = 0.08
    shap: Optional[bool] = True
    topk: Optional[int] = 10
    bg_max: Optional[int] = 80  # taille max du background SHAP (pour Render free)

class MetricsBody(BaseModel):
    cost_fp: float = 100.0
    cost_fn: float = 1000.0
    max_sample: int = 20000
    step: float = 0.001

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(model_path), "rows": int(len(X_all))}

@app.get("/sample_ids")
def sample_ids(n: int = 5):
    ids = X_all.index[: max(1, min(int(n), len(X_all)))].tolist()
    clean = []
    for v in ids:
        try:
            clean.append(int(v))
        except Exception:
            clean.append(v)
    return {"ids": clean}

@app.post("/predict")
def predict(body: PredictBody):
    # construire la ligne
    if body.client_id is not None:
        cid = body.client_id
        if cid not in X_all.index:
            raise HTTPException(status_code=404, detail=f"client_id {cid} introuvable")
        x_row = X_all.loc[[cid]]
    elif body.features is not None:
        row = {}
        for c in expected_cols:
            v = body.features.get(c, None)
            row[c] = np.nan if v is None else v
        x_row = pd.DataFrame([row], columns=expected_cols)
    else:
        raise HTTPException(status_code=400, detail="Fournir client_id OU features")

    # proba
    try:
        proba = float(model.predict_proba(x_row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {e}")

    resp: Dict[str, Any] = {"proba_default": proba, "threshold": body.threshold, "top_contrib": []}

    # SHAP (optionnel) — on renvoie aussi le statut/erreur en clair
    if body.shap:
        shap_status = "ok"
        shap_error = None
        try:
            import shap
            bg_base = _BG_BASE
            bg_max = int(body.bg_max or 80)
            bg_max = max(20, min(bg_max, len(bg_base)))
            # échantillonne un background léger pour fiabilité
            if len(bg_base) > bg_max:
                bg = bg_base.sample(bg_max, random_state=42)
            else:
                bg = bg_base

            def f(Xdf):
                if not isinstance(Xdf, pd.DataFrame):
                    Xdf = pd.DataFrame(Xdf, columns=list(bg.columns))
                return model.predict_proba(Xdf)[:, 1]

            masker = shap.maskers.Independent(bg)
            explainer = shap.Explainer(f, masker, feature_names=list(bg.columns))
            ex = explainer(x_row)

            vals = np.array(ex.values).reshape(-1)
            abs_vals = np.abs(vals)
            order = np.argsort(-abs_vals)[: int(body.topk or 10)]
            top_contrib = []
            for idx in order:
                feat = bg.columns[idx]
                top_contrib.append({
                    "feature": str(feat),
                    "shap_value": float(vals[idx]),
                    "value": (None if pd.isna(x_row.iloc[0, idx]) else x_row.iloc[0, idx]),
                })
            resp["top_contrib"] = top_contrib
            resp["shap_status"] = shap_status
            resp["shap_bg_size"] = int(len(bg))
        except Exception as e:
            shap_status = "error"
            shap_error = f"{type(e).__name__}: {str(e)}"
            # log stack dans les logs Render
            print("[SHAP ERROR]", shap_error)
            print(traceback.format_exc())
            resp["shap_status"] = shap_status
            resp["shap_error"] = shap_error
            resp["shap_bg_size"] = 0

    return resp

@app.post("/metrics")
def metrics(body: MetricsBody):
    if TARGET_COL is None or TARGET_COL not in pool_df.columns:
        raise HTTPException(status_code=400, detail="TARGET manquant dans les données serveur.")
    df_lab = pool_df.dropna(subset=[TARGET_COL]).copy()
    if df_lab.empty:
        raise HTTPException(status_code=400, detail="Aucune ligne labellisée.")
    df_lab = df_lab.set_index(ID_COL)
    for c in expected_cols:
        if c not in df_lab.columns:
            df_lab[c] = np.nan
    X = df_lab[expected_cols]
    y = df_lab[TARGET_COL].astype(int)

    if len(X) > int(body.max_sample):
        X = X.sample(int(body.max_sample), random_state=42)
        y = y.loc[X.index]

    try:
        p = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur scoring masse: {e}")

    # AUC, ROC, PR
    try:
        auc = float(roc_auc_score(y.values, p))
    except Exception:
        auc = float("nan")
    try:
        fpr, tpr, _ = roc_curve(y.values, p)
        fpr, tpr = fpr.tolist(), tpr.tolist()
    except Exception:
        fpr, tpr = [], []
    try:
        prec, rec, _ = precision_recall_curve(y.values, p)
        prec, rec = prec.tolist(), rec.tolist()
    except Exception:
        prec, rec = [], []

    # Courbe de coût
    df_cost, best = cost_curve(y.values, p, float(body.cost_fp), float(body.cost_fn), float(body.step))
    return {
        "auc": auc,
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"precision": prec, "recall": rec},
        "cost_curve": {
            "threshold": df_cost["threshold"].astype(float).tolist(),
            "cost": df_cost["cost"].astype(float).tolist(),
            "best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best.items()},
        },
        "n_scored": int(len(X)),
    }
