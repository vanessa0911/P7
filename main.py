# main.py — FastAPI pour scoring crédit (compatible Streamlit)
# Démarrage local: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from __future__ import annotations
import os
import json
import math
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============
# Config
# ============
MODEL_CANDIDATES = [
    "model_calibrated_isotonic.joblib",
    "model_calibrated_sigmoid.joblib",
    "model_baseline_logreg.joblib",
]
DATA_CANDIDATES = [
    "application_train_clean.csv",
    "clients_demo.csv",
]
FEATURE_NAMES_PATH = "feature_names.npy"

ID_COL_CANDIDATES = ["SK_ID_CURR", "ID", "id"]
TARGET_COL_CANDIDATES = ["TARGET", "target", "y"]

DEFAULT_THRESHOLD = 0.67

# ============
# App
# ============
app = FastAPI(title="P7 Credit API", version="1.0.0")

# CORS pour autoriser l'appel depuis le dashboard Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # autoriser tout (simple et efficace)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============
# Chargement global des artefacts
# ============
MODEL = None
X_COLUMNS: List[str] = []
TRAIN_DF = pd.DataFrame()
ID_COL: Optional[str] = None
TARGET_COL: Optional[str] = None

def _pick_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _infer_id_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    id_col = None
    tgt_col = None
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            id_col = c
            break
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            tgt_col = c
            break
    return id_col, tgt_col

def _load_model_and_data() -> Tuple[Any, List[str], pd.DataFrame, Optional[str], Optional[str]]:
    # modèle
    model_path = _pick_first(MODEL_CANDIDATES)
    if not model_path:
        raise RuntimeError("Aucun modèle .joblib trouvé à la racine.")
    model = joblib.load(model_path)

    # colonnes d'entrée attendues
    x_cols: List[str] = []
    if os.path.exists(FEATURE_NAMES_PATH):
        try:
            arr = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
            x_cols = list(arr.tolist())
        except Exception:
            x_cols = []

    # sinon, essayer d’inférer depuis le modèle
    if not x_cols:
        if hasattr(model, "feature_names_in_"):
            x_cols = list(model.feature_names_in_)
        else:
            # fallback: on attendra un CSV pour déduire
            x_cols = []

    # données train (pour /metrics et pour predict par id)
    data_path = _pick_first(DATA_CANDIDATES)
    df = pd.read_csv(data_path) if data_path else pd.DataFrame()

    # tenter d’inférer id/target
    id_col, tgt_col = (None, None)
    if not df.empty:
        id_col, tgt_col = _infer_id_target(df)
        # si on n'a pas de features, tenter d’inférer depuis le CSV
        if not x_cols:
            # columns sans TARGET ni ID
            x_cols = [c for c in df.columns if c not in set([id_col, tgt_col])]
    return model, x_cols, df, id_col, tgt_col

try:
    MODEL, X_COLUMNS, TRAIN_DF, ID_COL, TARGET_COL = _load_model_and_data()
except Exception as e:
    # on laisse démarrer l’API quand même, mais /predict échouera (Log clair dans /info)
    MODEL = None
    X_COLUMNS = []
    TRAIN_DF = pd.DataFrame()
    ID_COL = None
    TARGET_COL = None
    print("[LOAD ERROR]", e)

# ============
# Schémas d’entrées
# ============
class PredictPayload(BaseModel):
    client_id: Optional[Union[int, str]] = None
    features: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    shap: Optional[bool] = False
    topk: Optional[int] = 10

class MetricsPayload(BaseModel):
    cost_fp: float
    cost_fn: float
    max_sample: Optional[int] = 20000
    step: Optional[float] = 0.001

# ============
# Helpers
# ============
def _ensure_model_ready():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    if not X_COLUMNS:
        raise HTTPException(status_code=503, detail="Colonnes d'entrée inconnues (feature_names.npy absent et inférence impossible).")

def _row_from_client_id(client_id: Union[int, str]) -> pd.DataFrame:
    if TRAIN_DF.empty or ID_COL is None:
        raise HTTPException(status_code=400, detail="Données d'apprentissage indisponibles pour 'client_id'.")
    # cast id type de manière souple
    s_id = TRAIN_DF[ID_COL].astype(str)
    mask = (s_id == str(client_id))
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"client_id {client_id} introuvable.")
    row = TRAIN_DF.loc[mask].iloc[[0]]  # DataFrame (1 ligne)
    # conserver uniquement les colonnes features
    for c in X_COLUMNS:
        if c not in row.columns:
            row[c] = np.nan
    return row[X_COLUMNS]

def _row_from_features(feat: Dict[str, Any]) -> pd.DataFrame:
    x = pd.DataFrame([feat])
    # ajouter colonnes manquantes
    for c in X_COLUMNS:
        if c not in x.columns:
            x[c] = np.nan
    # réordonner
    x = x[X_COLUMNS]
    return x

def _predict_proba_one(x_row: pd.DataFrame) -> float:
    # laisse le Pipeline gérer l'encodage / NaN
    p = MODEL.predict_proba(x_row)[:, 1][0]
    return float(p)

def _cost_at_threshold(y_true: np.ndarray, p: np.ndarray, t: float, c_fp: float, c_fn: float):
    y_pred = (p >= t).astype(int)
    # TN, FP, FN, TP
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    cost = fp * c_fp + fn * c_fn

    # métriques
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    return dict(cost=float(cost), tn=tn, fp=fp, fn=fn, tp=tp,
                precision=float(precision), recall=float(recall), f1=float(f1))

# ============
# Endpoints
# ============
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_loaded": MODEL is not None,
        "n_features": len(X_COLUMNS),
        "features_sample": X_COLUMNS[:10],
        "train_rows": None if TRAIN_DF is None else int(len(TRAIN_DF)),
        "id_col": ID_COL,
        "target_col": TARGET_COL,
    }

@app.post("/predict")
def predict(payload: PredictPayload):
    _ensure_model_ready()
    thr = float(payload.threshold) if payload.threshold is not None else float(DEFAULT_THRESHOLD)

    # Priorité au client_id, sinon features
    if payload.client_id is not None:
        x_row = _row_from_client_id(payload.client_id)
    elif payload.features is not None:
        x_row = _row_from_features(payload.features)
    else:
        raise HTTPException(status_code=400, detail="Fournir 'client_id' ou 'features'.")

    try:
        proba = _predict_proba_one(x_row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec predict_proba: {e}")

    # Tu n'utilises pas SHAP côté API (trop lourd). On retourne juste proba.
    out = {
        "proba_default": proba,
        "threshold": thr,
        "decision": "refus" if proba >= thr else "accord",
        "top_contrib": [],  # streamlit gère l'absence sans planter
    }
    return out

@app.post("/metrics")
def metrics(m: MetricsPayload):
    _ensure_model_ready()
    if TRAIN_DF.empty or TARGET_COL is None:
        raise HTTPException(status_code=400, detail="Données labellisées indisponibles pour /metrics.")

    # Prépare X/y
    df = TRAIN_DF.dropna(subset=[TARGET_COL]).copy()
    if df.empty:
        raise HTTPException(status_code=400, detail="Aucune ligne labellisée trouvée (TARGET manquant).")

    # aligne colonnes
    for c in X_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    X = df[X_COLUMNS]
    y = df[TARGET_COL].astype(int).to_numpy()

    # échantillonnage
    max_n = int(m.max_sample or 20000)
    if len(X) > max_n:
        X = X.sample(max_n, random_state=42)
        y = y[X.index]

    # scores
    try:
        p = MODEL.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec predict_proba sur /metrics: {e}")

    # ROC
    try:
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
        fpr, tpr, _ = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        prec, rec, _ = precision_recall_curve(y, p)
    except Exception:
        # si jamais sklearn pas dispo, renvoie vide
        fpr, tpr, auc, prec, rec = np.array([]), np.array([]), float("nan"), np.array([]), np.array([])

    # Courbe de coût
    step = float(m.step or 0.001)
    ts = np.arange(0.0, 1.0 + step, step, dtype=float)
    thresholds, costs = [], []
    best = None
    best_cost = math.inf
    for t in ts:
        d = _cost_at_threshold(y, p, float(t), float(m.cost_fp), float(m.cost_fn))
        thresholds.append(float(t))
        costs.append(float(d["cost"]))
        if d["cost"] < best_cost:
            best_cost = d["cost"]
            best = dict(d)
            best["threshold"] = float(t)

    return {
        "n_scored": int(len(X)),
        "auc": auc,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "cost_curve": {"threshold": thresholds, "cost": costs, "best": best},
    }
