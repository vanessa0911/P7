# main.py
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# Config & artefacts
# -------------------------
# Place tes fichiers à la racine du projet Render:
# - model_calibrated_isotonic.joblib (ou adapte le nom ci-dessous)
# - clients_demo.csv OU application_train_clean.csv (pour retrouver client_id + background SHAP)
# - feature_names.npy (optionnel, aide à imposer l'ordre des colonnes)
MODEL_PATHS = [
    "model_calibrated_isotonic.joblib",
    "model_calibrated_sigmoid.joblib",
    "model_baseline_logreg.joblib",
]
DATA_CANDIDATES = [
    "application_train_clean.csv",
    "clients_demo.csv",
    "clients_demo.parquet",
]
FEAT_NAMES_PATH = "feature_names.npy"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"  # seulement pour /metrics; facultatif

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="P7 Credit API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------------------
# Charge artefacts
# -------------------------
def _pick_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

MODEL_PATH = _pick_first_existing(MODEL_PATHS)
DATA_PATH = _pick_first_existing(DATA_CANDIDATES)

if MODEL_PATH is None:
    raise RuntimeError("Aucun modèle .joblib trouvé à la racine.")

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Échec chargement modèle {MODEL_PATH}: {e}")

# dataset pour lookup client_id + background SHAP
if DATA_PATH is None:
    POOL_DF = pd.DataFrame()
else:
    ext = os.path.splitext(DATA_PATH)[1].lower()
    if ext in [".parquet", ".pq"]:
        POOL_DF = pd.read_parquet(DATA_PATH)
    else:
        POOL_DF = pd.read_csv(DATA_PATH)

# feature names attendus
if os.path.exists(FEAT_NAMES_PATH):
    try:
        FEAT_NAMES = list(np.load(FEAT_NAMES_PATH, allow_pickle=True).tolist())
    except Exception:
        FEAT_NAMES = None
else:
    FEAT_NAMES = None

def get_expected_input_columns():
    # priorité: feature_names.npy
    if FEAT_NAMES:
        return [c for c in FEAT_NAMES if c in POOL_DF.columns or True]  # on garde l'ordre

    # ensuite: modèle sklearn
    try:
        if hasattr(MODEL, "feature_names_in_"):
            return list(MODEL.feature_names_in_)
    except Exception:
        pass

    # sinon: colonnes du pool_df sans ID/TARGET
    if not POOL_DF.empty:
        cols = list(POOL_DF.columns)
        return [c for c in cols if c not in {ID_COL, TARGET_COL}]
    return []

EXPECTED_COLS = get_expected_input_columns()

# -------------------------
# Schémas Pydantic
# -------------------------
class PredictPayload(BaseModel):
    client_id: Optional[str] = Field(default=None, description="ID client connu du dataset côté API")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Dictionnaire de features pour un nouveau client")
    threshold: Optional[float] = 0.67
    shap: bool = True
    topk: int = 10

class MetricsPayload(BaseModel):
    cost_fp: float = 100.0
    cost_fn: float = 1000.0
    max_sample: int = 20000
    step: float = 0.001

# -------------------------
# Utils
# -------------------------
def _ensure_dataframe_one_row(features: Dict[str, Any], expected_cols: List[str]) -> pd.DataFrame:
    """Crée une ligne de DataFrame alignée sur expected_cols, en remplissant les manquants par NaN."""
    row = {c: features.get(c, np.nan) for c in expected_cols}
    return pd.DataFrame([row], columns=expected_cols)

def _get_row_by_id(client_id: Any, expected_cols: List[str]) -> pd.DataFrame:
    if POOL_DF.empty:
        raise HTTPException(status_code=404, detail="Dataset côté API indisponible pour lookup par client_id.")
    if ID_COL not in POOL_DF.columns:
        raise HTTPException(status_code=500, detail=f"Colonne ID {ID_COL} absente du dataset côté API.")
    df = POOL_DF.set_index(ID_COL)
    if client_id not in df.index:
        raise HTTPException(status_code=404, detail=f"client_id {client_id} introuvable.")
    x = df.loc[[client_id]]
    # aligne colonnes
    for c in expected_cols:
        if c not in x.columns:
            x[c] = np.nan
    return x[expected_cols]

def _predict_proba_one(x_row: pd.DataFrame) -> float:
    # Certains modèles n'ont pas predict_proba mais decision_function, on gère les deux.
    if hasattr(MODEL, "predict_proba"):
        p = MODEL.predict_proba(x_row)[:, 1][0]
        return float(p)
    elif hasattr(MODEL, "decision_function"):
        from sklearn.metrics import auc
        # fallback très basique: sigmoid sur decision_function
        z = float(MODEL.decision_function(x_row)[0])
        return float(1 / (1 + math.exp(-z)))
    else:
        # pire des cas, predict binaire
        y = int(MODEL.predict(x_row)[0])
        return float(y)

def _compute_shap(x_row: pd.DataFrame, expected_cols: List[str], topk: int) -> List[Dict[str, Any]]:
    """
    SHAP robuste via wrapper fonctionnel: f(DataFrame) -> proba.
    On utilise un background échantillonné (<=200).
    """
    import shap

    # Background: on part du dataset si dispo, sinon on répète la ligne
    if not POOL_DF.empty:
        bg = POOL_DF.copy()
        # aligne colonnes
        for c in expected_cols:
            if c not in bg.columns:
                bg[c] = np.nan
        bg = bg[expected_cols].copy()
        if len(bg) > 200:
            bg = bg.sample(200, random_state=42)
    else:
        bg = pd.concat([x_row] * 50, ignore_index=True)

    # Fonction prédictive compatible DataFrame
    def f(Xdf):
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf, columns=expected_cols)
        # Ensure exact col order
        Xdf = Xdf[expected_cols]
        return MODEL.predict_proba(Xdf)[:, 1]

    # Masker + Explainer
    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(f, masker, feature_names=expected_cols)

    ex = explainer(x_row)
    vals = np.array(ex.values).reshape(-1)
    # construit topk
    names = np.array(expected_cols)
    abs_vals = np.abs(vals)
    order = np.argsort(-abs_vals)[:topk]
    out = []
    for i in order:
        out.append({
            "feature": names[i].item() if hasattr(names[i], "item") else str(names[i]),
            "shap_value": float(vals[i]),
            "value": (x_row.iloc[0, i] if i < x_row.shape[1] else None),
        })
    return out

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        if payload.client_id is None and payload.features is None:
            raise HTTPException(status_code=422, detail="Fournir soit client_id, soit features.")

        # Prépare la ligne x_row
        if payload.client_id is not None:
            # client existant côté API
            try:
                cid = int(payload.client_id)
            except Exception:
                cid = payload.client_id  # peut être string
            x_row = _get_row_by_id(cid, EXPECTED_COLS)
        else:
            # nouveau client: features dict -> DataFrame alignée
            x_row = _ensure_dataframe_one_row(payload.features or {}, EXPECTED_COLS)

        # Proba
        proba = _predict_proba_one(x_row)

        # SHAP optionnel
        top_contrib = []
        if payload.shap:
            try:
                top_contrib = _compute_shap(x_row, EXPECTED_COLS, topk=payload.topk)
            except Exception as e:
                # On n’échoue pas: on retourne juste sans SHAP
                print(f"[WARN] SHAP indisponible: {e}")

        return {
            "proba_default": float(proba),
            "threshold": float(payload.threshold if payload.threshold is not None else 0.67),
            "top_contrib": top_contrib,  # liste (feature, shap_value, value)
            "n_features": int(len(EXPECTED_COLS)),
            "mode": "by_id" if payload.client_id is not None else "by_payload",
        }
    except HTTPException:
        raise
    except Exception as e:
        # log et renvoie 500
        print(f"[ERROR] /predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
def metrics(m: MetricsPayload):
    """
    Facultatif: utilisé par l’onglet Seuil & coût métier du dashboard.
    Renvoie ROC/PR/Courbe de coût si TARGET est disponible dans le dataset côté API.
    """
    try:
        if POOL_DF.empty or TARGET_COL not in POOL_DF.columns:
            return {"message": "Dataset labellisé indisponible côté API.", "n_scored": 0}

        df = POOL_DF.dropna(subset=[TARGET_COL]).copy()
        if len(df) == 0:
            return {"message": "Aucunes lignes labellisées (TARGET).", "n_scored": 0}

        # Aligner colonnes
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        X = df[EXPECTED_COLS]
        y = df[TARGET_COL].astype(int)

        # Échantillonnage
        if len(X) > m.max_sample:
            X = X.sample(int(m.max_sample), random_state=42)
            y = y.loc[X.index]

        # Score proba
        if hasattr(MODEL, "predict_proba"):
            p = MODEL.predict_proba(X)[:, 1]
        else:
            # fallback très simple
            if hasattr(MODEL, "decision_function"):
                z = MODEL.decision_function(X)
                p = 1 / (1 + np.exp(-z))
            else:
                p = MODEL.predict(X).astype(float)

        # ROC/PR
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
        fpr, tpr, _ = roc_curve(y.values, p)
        auc = roc_auc_score(y.values, p)
        prec, rec, _ = precision_recall_curve(y.values, p)

        # Courbe de coût
        def _cost_at_thr(t: float) -> Dict[str, Any]:
            y_pred = (p >= t).astype(int)
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            cost = fp * m.cost_fp + fn * m.cost_fn
            return {
                "threshold": t, "cost": float(cost),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1": float(f1_score(y, y_pred, zero_division=0)),
            }

        step = float(m.step if m.step and m.step > 0 else 0.001)
        thrs = np.arange(0.0, 1.0 + step, step, dtype=float)
        rows = [_cost_at_thr(float(t)) for t in thrs]
        dfc = pd.DataFrame(rows)
        best_idx = int(np.argmin(dfc["cost"].values))
        best = dfc.iloc[best_idx].to_dict()

        return {
            "n_scored": int(len(X)),
            "auc": float(auc),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
            "cost_curve": {
                "threshold": dfc["threshold"].tolist(),
                "cost": dfc["cost"].tolist(),
                "best": best,
            },
        }
    except Exception as e:
        print(f"[ERROR] /metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
