# main.py — API FastAPI simple et robuste pour Render

import os
import json
import traceback
from typing import Optional, Dict, Any, List, Union

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- Config chemins (env vars possibles) ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "model_calibrated_isotonic.joblib")
FEATS_PATH = os.environ.get("FEATURES_PATH", "feature_names.npy")
CLIENTS_PATH = os.environ.get("CLIENTS_PATH", "clients_demo.csv")  # optionnel

app = FastAPI(title="P7 Credit API", version="1.0.0")

model = None
feature_names: Optional[List[str]] = None
clients_df: Optional[pd.DataFrame] = None
id_col = "SK_ID_CURR"  # pour clients_demo.csv si présent


def log(msg: str):
    print(f"[P7-API] {msg}", flush=True)


def safe_load_model() -> Optional[Any]:
    try:
        if not os.path.exists(MODEL_PATH):
            log(f"Model not found at {MODEL_PATH}")
            return None
        m = joblib.load(MODEL_PATH)
        log(f"Model loaded from {MODEL_PATH}")
        return m
    except Exception as e:
        log(f"ERROR loading model: {e}\n{traceback.format_exc()}")
        return None


def safe_load_feature_names() -> Optional[List[str]]:
    try:
        if not os.path.exists(FEATS_PATH):
            log(f"Feature names file not found at {FEATS_PATH} (will infer from model if possible)")
            return None
        arr = np.load(FEATS_PATH, allow_pickle=True)
        names = list(arr.tolist())
        log(f"Loaded {len(names)} feature names from {FEATS_PATH}")
        return names
    except Exception as e:
        log(f"ERROR loading feature names: {e}\n{traceback.format_exc()}")
        return None


def safe_load_clients() -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(CLIENTS_PATH):
            df = pd.read_csv(CLIENTS_PATH)
            log(f"Loaded clients dataset: {CLIENTS_PATH} ({len(df)} rows)")
            return df
        else:
            log(f"No clients dataset at {CLIENTS_PATH} (client_id mode will be unavailable)")
            return None
    except Exception as e:
        log(f"ERROR loading clients CSV: {e}\n{traceback.format_exc()}")
        return None


def expected_input_columns(m) -> Optional[List[str]]:
    try:
        # sklearn >=1.0 frequently exposes feature_names_in_
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
        # pipelines may hide it; we rely on FEATS_PATH otherwise
        return None
    except Exception:
        return None


@app.on_event("startup")
def startup_event():
    global model, feature_names, clients_df
    model = safe_load_model()
    feature_names = safe_load_feature_names()
    if feature_names is None and model is not None:
        feature_names = expected_input_columns(model)
        if feature_names:
            log(f"Inferred {len(feature_names)} feature names from model.feature_names_in_")
        else:
            log("Could not infer feature names; will accept posted columns as-is (pipeline must handle preprocessing).")
    clients_df = safe_load_clients()


class PredictRequest(BaseModel):
    client_id: Optional[Union[int, str]] = Field(default=None, description="SK_ID_CURR si disponible côté API")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Dictionnaire de features (clé=nom de variable)")
    threshold: Optional[float] = Field(default=0.67, ge=0.0, le=1.0)
    shap: Optional[bool] = Field(default=False, description="ignoré côté API")
    topk: Optional[int] = Field(default=10, ge=1, le=50, description="ignoré côté API")


class PredictResponse(BaseModel):
    proba_default: float
    decision: str
    threshold: float
    top_contrib: List[Dict[str, Any]] = []  # laissé vide (pas de SHAP côté API)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(model is not None),
        "feature_names_loaded": bool(feature_names is not None),
        "clients_loaded": bool(clients_df is not None),
        "model_path": MODEL_PATH,
    }


def build_frame_from_features(data: Dict[str, Any]) -> pd.DataFrame:
    # Convertit en DataFrame 1 ligne + casting simple
    df = pd.DataFrame([data])
    # Tentative de cast numérique là où c'est possible (les catégories resteront object)
    for c in df.columns:
        if isinstance(df[c].iloc[0], str):
            # essaye float si ça a l'air numérique
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df


def align_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligne l'input aux colonnes attendues par le modèle (si connues).
    Si on ne connaît pas les features, on renvoie tel quel (pipeline doit gérer).
    """
    if feature_names:
        out = pd.DataFrame(columns=feature_names)
        for c in feature_names:
            out[c] = df[c] if c in df.columns else np.nan
        return out
    return df


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    X: Optional[pd.DataFrame] = None

    if req.features:
        X0 = build_frame_from_features(req.features)
        X = align_to_expected(X0)

    elif req.client_id is not None:
        if clients_df is None or id_col not in clients_df.columns:
            raise HTTPException(status_code=400, detail="client_id mode unavailable (no clients dataset on server)")
        sub = clients_df[clients_df[id_col] == req.client_id]
        if sub.empty:
            raise HTTPException(status_code=404, detail=f"Client {req.client_id} not found in server dataset")
        # On aligne aussi si possible
        X = align_to_expected(sub)

    else:
        raise HTTPException(status_code=400, detail="Provide either features or client_id")

    try:
        proba = float(model.predict_proba(X)[0, 1])
        thr = float(req.threshold if req.threshold is not None else 0.67)
        decision = "Refus" if proba >= thr else "Accord"
        return PredictResponse(
            proba_default=proba,
            decision=decision,
            threshold=thr,
            top_contrib=[],  # pas de SHAP côté API
        )
    except Exception as e:
        log(f"ERROR during predict: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
