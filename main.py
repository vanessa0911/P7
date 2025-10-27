import os, traceback
from typing import Optional, Dict, Any, List, Union
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model_calibrated_isotonic.joblib")
FEATS_PATH = os.environ.get("FEATURES_PATH", "feature_names.npy")
CLIENTS_PATH = os.environ.get("CLIENTS_PATH", "clients_demo.csv")
ID_COL = "SK_ID_CURR"

app = FastAPI(title="P7 Credit API", version="1.0.0")
model = None
feature_names: Optional[List[str]] = None
clients_df: Optional[pd.DataFrame] = None

def log(msg: str): print(f"[P7-API] {msg}", flush=True)

def safe_load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            log(f"Model not found at {MODEL_PATH}")
            return None
        m = joblib.load(MODEL_PATH)
        log(f"Model loaded from {MODEL_PATH}")
        return m
    except Exception as e:
        log(f"ERROR loading model: {e}\n{traceback.format_exc()}"); return None

def safe_load_feature_names():
    try:
        if not os.path.exists(FEATS_PATH):
            log(f"No feature names file at {FEATS_PATH}")
            return None
        arr = np.load(FEATS_PATH, allow_pickle=True)
        names = list(arr.tolist())
        log(f"Loaded {len(names)} feature names")
        return names
    except Exception as e:
        log(f"ERROR loading feature names: {e}\n{traceback.format_exc()}"); return None

def safe_load_clients():
    try:
        if os.path.exists(CLIENTS_PATH):
            df = pd.read_csv(CLIENTS_PATH)
            log(f"Loaded clients: {len(df)} rows from {CLIENTS_PATH}")
            return df
        log(f"No clients CSV at {CLIENTS_PATH}")
        return None
    except Exception as e:
        log(f"ERROR loading clients CSV: {e}\n{traceback.format_exc()}"); return None

def expected_input_columns(m):
    try:
        return list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else None
    except Exception:
        return None

@app.on_event("startup")
def startup():
    global model, feature_names, clients_df
    model = safe_load_model()
    feature_names = safe_load_feature_names()
    if feature_names is None and model is not None:
        feature_names = expected_input_columns(model)
        if feature_names: log(f"Inferred {len(feature_names)} feature names from model")
    clients_df = safe_load_clients()

class PredictRequest(BaseModel):
    client_id: Optional[Union[int,str]] = None
    features: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = Field(default=0.67, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    proba_default: float
    decision: str
    threshold: float

# main.py
import os
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


def df_from_features(d: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([d])
    for c in df.columns:
        try: df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception: pass
    return df

def align(df: pd.DataFrame) -> pd.DataFrame:
    if feature_names:
        out = pd.DataFrame(columns=feature_names)
        for c in feature_names:
            out[c] = df[c] if c in df.columns else np.nan
        return out
    return df

@app.get("/")
def root():
    return {"status": "up"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(500, "Model not loaded")
    X = None
    if req.features:
        X = align(df_from_features(req.features))
    elif req.client_id is not None:
        if clients_df is None or ID_COL not in clients_df.columns:
            raise HTTPException(400, "client_id mode unavailable on server")
        sub = clients_df[clients_df[ID_COL] == req.client_id]
        if sub.empty:
            raise HTTPException(404, f"Client {req.client_id} not found")
        X = align(sub)
    else:
        raise HTTPException(400, "Provide features or client_id")
    try:
        proba = float(model.predict_proba(X)[0, 1])
        thr = float(req.threshold if req.threshold is not None else 0.67)
        decision = "Refus" if proba >= thr else "Accord"
        return PredictResponse(proba_default=proba, decision=decision, threshold=thr)
    except Exception as e:
        log(f"Predict error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Prediction error: {e}")
