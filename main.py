import os, io, traceback
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
import joblib
import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---- Config via env vars (defaults assume files at repo root) ----
MODEL_PATH   = os.environ.get("MODEL_PATH", "model_calibrated_isotonic.joblib")
FEATS_PATH   = os.environ.get("FEATURES_PATH", "feature_names.npy")
CLIENTS_PATH = os.environ.get("CLIENTS_PATH", "clients_demo.csv")
MODEL_URL    = os.environ.get("MODEL_URL", "")  # optional http(s) url to download model if missing
ID_COL       = os.environ.get("ID_COL", "SK_ID_CURR")

app = FastAPI(title="P7 Credit API", version="1.0.1")

model: Optional[Any] = None
feature_names: Optional[List[str]] = None
clients_df: Optional[pd.DataFrame] = None

def log(msg: str): 
    print(f"[P7-API] {msg}", flush=True)

# ---- Utilities ----
def safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def download_if_missing(url: str, dst: str) -> bool:
    """Download file from url to dst if dst is missing. Returns True if file exists after call."""
    if not url:
        return False
    if safe_exists(dst):
        return True
    try:
        log(f"Model missing at '{dst}', downloading from {url} ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dst, "wb") as f:
            f.write(r.content)
        log(f"Downloaded model to {dst} ({len(r.content)} bytes)")
        return True
    except Exception as e:
        log(f"ERROR downloading model: {e}\n{traceback.format_exc()}")
        return False

def safe_load_model() -> Optional[Any]:
    try:
        if not safe_exists(MODEL_PATH):
            # attempt download if MODEL_URL provided
            if MODEL_URL:
                ok = download_if_missing(MODEL_URL, MODEL_PATH)
                if not ok:
                    log(f"Model not found and download failed. MODEL_PATH={MODEL_PATH}")
                    return None
            else:
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
        if not safe_exists(FEATS_PATH):
            log(f"No feature names file at {FEATS_PATH}")
            return None
        arr = np.load(FEATS_PATH, allow_pickle=True)
        names = list(arr.tolist())
        log(f"Loaded {len(names)} feature names from {FEATS_PATH}")
        return names
    except Exception as e:
        log(f"ERROR loading feature names: {e}\n{traceback.format_traceback()}")
        return None

def safe_load_clients() -> Optional[pd.DataFrame]:
    try:
        if safe_exists(CLIENTS_PATH):
            df = pd.read_csv(CLIENTS_PATH)
            log(f"Loaded clients: {len(df)} rows from {CLIENTS_PATH}")
            return df
        log(f"No clients CSV at {CLIENTS_PATH}")
        return None
    except Exception as e:
        log(f"ERROR loading clients CSV: {e}\n{traceback.format_exc()}")
        return None

def expected_input_columns(m) -> Optional[List[str]]:
    try:
        return list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else None
    except Exception:
        return None

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="ignore")
        except Exception:
            pass
    return out

def df_from_features(d: Dict[str, Any]) -> pd.DataFrame:
    return to_numeric_df(pd.DataFrame([d]))

def align_to_features(df: pd.DataFrame, feat_names: Optional[List[str]]) -> pd.DataFrame:
    if feat_names:
        out = pd.DataFrame(columns=feat_names)
        for c in feat_names:
            out[c] = df[c] if c in df.columns else np.nan
        return out
    return df

# ---- API models ----
class PredictRequest(BaseModel):
    client_id: Optional[Union[int, str]] = None
    features: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = Field(default=0.67, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    proba_default: float
    decision: str
    threshold: float

class IdListResponse(BaseModel):
    ids: List[Union[int, str]]
    count: int

# ---- Startup ----
@app.on_event("startup")
def startup():
    global model, feature_names, clients_df
    log(f"Working dir: {os.getcwd()}")
    log(f"Listing repo root: {os.listdir('.')}")
    model = safe_load_model()
    feature_names = safe_load_feature_names()
    if feature_names is None and model is not None:
        feature_names = expected_input_columns(model)
        if feature_names:
            log(f"Inferred {len(feature_names)} feature names from model")
    clients_df = safe_load_clients()

# ---- Routes ----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_names_loaded": feature_names is not None,
        "clients_loaded": clients_df is not None,
        "model_path": MODEL_PATH,
        "features_path": FEATS_PATH,
        "clients_path": CLIENTS_PATH,
    }

@app.get("/ids", response_model=IdListResponse)
def list_ids(limit: int = Query(default=10, ge=1, le=1000)):
    if clients_df is None or ID_COL not in clients_df.columns:
        raise HTTPException(400, "Clients CSV not available on server")
    vals = clients_df[ID_COL].astype(str).head(limit).tolist()
    return {"ids": vals, "count": len(vals)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(500, "Model not loaded")

    if req.features:
        X_raw = df_from_features(req.features)
    elif req.client_id is not None:
        if clients_df is None or ID_COL not in clients_df.columns:
            raise HTTPException(400, "client_id mode unavailable on server")
        sub = clients_df[clients_df[ID_COL].astype(str) == str(req.client_id)]
        if sub.empty:
            raise HTTPException(404, f"Client {req.client_id} not found")
        X_raw = sub
    else:
        raise HTTPException(400, "Provide either 'features' or 'client_id'")

    X = align_to_features(X_raw, feature_names)
    try:
        proba = float(model.predict_proba(X)[0, 1])
        thr = float(req.threshold if req.threshold is not None else 0.67)
        decision = "Refus" if proba >= thr else "Accord"
        return PredictResponse(proba_default=proba, decision=decision, threshold=thr)
    except Exception as e:
        log(f"Predict error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Prediction error: {e}")
