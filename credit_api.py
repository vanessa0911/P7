# credit_api.py — Prêt à dépenser - API de scoring (v0.9.0)
# Lancement local:
#   uvicorn credit_api:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints principaux:
#   GET  /health           → {"status":"ok"}
#   GET  /ready            → vérifie que le modèle est chargé
#   GET  /metadata         → infos modèle/données
#   GET  /features         → liste des features attendues par le modèle
#   GET  /clients?limit=50 → quelques IDs disponibles (si dataset présent)
#   POST /predict          → prédiction par client_id OU par "features" (dict)
#
# Remarques:
# - Recherche automatiquement les artefacts à la racine:
#     clients_demo.csv | clients_demo.parquet | application_train_clean.csv
#     model_calibrated_isotonic.joblib | model_calibrated_sigmoid.joblib | model_baseline_logreg.joblib
#     feature_names.npy (optionnel), global_importance.csv (optionnel)
# - Alignement des colonnes (creation colonnes manquantes) comme dans le Streamlit

APP_VERSION = "0.9.0-api"

import os
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# scikit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer as SkColumnTransformer

# SHAP (optionnel)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# -------------------------------
# Helpers
# -------------------------------
def _pick_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _load_model(path: str):
    return joblib.load(path)

def _get_expected_input_columns(model) -> Optional[List[str]]:
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

def _load_feature_names(path: Optional[str], df_cols: List[str]) -> List[str]:
    if path and os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        names = list(arr.tolist())
        return [c for c in names if c in df_cols]
    return [c for c in df_cols if c not in {"TARGET", "SK_ID_CURR"}]

def _compute_local_shap(model, background: pd.DataFrame, x_row: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Retourne un DataFrame avec colonnes: feature, shap_value, abs_val, value
    """
    if not SHAP_AVAILABLE:
        return None
    bg = background
    if len(bg) > 200:
        bg = bg.sample(200, random_state=42)

    def f(Xdf):
        if not isinstance(Xdf, pd.DataFrame):
            Xdf = pd.DataFrame(Xdf, columns=list(bg.columns))
        return model.predict_proba(Xdf)[:, 1]

    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(f, masker, feature_names=list(bg.columns))
    ex = explainer(x_row)
    vals = np.array(ex.values).reshape(-1)
    df = pd.DataFrame({
        "feature": list(x_row.columns),
        "shap_value": vals,
        "abs_val": np.abs(vals),
        "value": x_row.iloc[0].values,
    }).sort_values("abs_val", ascending=False)
    return df

def _sha_file(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]
    except Exception:
        return "n/a"


# -------------------------------
# Localisation des artefacts (racine)
# -------------------------------
DATA_TRAIN = _pick_first_existing([
    "clients_demo.csv",
    "clients_demo.parquet",
    "application_train_clean.csv",
])
MODEL_ISO  = _pick_first_existing(["model_calibrated_isotonic.joblib"])
MODEL_SIG  = _pick_first_existing(["model_calibrated_sigmoid.joblib"])
MODEL_BASE = _pick_first_existing(["model_baseline_logreg.joblib"])
FEATS_PATH = _pick_first_existing(["feature_names.npy"])
GLOBIMP    = _pick_first_existing(["global_importance.csv"])

# -------------------------------
# Chargement des données & modèle
# -------------------------------
pool_df = _load_table(DATA_TRAIN) if DATA_TRAIN else pd.DataFrame()
ID_COL = "SK_ID_CURR" if (not pool_df.empty and "SK_ID_CURR" in pool_df.columns) else (pool_df.columns[0] if not pool_df.empty else None)

feature_names = _load_feature_names(FEATS_PATH, list(pool_df.columns)) if not pool_df.empty else []

model_path = MODEL_ISO or MODEL_SIG or MODEL_BASE
if not model_path:
    raise RuntimeError("Aucun fichier modèle trouvé à la racine. Ajoutez un .joblib (isotonic/sigmoid/baseline).")

try:
    model = _load_model(model_path)
except ModuleNotFoundError as e:
    raise RuntimeError(f"Le modèle nécessite un paquet manquant: {e.name}. Installez-le dans l'environnement.")
except Exception as e:
    raise RuntimeError(f"Échec du chargement du modèle `{model_path}`: {e}")

expected_cols = _get_expected_input_columns(model)
if expected_cols is None:
    # fallback: utiliser feature_names ou colonnes du dataset
    if feature_names:
        expected_cols = feature_names
    elif not pool_df.empty:
        expected_cols = [c for c in pool_df.columns if c not in {ID_COL, "TARGET"}]
    else:
        expected_cols = []

# -------------------------------
# Préparation SHAP: background
# -------------------------------
if not pool_df.empty and expected_cols:
    df_idx = pool_df.set_index(ID_COL) if ID_COL else pool_df
    # Ajoute colonnes manquantes
    for c in expected_cols:
        if c not in df_idx.columns:
            df_idx[c] = np.nan
    BACKGROUND = df_idx[expected_cols]
    BACKGROUND_SAMPLE = BACKGROUND.sample(min(200, len(BACKGROUND)), random_state=42) if len(BACKGROUND) > 0 else BACKGROUND
else:
    BACKGROUND = pd.DataFrame(columns=expected_cols)
    BACKGROUND_SAMPLE = BACKGROUND

# -------------------------------
# Pydantic models
# -------------------------------
class PredictRequest(BaseModel):
    client_id: str | int | None = Field(default=None, description="ID client présent dans le dataset (optionnel)")
    features: Dict[str, Any] | None = Field(default=None, description="Dictionnaire feature→valeur pour scorer un nouveau client")
    threshold: float | None = Field(default=0.08, description="Seuil décisionnel (proba défaut)")
    shap: bool = Field(default=False, description="Retourner aussi les contributions locales (topK)")
    topk: int = Field(default=10, description="Nombre de variables pour le top SHAP")

class PredictResponse(BaseModel):
    client_id: str | None = None
    proba_default: float
    decision: str
    threshold_used: float
    warnings: List[str] = []
    top_contrib: List[Dict[str, Any]] | None = None  # si shap=True

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(
    title="Prêt à dépenser — API de scoring",
    version=APP_VERSION,
    description="API FastAPI pour prédire la probabilité de défaut et expliquer la décision."
)

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.get("/ready")
def ready():
    ok = model is not None and isinstance(expected_cols, list)
    return {"ready": bool(ok), "n_features_expected": len(expected_cols)}

@app.get("/metadata")
def metadata():
    data = {
        "app_version": APP_VERSION,
        "model_path": model_path,
        "model_sha": _sha_file(model_path),
        "data_path": DATA_TRAIN,
        "data_sha": _sha_file(DATA_TRAIN) if DATA_TRAIN else None,
        "n_rows_data": len(pool_df) if not pool_df.empty else 0,
        "id_column": ID_COL,
        "n_expected_features": len(expected_cols),
        "shap_available": SHAP_AVAILABLE,
        "now": datetime.utcnow().isoformat(),
    }
    return data

@app.get("/features")
def features():
    return {"features": expected_cols}

@app.get("/clients")
def clients(limit: int = Query(50, ge=1, le=500)):
    if pool_df.empty or ID_COL is None:
        return {"clients": []}
    return {"clients": pool_df[ID_COL].head(limit).astype(str).tolist()}

def _align_one_row(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    Reçoit un DataFrame (1 ligne), ajoute colonnes manquantes et réordonne selon expected_cols
    """
    out = df_like.copy()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[expected_cols]
    return out

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    warnings: List[str] = []
    x_row: pd.DataFrame | None = None
    client_id_str: str | None = None

    # 1) Client existant via client_id
    if req.client_id is not None:
        if pool_df.empty or ID_COL is None:
            raise HTTPException(status_code=400, detail="Dataset indisponible pour chercher un client_id.")
        mask = pool_df[ID_COL].astype(str) == str(req.client_id)
        if not mask.any():
            raise HTTPException(status_code=404, detail=f"Client_id {req.client_id} introuvable.")
        row_raw = pool_df.loc[mask].iloc[[0]]  # DataFrame (1 ligne)
        row_raw = row_raw.drop(columns=[c for c in [ID_COL, "TARGET"] if c in row_raw.columns], errors="ignore")
        x_row = _align_one_row(row_raw)
        client_id_str = str(req.client_id)

    # 2) Nouveau client via features
    if req.features is not None:
        # Si BOTH sont fournis: on privilégie features en avertissant
        if client_id_str is not None:
            warnings.append("client_id fourni mais ignoré car 'features' est également fourni.")
        feat_df = pd.DataFrame([req.features])
        x_row = _align_one_row(feat_df)
        client_id_str = None

    if x_row is None:
        raise HTTPException(status_code=400, detail="Fournir soit 'client_id', soit 'features' (dict).")

    # Prédiction
    try:
        proba = float(model.predict_proba(x_row)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de scorer: {e}")

    # Décision
    thr = 0.08 if req.threshold is None else float(req.threshold)
    decision = "refus" if proba >= thr else "accord"

    # SHAP (optionnel)
    top_contrib = None
    if req.shap:
        if not SHAP_AVAILABLE:
            warnings.append("SHAP indisponible: module non installé.")
        else:
            # background = BACKGROUND_SAMPLE si dispo, sinon x_row même
            bg = BACKGROUND_SAMPLE if BACKGROUND_SAMPLE is not None and len(BACKGROUND_SAMPLE) > 0 else x_row
            try:
                df_shap = _compute_local_shap(model, bg, x_row)
                if df_shap is not None and not df_shap.empty:
                    df_top = df_shap.head(int(req.topk))
                    top_contrib = [
                        {
                            "feature": str(r["feature"]),
                            "value": None if pd.isna(r["value"]) else (float(r["value"]) if isinstance(r["value"], (int, float, np.floating)) else str(r["value"])),
                            "shap_value": float(r["shap_value"]),
                        }
                        for _, r in df_top.iterrows()
                    ]
            except Exception as e:
                warnings.append(f"SHAP en échec: {e}")

    return PredictResponse(
        client_id=client_id_str,
        proba_default=proba,
        decision=decision,
        threshold_used=thr,
        warnings=warnings,
        top_contrib=top_contrib
    )
