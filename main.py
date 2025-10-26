from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, List, Dict
import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="P7 Credit API", version="1.0.1")

# --- Artefacts / env vars ---
MODEL_PATH = os.getenv("MODEL_PATH", "model_calibrated_isotonic.joblib")
DATA_TRAIN = os.getenv("DATA_TRAIN", "clients_demo.csv")
ID_COL     = os.getenv("ID_COL", "SK_ID_CURR")

# --- Charge modèle ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at {MODEL_PATH}: {e}")

# --- Charge pool_df (pour mode client_id) ---
try:
    pool_df = pd.read_csv(DATA_TRAIN)
    if ID_COL not in pool_df.columns:
        raise RuntimeError(f"{ID_COL} not found in {DATA_TRAIN} columns")
    pool_df = pool_df.set_index(ID_COL)
except Exception as e:
    pool_df = pd.DataFrame()
    print(f"WARNING: cannot load pool data: {e}")

# Features attendues par le modèle
FEATURES_IN: List[str] = list(getattr(model, "feature_names_in_", []))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_path": MODEL_PATH,
        "data_train": DATA_TRAIN,
        "id_col": ID_COL,
        "features_in_len": len(FEATURES_IN),
        "features_in_sample": FEATURES_IN[:10],
        "pool_df_loaded": (not pool_df.empty),
        "pool_df_cols_len": int(len(pool_df.columns)) if not pool_df.empty else 0,
    }

@app.get("/ids")
def list_ids(limit: int = 50):
    if pool_df.empty:
        return {"ids": []}
    idx = list(pool_df.index[:max(0, min(limit, len(pool_df)))])
    return {"ids": idx}

class PredictPayload(BaseModel):
    client_id: Optional[Union[int, str]] = None
    features: Optional[Dict] = None
    threshold: Optional[float] = None
    shap: bool = False
    topk: int = 10

def align_row_like(row_like: pd.DataFrame) -> pd.DataFrame:
    if not FEATURES_IN:
        raise HTTPException(500, "Model has no feature_names_in_. Re-export with sklearn >=1.0.")
    # crée une ligne avec toutes les colonnes attendues
    X = pd.DataFrame(columns=FEATURES_IN)
    # remplit depuis row_like si dispo
    src = row_like.iloc[0].to_dict()
    for c in FEATURES_IN:
        X.loc[0, c] = src.get(c, np.nan)
    # Convertit colonnes non-numériques en objets mais autorise le modèle (OneHot etc.)
    # Si votre modèle attend déjà des numériques, à vous de caster en float ici.
    return X

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        # 1) Récupère la ligne
        if payload.client_id is not None:
            if pool_df.empty:
                raise HTTPException(500, "Server has no pool_df; cannot use client_id mode.")
            cid = payload.client_id
            # essaie int puis str
            row = None
            try:
                row = pool_df.loc[[int(cid)]]
            except Exception:
                try:
                    row = pool_df.loc[[str(cid)]]
                except Exception:
                    raise HTTPException(404, f"client_id {cid} not found.")
        elif payload.features is not None:
            row = pd.DataFrame([payload.features])
        else:
            raise HTTPException(400, "Provide either client_id or features")

        # 2) Aligne colonnes
        X = align_row_like(row)

        # 3) Proba
        proba = float(model.predict_proba(X)[0, 1])

        # 4) SHAP (optionnel, renvoie vide si shap indispo)
        top_contrib = []
        if payload.shap:
            try:
                import shap  # shap n'est pas dans requirements par défaut; ajoutez-le si vous le voulez côté serveur
                # background = sous-ensemble propre
                if not pool_df.empty:
                    bg_src = pool_df[FEATURES_IN]
                    bg = bg_src.dropna()
                    if len(bg) == 0:
                        bg = X.copy()
                    else:
                        bg = bg.sample(min(200, len(bg)), random_state=42)
                else:
                    bg = X.copy()

                def f(Xarr):
                    Xdf = pd.DataFrame(Xarr, columns=FEATURES_IN)
                    return model.predict_proba(Xdf)[:, 1]

                explainer = shap.Explainer(f, shap.maskers.Independent(bg), feature_names=FEATURES_IN)
                ex = explainer(X)
                vals = np.array(ex.values).reshape(-1)
                order = np.argsort(np.abs(vals))[::-1][:max(1, payload.topk)]
                top_contrib = [
                    {"feature": FEATURES_IN[i], "shap_value": float(vals[i]), "value": X.iloc[0][FEATURES_IN[i]]}
                    for i in order
                ]
            except Exception as e:
                # SHAP serveur facultatif
                top_contrib = []

        return {"proba_default": proba, "top_contrib": top_contrib}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
