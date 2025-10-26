# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="P7 Credit API", version="1.0.0")

# --- Artefacts (dans le repo !) ---
MODEL_PATH = os.getenv("MODEL_PATH", "model_calibrated_isotonic.joblib")
DATA_TRAIN = os.getenv("DATA_TRAIN", "clients_demo.csv")
ID_COL = os.getenv("ID_COL", "SK_ID_CURR")

# --- Chargements ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at {MODEL_PATH}: {e}")

try:
    pool_df = pd.read_csv(DATA_TRAIN)
    if ID_COL not in pool_df.columns:
        raise RuntimeError(f"{ID_COL} not found in {DATA_TRAIN}")
    pool_df = pool_df.set_index(ID_COL)
except Exception as e:
    # Si tu n’as pas de dataset côté serveur, tu peux démarrer sans, mais /predict avec client_id échouera.
    pool_df = pd.DataFrame()
    print(f"WARNING: cannot load pool data: {e}")

FEATURES_IN = list(getattr(model, "feature_names_in_", []))

@app.get("/health")
def health():
    return {"status": "ok"}

class PredictPayload(BaseModel):
    client_id: int | str | None = None
    features: dict | None = None
    threshold: float | None = None
    shap: bool = False
    topk: int = 10

def _align_row(row_like: pd.DataFrame) -> pd.DataFrame:
    """Aligne une ligne sur model.feature_names_in_ en conservant l'ordre et en insérant les colonnes manquantes."""
    if not FEATURES_IN:
        raise HTTPException(500, "Model has no feature_names_in_. Export or train with scikit-learn 1.0+.")
    X = pd.DataFrame(columns=FEATURES_IN)
    for c in FEATURES_IN:
        X.loc[0, c] = row_like.iloc[0].get(c, np.nan)
    # conserver types bruts (catégoriel/str OK si ton pipeline les gère)
    return X

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        if payload.client_id is not None:
            if pool_df.empty:
                raise HTTPException(500, "Server has no pool_df loaded; cannot use client_id mode.")
            cid = payload.client_id
            # cast index types permissivement
            try:
                row = pool_df.loc[[int(cid)]]
            except Exception:
                try:
                    row = pool_df.loc[[str(cid)]]
                except Exception:
                    raise HTTPException(404, f"client_id {cid} not found in server dataset.")
        elif payload.features is not None:
            row = pd.DataFrame([payload.features])
        else:
            raise HTTPException(400, "Provide either client_id or features")

        X = _align_row(row)
        proba = float(model.predict_proba(X)[0, 1])

        top_contrib = []
        if payload.shap:
            try:
                import shap
                # background: échantillon petit, columns = FEATURES_IN
                if pool_df.empty:
                    bg = X.copy()
                else:
                    bg_src = pool_df[FEATURES_IN].copy()
                    # coerce safe for SHAP / function
                    for c in bg_src.columns:
                        # on évite les objets non numériques si le modèle les refuse; on garde tel quel si le pipeline gère
                        pass
                    bg = bg_src.dropna().sample(min(200, len(bg_src.dropna())), random_state=42) if len(bg_src.dropna()) else X.copy()

                def f(Xarr):
                    Xdf = pd.DataFrame(Xarr, columns=FEATURES_IN)
                    return model.predict_proba(Xdf)[:, 1]

                explainer = shap.Explainer(f, shap.maskers.Independent(bg), feature_names=FEATURES_IN)
                ex = explainer(X)
                vals = ex.values[0]
                order = np.argsort(np.abs(vals))[::-1][:max(1, payload.topk)]
                top_contrib = [
                    {"feature": FEATURES_IN[i], "shap_value": float(vals[i]), "value": X.iloc[0][FEATURES_IN[i]]}
                    for i in order
                ]
            except Exception as e:
                # pas d’échec total si SHAP plante
                top_contrib = []

        return {"proba_default": proba, "top_contrib": top_contrib}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
