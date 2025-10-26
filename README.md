# P7

# 0) Nettoyage total
deactivate 2>/dev/null || true
rm -rf .venv

# 1) Recréer l’environnement
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Installer les deps (à partir du requirements.txt patché)
pip install -r requirements.txt

# 3) Vérifier que tout est bien importable
python - <<'PY'
import numpy, scipy, pandas, sklearn, joblib, pyarrow
import shap, plotly, streamlit, fastapi, pydantic, requests, reportlab
print("OK: imports chargés.")
PY

# 4) (optionnel) vérifier l’intégrité des deps
pip check

# 5) Lancer le dashboard
python -m streamlit run dashboard_streamlit_app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true









-----------------

source .venv/bin/activate || (python -m venv .venv && source .venv/bin/activate)

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit cache clear

python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true


API base URL = https://p7-credit-api.onrender.com → Statut : ✅ OK —
