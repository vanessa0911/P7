
-----------------

source .venv/bin/activate || (python -m venv .venv && source .venv/bin/activate)

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit cache clear

python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true


API base URL 
(https://p7youpi.onrender.com)

-----------
render : 
Build Command : pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements-api.txt
Start Command : uvicorn main:app --host 0.0.0.0 --port $PORT
-------------

Comment tester vite que tout est OK

Va sur https://p7youpi.onrender.com/health → tu dois voir {"status":"ok"}. voir même : {"status":"ok","model_loaded":false,"feature_names_loaded":true,"clients_loaded":true,"model_path":"model_calibrated_isotonic.joblib"}

Va sur https://p7youpi.onrender.com/info → tu dois voir les features, si le CSV est chargé, etc.

Va sur https://p7youpi.onrender.com/ids → tu dois voir une petite liste d’IDs (si clients_demo.csv est bien chargé).

Teste /predict (mode client_id) depuis ton ordi :

curl -X POST https://p7-tentative2.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 100002, "threshold": 0.67, "shap": true, "topk": 10}'


Remplace 100002 par un ID retourné par /ids
