# P7

source .venv/bin/activate || (python -m venv .venv && source .venv/bin/activate)

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit cache clear

python -m streamlit run dashboard_streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true


API base URL = https://p7-credit-api.onrender.com → Statut : ✅ OK —
