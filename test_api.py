import requests

BASE = "https://p7youpi.onrender.com"

def check_health():
    r = requests.get(f"{BASE}/health", timeout=15)
    print("GET /health ->", r.status_code, r.json())

def predict_by_id(sk_id=384575, threshold=0.67):
    r = requests.post(f"{BASE}/predict",
                      json={"client_id": sk_id, "threshold": threshold},
                      timeout=30)
    print("POST /predict (client_id) ->", r.status_code, r.json())

def predict_by_features(threshold=0.67):
    payload = {
        "features": {
            "AGE_YEARS": 42,
            "ANNUITY_INCOME_RATIO": 0.12,
            "EXT_SOURCE_1": 0.56
        },
        "threshold": threshold
    }
    r = requests.post(f"{BASE}/predict", json=payload, timeout=30)
    print("POST /predict (features) ->", r.status_code, r.json())

if __name__ == "__main__":
    check_health()
    predict_by_id()
    predict_by_features()
