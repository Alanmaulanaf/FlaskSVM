import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# Konfigurasi .venv\Scripts\activate
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/svm_amdk_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "svm_rbf_v1")
ROUND_SUHU = 1

WARNA_SET = {"tidak berwarna", "berwarna"}
BAU_SET   = {"tidak berbau", "berbau"}
RASA_SET  = {"tawar", "tidak berasa", "asam", "manis"}

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal load model dari {MODEL_PATH}: {e}")

def clean_str(x: str) -> str:
    return str(x).strip().lower()

def to_float(val):
    try: return float(val)
    except Exception: return None

def permenkes_proxy(warna: str, bau: str, rasa: str) -> bool:
    return (warna == "tidak berwarna") and (bau == "tidak berbau") and (rasa in {"tawar", "tidak berasa"})

def validate_payload(js):
    missing = [k for k in ["suhu_c","warna","bau","rasa"] if k not in js]
    if missing: return False, f"Field wajib hilang: {missing}"
    suhu  = to_float(js["suhu_c"])
    warna = clean_str(js["warna"])
    bau   = clean_str(js["bau"])
    rasa  = clean_str(js["rasa"])
    if suhu is None: return False, "suhu_c harus numerik."
    if warna not in WARNA_SET: return False, f"warna harus salah satu dari {sorted(WARNA_SET)}."
    if bau   not in BAU_SET:   return False, f"bau harus salah satu dari {sorted(BAU_SET)}."
    if rasa  not in RASA_SET:  return False, f"rasa harus salah satu dari {sorted(RASA_SET)}."
    return True, {"suhu_c": round(suhu, ROUND_SUHU), "warna": warna, "bau": bau, "rasa": rasa}

app = Flask(__name__)
CORS(app)

@app.get("/")
def health():
    return jsonify({"ok": True, "model_version": MODEL_VERSION, "model_path": MODEL_PATH})

@app.post("/predict")
def predict():
    try:
        js = request.get_json(silent=True) or {}
        ok, data = validate_payload(js)
        if not ok:
            return jsonify({"error": data}), 400

       
        cols = ["suhu_c","warna","bau","rasa"]
        df = pd.DataFrame([[data["suhu_c"], data["warna"], data["bau"], data["rasa"]]], columns=cols)

        pred_label = model.predict(df)[0]

        try:
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                classes = model.named_steps["clf"].classes_
            idx_ya = list(classes).index("ya")
            prob_ya = float(model.predict_proba(df)[0][idx_ya])
        except Exception:
            prob_ya = 1.0 if pred_label == "ya" else 0.0

        return jsonify({
            "input": data,
            "label": pred_label,
            "prob_ya": round(prob_ya, 6),
            "is_permenkes_proxy": permenkes_proxy(data["warna"], data["bau"], data["rasa"]),
            "model_version": MODEL_VERSION,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Gagal memproses: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)