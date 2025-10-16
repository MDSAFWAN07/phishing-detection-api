from flask import Flask, request, jsonify
import joblib, os, traceback
import tldextract
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# === Safe model loading with helpful errors ===
model = None
scaler = None

try:
    # Try loading XGBoost model
    model = xgb.XGBClassifier()
    model.load_model("phishing_xgboost_model.json")
except Exception as e:
    # Log to stdout (Render will capture this)
    print("ERROR loading XGBoost model:", e)
    traceback.print_exc()

try:
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    else:
        print("scaler.pkl not found - continuing without scaler")
except Exception as e:
    print("ERROR loading scaler.pkl:", e)
    traceback.print_exc()

# === Feature extraction ===
def featurize(url):
    try:
        length = len(url)
        nb_dots = url.count('.')
        nb_hyphens = url.count('-')
        has_at = 1 if '@' in url else 0
        return [length, nb_dots, nb_hyphens, has_at]
    except Exception:
        return [0,0,0,0]

# === Base route ===
@app.route("/", methods=["GET"])
def home():
    return """
    <h2>Phishing Detection API is Live!</h2>
    <p>POST /predict with JSON: <code>{"url":"http://example.com"}</code></p>
    """

# === Prediction route (POST) and friendly GET handler ===
@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return jsonify({
            "info": "Send a POST request with JSON {\"url\": \"http://example.com\"}"
        })

    try:
        # quick API key check (optional)
        if os.environ.get("API_KEY"):
            key = request.headers.get("x-api-key")
            if key != os.environ.get("API_KEY"):
                return jsonify({"error":"unauthorized"}), 401

        data = request.get_json(force=True, silent=True)
        if not data or "url" not in data:
            return jsonify({"error":"no url provided"}), 400

        url = data["url"]
        feats = featurize(url)
        X = np.array([feats])

        if scaler is not None:
            X = scaler.transform(X)

        if model is None:
            return jsonify({"error":"model not loaded on server. Check logs."}), 500

        proba = model.predict_proba(X)[0].tolist()
        pred = int(model.predict(X)[0])

        return jsonify({"prediction": pred, "probabilities": proba})

    except Exception as e:
        tb = traceback.format_exc()
        # return error and stacktrace to help debug (remove in production)
        return jsonify({"error": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    # Use PORT env var set by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
