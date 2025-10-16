from flask import Flask, request, jsonify
import joblib, os
import tldextract
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# === Load XGBoost model and scaler ===
model = xgb.XGBClassifier()
model.load_model("phishing_xgboost_model.json")  # Load JSON model
scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

# === Feature extraction ===
def featurize(url):
    length = len(url)
    nb_dots = url.count('.')
    nb_hyphens = url.count('-')
    has_at = 1 if '@' in url else 0
    return [length, nb_dots, nb_hyphens, has_at]

# === Base route for info page ===
@app.route("/", methods=["GET"])
def home():
    return """
    <h2>Phishing Detection API is Live!</h2>
    <p>Use the POST endpoint <code>/predict</code> with JSON payload:</p>
    <pre>{"url": "http://example.com"}</pre>
    """

# === Prediction route ===
@app.route("/predict", methods=["POST"])
def predict():
    # Optional API key check
    if os.environ.get("API_KEY"):
        key = request.headers.get("x-api-key")
        if key != os.environ.get("API_KEY"):
            return jsonify({"error":"unauthorized"}),401

    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error":"no url provided"}),400

    url = data["url"]
    feats = featurize(url)
    X = np.array([feats])
    if scaler is not None:
        X = scaler.transform(X)

    # XGBoost prediction
    proba = model.predict_proba(X)[0].tolist()
    pred = int(model.predict(X)[0])

    return jsonify({"prediction": pred, "probabilities": proba})

# === Run app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
