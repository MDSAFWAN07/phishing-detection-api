from flask import Flask, request, jsonify
import joblib, os
import tldextract
import numpy as np

app = Flask(__name__)

# load model & optional scaler
model = joblib.load("phishing_model.joblib")
scaler = joblib.load("scaler.joblib") if os.path.exists("scaler.joblib") else None

def featurize(url):
    length = len(url)
    nb_dots = url.count('.')
    nb_hyphens = url.count('-')
    has_at = 1 if '@' in url else 0
    return [length, nb_dots, nb_hyphens, has_at]

@app.route("/predict", methods=["POST"])
def predict():
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
    proba = model.predict_proba(X)[0].tolist()   # adjust if model doesn't have predict_proba
    pred = int(model.predict(X)[0])
    return jsonify({"prediction": pred, "probabilities": proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
