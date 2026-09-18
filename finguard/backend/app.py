"""
FinGuard AI - Persona-Adaptive Fraud Detection & Risk Intelligence Platform
Backend: Flask REST API serving:
  - Live transaction stream simulation (from real held-out test transactions)
  - Fraud predictions from a RandomForest model trained on creditcard.csv
  - Persona-adaptive risk scoring (adjusts alert threshold per customer segment,
    segments learned from marketing_campaign.csv via KMeans)
  - Explainability (per-transaction top contributing features)
  - Model performance dashboard (precision/recall/ROC/confusion matrix)
  - "Try it yourself" manual transaction simulator

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""
import json
import os
import random
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
print("Loading FinGuard AI model artifacts...")
fraud_model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
persona_bundle = joblib.load(os.path.join(MODEL_DIR, "persona_model.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)
with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
    FEATURE_IMPORTANCE = json.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "threshold.json")) as f:
    BASE_THRESHOLD = json.load(f)["threshold"]
with open(os.path.join(MODEL_DIR, "personas.json")) as f:
    PERSONAS = json.load(f)
with open(os.path.join(MODEL_DIR, "feature_ranges.json")) as f:
    FEATURE_RANGES = json.load(f)
with open(os.path.join(DATA_DIR, "sample_transactions.json")) as f:
    SAMPLE_TRANSACTIONS = json.load(f)

PERSONA_BY_ID = {p["cluster_id"]: p for p in PERSONAS}
TOP_FEATURES = [f["feature"] for f in FEATURE_IMPORTANCE[:6]]

# ---------------------------------------------------------------------------
# In-memory simulation state (resets on server restart -- perfect for a live demo)
# ---------------------------------------------------------------------------
STATE = {
    "cursor": 0,
    "processed_log": [],       # list of scored transactions shown so far
    "stats": {"total": 0, "flagged": 0, "true_positive": 0, "false_positive": 0,
              "false_negative": 0, "true_negative": 0},
}

random.seed(7)
# Pre-assign each demo transaction a persona (synthetic linkage for the demo --
# public datasets don't share customer IDs across sources; see README)
for txn in SAMPLE_TRANSACTIONS:
    txn["persona_cluster"] = random.choice(list(PERSONA_BY_ID.keys()))


def explain_transaction(feature_dict):
    """Return the top contributing features for one transaction, ranked by
    |value * global_importance| -- a lightweight, dependency-free stand-in
    for a full SHAP explanation, ideal for a live demo."""
    scored = []
    for feat in FEATURE_COLS:
        val = feature_dict.get(feat, 0)
        imp = next((f["importance"] for f in FEATURE_IMPORTANCE if f["feature"] == feat), 0)
        scored.append((feat, val, abs(val) * imp))
    scored.sort(key=lambda x: -x[2])
    top = scored[:5]
    return [
        {
            "feature": f,
            "value": round(val, 3),
            "direction": "increases risk" if val < 0 else "typical / lowers risk"
            if f.startswith("V") else "neutral",
        }
        for f, val, _ in top
    ]


def score_with_persona(base_proba, persona_cluster_id):
    """Apply the persona risk multiplier to get an adaptive effective threshold,
    then classify into a risk band using both raw model probability and the
    persona-adjusted verdict."""
    try:
        persona_cluster_id = int(persona_cluster_id)
    except (ValueError, TypeError):
        persona_cluster_id = 0
    persona = PERSONA_BY_ID.get(persona_cluster_id, PERSONAS[1])
    effective_threshold = min(0.97, max(0.03, BASE_THRESHOLD * persona["risk_multiplier"]))
    flagged = base_proba >= effective_threshold
    if base_proba >= 0.85:
        band = "critical"
    elif base_proba >= effective_threshold:
        band = "high"
    elif base_proba >= effective_threshold * 0.5:
        band = "watch"
    else:
        band = "low"
    return {
        "persona": persona["name"],
        "persona_color": persona["color"],
        "risk_multiplier": persona["risk_multiplier"],
        "effective_threshold": round(effective_threshold, 3),
        "flagged": bool(flagged),
        "risk_band": band,
    }


# ---------------------------------------------------------------------------
# Routes: static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ---------------------------------------------------------------------------
# API: dashboard summary
# ---------------------------------------------------------------------------
@app.route("/api/overview")
def api_overview():
    return jsonify({
        "app_name": "FinGuard AI",
        "tagline": "Persona-Adaptive Fraud Detection & Risk Intelligence",
        "model_metrics": METRICS,
        "personas": PERSONAS,
        "dataset_info": {
            "fraud_dataset_rows": 284807,
            "fraud_cases": 492,
            "customer_dataset_rows": 2240,
            "fraud_rate_pct": round(492 / 284807 * 100, 3),
        },
        "live_stats": STATE["stats"],
    })


@app.route("/api/model/performance")
def api_model_performance():
    return jsonify({
        "metrics": METRICS,
        "feature_importance": FEATURE_IMPORTANCE[:12],
    })


@app.route("/api/personas")
def api_personas():
    return jsonify(PERSONAS)


# ---------------------------------------------------------------------------
# API: live transaction stream simulation
# ---------------------------------------------------------------------------
@app.route("/api/stream/next")
def api_stream_next():
    n = int(request.args.get("n", 1))
    results = []
    for _ in range(n):
        if STATE["cursor"] >= len(SAMPLE_TRANSACTIONS):
            STATE["cursor"] = 0  # loop the demo feed
        txn = SAMPLE_TRANSACTIONS[STATE["cursor"]]
        STATE["cursor"] += 1

        proba = txn["fraud_probability"]
        persona_id = txn["persona_cluster"]
        verdict = score_with_persona(proba, persona_id)
        explanation = explain_transaction(txn["features"])

        actual = txn["actual_class"]
        predicted = 1 if verdict["flagged"] else 0
        STATE["stats"]["total"] += 1
        if predicted == 1:
            STATE["stats"]["flagged"] += 1
        if predicted == 1 and actual == 1:
            STATE["stats"]["true_positive"] += 1
        elif predicted == 1 and actual == 0:
            STATE["stats"]["false_positive"] += 1
        elif predicted == 0 and actual == 1:
            STATE["stats"]["false_negative"] += 1
        else:
            STATE["stats"]["true_negative"] += 1

        record = {
            "txn_id": txn["txn_id"],
            "customer_id": txn["customer_id"],
            "amount": txn["amount"],
            "hour": txn["hour"],
            "timestamp": (datetime.now()).strftime("%H:%M:%S"),
            "fraud_probability": proba,
            "actual_class": actual,
            "explanation": explanation,
            **verdict,
        }
        STATE["processed_log"].append(record)
        STATE["processed_log"] = STATE["processed_log"][-200:]  # cap memory
        results.append(record)

    return jsonify({"transactions": results, "stats": STATE["stats"]})


@app.route("/api/stream/reset", methods=["POST"])
def api_stream_reset():
    STATE["cursor"] = 0
    STATE["processed_log"] = []
    STATE["stats"] = {"total": 0, "flagged": 0, "true_positive": 0,
                       "false_positive": 0, "false_negative": 0, "true_negative": 0}
    return jsonify({"ok": True})


@app.route("/api/transactions/recent")
def api_transactions_recent():
    limit = int(request.args.get("limit", 25))
    return jsonify(STATE["processed_log"][-limit:][::-1])


# ---------------------------------------------------------------------------
# API: "Try it yourself" manual simulator
# ---------------------------------------------------------------------------
@app.route("/api/simulate/ranges")
def api_simulate_ranges():
    """Ranges for the interactive sliders (top contributing features only)."""
    top = FEATURE_IMPORTANCE[:6]
    ranges = {f["feature"]: FEATURE_RANGES.get(f["feature"], {"min": -5, "max": 5, "median": 0})
              for f in top}
    return jsonify({"ranges": ranges, "order": [f["feature"] for f in top]})


@app.route("/api/simulate/predict", methods=["POST"])
def api_simulate_predict():
    payload = request.get_json(force=True)
    amount = float(payload.get("amount", 100))
    hour = int(payload.get("hour", 12))
    persona_id = int(payload.get("persona_cluster", 1))
    overrides = payload.get("features", {})  # e.g. {"V14": -8.2, "V4": 5.1}

    # Build the full feature vector: defaults from training medians, overridden
    # by whatever the user adjusted on the sliders
    row = {}
    for feat in FEATURE_COLS:
        if feat == "Amount_scaled":
            row[feat] = (amount - 88.35) / 250.12  # approx scaling from training stats
        elif feat == "Hour":
            row[feat] = hour
        else:
            row[feat] = overrides.get(feat, FEATURE_RANGES.get(feat, {}).get("median", 0))

    X = pd.DataFrame([[row[f] for f in FEATURE_COLS]], columns=FEATURE_COLS)
    proba = float(fraud_model.predict_proba(X)[0, 1])
    verdict = score_with_persona(proba, persona_id)
    explanation = explain_transaction(row)

    return jsonify({
        "fraud_probability": round(proba, 4),
        "explanation": explanation,
        **verdict,
    })


if __name__ == "__main__":
    print("=" * 60)
    print(" FinGuard AI backend running -> http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
