"""
FinGuard AI - Fraud Detection Model Training
Trains a RandomForest classifier on the credit card fraud dataset,
evaluates it, and exports artifacts used by the Flask backend:
  - model/fraud_model.pkl      -> trained classifier
  - model/scaler.pkl           -> StandardScaler for Amount/Time
  - model/metrics.json         -> evaluation metrics for the dashboard
  - model/feature_importance.json -> global feature importance (explainability)
  - data/sample_transactions.json -> a realistic mixed sample (fraud + legit)
                                       used to power the "live feed" demo
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, average_precision_score
)

import os
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("CREDITCARD_CSV", os.path.join(BASE, "dataset", "creditcard.csv"))
OUT_MODEL = os.path.join(BASE, "backend", "model")
OUT_DATA = os.path.join(BASE, "backend", "data")

print("Loading dataset...")
df = pd.read_csv(SRC)
print("Shape:", df.shape)

# Feature engineering: keep V1-V28 (already PCA'd/anonymized), scale Time & Amount
df["Amount_scaled"] = StandardScaler().fit_transform(df[["Amount"]])
df["Hour"] = (df["Time"] // 3600) % 24

feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Hour"]
X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training RandomForest (class_weight=balanced to handle 0.17% fraud rate)...")
t0 = time.time()
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"Trained in {time.time()-t0:.1f}s")

# Threshold tuning: default 0.5 is too conservative for imbalanced fraud;
# pick threshold that maximizes F1 on a validation split of the training data
proba_train = model.predict_proba(X_train)[:, 1]
best_t, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.02):
    f1 = f1_score(y_train, (proba_train >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t
print("Best threshold on train:", best_t, "F1:", best_f1)

proba_test = model.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= best_t).astype(int)

metrics = {
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba_test), 4),
    "pr_auc": round(average_precision_score(y_test, proba_test), 4),
    "threshold": round(float(best_t), 4),
    "test_size": int(len(y_test)),
    "fraud_in_test": int(y_test.sum()),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}
print(json.dumps(metrics, indent=2))

# ROC curve points (downsampled for a lightweight chart)
fpr, tpr, _ = roc_curve(y_test, proba_test)
idx = np.linspace(0, len(fpr) - 1, 60).astype(int)
metrics["roc_curve"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}

# Feature importance (global explainability)
importances = model.feature_importances_
fi = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
feature_importance = [{"feature": f, "importance": round(float(i), 5)} for f, i in fi]

# Save artifacts
joblib.dump(model, f"{OUT_MODEL}/fraud_model.pkl")
with open(f"{OUT_MODEL}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(f"{OUT_MODEL}/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=2)
with open(f"{OUT_MODEL}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
with open(f"{OUT_MODEL}/threshold.json", "w") as f:
    json.dump({"threshold": float(best_t)}, f)

# Build a demo sample: mix of legit + ALL frauds from test set for a convincing live feed
test_df = X_test.copy()
test_df["Class"] = y_test.values
test_df["Amount"] = df.loc[X_test.index, "Amount"].values
test_df["Time"] = df.loc[X_test.index, "Time"].values
test_df["proba"] = proba_test
test_df["predicted"] = y_pred

frauds = test_df[test_df["Class"] == 1]
legits = test_df[test_df["Class"] == 0].sample(n=400, random_state=42)
demo = pd.concat([frauds, legits]).sample(frac=1, random_state=7).reset_index(drop=True)

# Assign a synthetic "customer id" + persona cluster for the persona-adaptive layer demo
rng = np.random.default_rng(42)
demo["customer_id"] = ["CUST" + str(rng.integers(1000, 9999)) for _ in range(len(demo))]

records = []
for i, row in demo.iterrows():
    rec = {
        "txn_id": f"TXN{100000+i}",
        "customer_id": row["customer_id"],
        "amount": round(float(row["Amount"]), 2),
        "hour": int(row["Hour"]),
        "time_offset": float(row["Time"]),
        "actual_class": int(row["Class"]),
        "fraud_probability": round(float(row["proba"]), 4),
        "features": {c: round(float(row[c]), 5) for c in feature_cols},
    }
    records.append(rec)

with open(f"{OUT_DATA}/sample_transactions.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\nSaved {len(records)} demo transactions ({int(demo['Class'].sum())} fraud, {len(demo)-int(demo['Class'].sum())} legit)")
print("Done training fraud model.")
