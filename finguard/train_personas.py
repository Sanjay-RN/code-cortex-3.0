"""
FinGuard AI - Customer Persona Segmentation
Clusters customers from marketing_campaign.csv into behavioral personas using
KMeans, then assigns each persona a "risk sensitivity multiplier" that the
fraud engine uses to adapt its alert threshold per customer type.

This is the core unique differentiator of FinGuard AI: instead of one static
fraud threshold for every transaction, the system asks "is this unusual FOR
THIS TYPE OF CUSTOMER?" -- a $2,000 online purchase is business-as-usual for
a "Premium Digital Spender" but a red flag for a "Cautious Saver".

Outputs:
  - model/persona_model.pkl     -> fitted KMeans + scaler (bundled)
  - model/personas.json         -> persona definitions, stats, risk multipliers
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import os
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("MARKETING_CSV", os.path.join(BASE, "dataset", "marketing_campaign.csv"))
OUT_MODEL = os.path.join(BASE, "backend", "model")

df = pd.read_csv(SRC, sep="\t")
df["Income"] = df["Income"].fillna(df["Income"].median())

df["Age"] = 2026 - df["Year_Birth"]
df["TotalSpend"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)
df["TotalPurchases"] = (
    df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
)
df["DigitalRatio"] = df["NumWebPurchases"] / df["TotalPurchases"].replace(0, 1)

# Drop extreme outliers (data quality issues in this well-known dataset)
df = df[(df["Age"] < 100) & (df["Income"] < 200000)]

features = ["Income", "Age", "TotalSpend", "TotalPurchases", "DigitalRatio", "NumWebVisitsMonth"]
X = df[features].copy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(Xs)

# Describe clusters, then hand-name them based on their statistical profile
cluster_stats = df.groupby("cluster")[features].mean().round(1)
sizes = df["cluster"].value_counts().sort_index()

print(cluster_stats)
print(sizes)

# Rank clusters by income & digital ratio to assign meaningful, presentable names
ranked = cluster_stats.sort_values("Income")
persona_library = [
    {
        "name": "Cautious Saver",
        "description": "Lower discretionary spend, infrequent purchases, prefers predictable patterns. Small deviations in spend are meaningful.",
        "risk_multiplier": 1.35,  # more sensitive -> lower effective threshold
        "color": "#3b82f6",
    },
    {
        "name": "Steady Spender",
        "description": "Moderate income and balanced spending across categories. Represents typical baseline behavior.",
        "risk_multiplier": 1.0,
        "color": "#10b981",
    },
    {
        "name": "Digital-First Shopper",
        "description": "High web-purchase ratio and frequent site visits. Comfortable with online/high-frequency transactions.",
        "risk_multiplier": 0.85,
        "color": "#f59e0b",
    },
    {
        "name": "Premium High-Spender",
        "description": "High income and high total spend across categories, including premium/gold products. Large transactions are normal.",
        "risk_multiplier": 0.65,  # less sensitive -> higher effective threshold
        "color": "#8b5cf6",
    },
]

# Sort actual clusters by income to map onto the 4 personas from low->high income
order = ranked.index.tolist()
# but Digital-First should be picked by DigitalRatio among remaining, keep simple: sort by income ascending
cluster_to_persona = {}
income_sorted = cluster_stats.sort_values("Income").index.tolist()
digital_sorted = cluster_stats.sort_values("DigitalRatio", ascending=False).index.tolist()

# Assign: lowest income -> Cautious Saver, highest income -> Premium High-Spender
# Among the middle two, higher digital ratio -> Digital-First Shopper, other -> Steady Spender
assigned = {}
remaining = list(cluster_stats.index)
lowest_income_cluster = income_sorted[0]
assigned[lowest_income_cluster] = 0  # Cautious Saver
remaining.remove(lowest_income_cluster)
highest_income_cluster = income_sorted[-1]
assigned[highest_income_cluster] = 3  # Premium High-Spender
remaining.remove(highest_income_cluster)
# of remaining 2, higher digital ratio -> Digital-First
remaining_sorted_digital = cluster_stats.loc[remaining].sort_values("DigitalRatio", ascending=False).index.tolist()
assigned[remaining_sorted_digital[0]] = 2  # Digital-First Shopper
assigned[remaining_sorted_digital[1]] = 1  # Steady Spender

personas_output = []
for cluster_id, persona_idx in assigned.items():
    p = dict(persona_library[persona_idx])
    p["cluster_id"] = int(cluster_id)
    p["size"] = int(sizes[cluster_id])
    p["pct_of_customers"] = round(100 * sizes[cluster_id] / sizes.sum(), 1)
    p["avg_income"] = float(cluster_stats.loc[cluster_id, "Income"])
    p["avg_age"] = float(cluster_stats.loc[cluster_id, "Age"])
    p["avg_total_spend"] = float(cluster_stats.loc[cluster_id, "TotalSpend"])
    p["avg_digital_ratio"] = round(float(cluster_stats.loc[cluster_id, "DigitalRatio"]), 2)
    personas_output.append(p)

personas_output.sort(key=lambda p: p["risk_multiplier"], reverse=True)

joblib.dump({"scaler": scaler, "kmeans": km, "features": features, "cluster_map": assigned}, f"{OUT_MODEL}/persona_model.pkl")
with open(f"{OUT_MODEL}/personas.json", "w") as f:
    json.dump(personas_output, f, indent=2)

print(json.dumps(personas_output, indent=2))
print("Saved persona model + definitions.")
