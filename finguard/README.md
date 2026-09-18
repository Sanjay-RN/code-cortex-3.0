# FinGuard AI
### Persona-Adaptive Fraud Detection & Risk Intelligence Platform
Built for **CodeCortex 3.0** — Track T1: Finance

---

## 1. The idea, in one sentence

Every fraud detector on the market uses **one fixed threshold for every customer**.
FinGuard AI is different: it learns **who each customer is** (a "Cautious Saver," a
"Premium High-Spender," etc.) from their spending behavior, and **adapts the fraud
alert threshold to that persona** — so a $2,000 purchase that's normal for a high
spender doesn't get flagged, while the same amount is treated as suspicious for
someone who never spends that much.

This is the "unique angle" for judges: not just "we detect fraud" (every team will
say that), but **"we detect fraud the way a human fraud analyst would — by knowing
the customer, not just the transaction."**

---

## 2. Datasets used and why

Out of the four datasets provided, two were combined (the other two — NIFTY-50
and the raw bank statement — were evaluated but not used; see §6):

| Dataset | Rows | Role |
|---|---|---|
| `creditcard.csv` | 284,807 transactions, 492 confirmed frauds | Trains the core fraud-detection ML model |
| `marketing_campaign.csv` | 2,240 customers | Trains the customer persona / segmentation model |

**Why these two:** `creditcard.csv` is the only dataset with **ground-truth fraud
labels**, which is what makes a supervised fraud model possible and measurable
(precision/recall/ROC-AUC). `marketing_campaign.csv` is the richest dataset for
**customer behavior** (income, spend by category, channel preference), which is
exactly what's needed to build believable personas.

> **Honesty note for judges:** these are two separate public datasets and don't
> share real customer IDs. In the live demo, each simulated transaction is
> assigned a persona to *demonstrate the mechanism*. In a real deployment, a bank
> already has both pieces of data on the same customer, so persona and
> transaction would be linked for real — this is a proof-of-concept of the
> **method**, which is honest and typical of hackathon scope, not a limitation of
> the idea itself. Say this proactively in your pitch — it shows maturity, not
> weakness.

---

## 3. What's actually in the app

### 3.1 Fraud detection engine
- RandomForest classifier trained on all 284,807 transactions (`class_weight="balanced_subsample"`
  to handle the severe 0.17% fraud rate).
- Held-out test evaluation (25% split, 71,202 transactions the model never saw):
  - **Precision: 85.7%** — when it flags something, it's usually right
  - **Recall: 78.0%** — it catches most of the fraud
  - **ROC-AUC: 0.973**
  - Full confusion matrix and ROC curve shown live in the "Model Performance" tab
- Threshold auto-tuned to maximize F1 rather than using the naive 0.5 cutoff
  (essential for imbalanced fraud data — this is a talking point judges like).

### 3.2 Persona segmentation
- K-Means clustering (k=4) on income, age, total spend, purchase count, and
  digital-channel ratio.
- Four named personas, each with a **risk multiplier** that shifts the effective
  fraud threshold up or down:

| Persona | Behavior | Risk multiplier | Effect |
|---|---|---|---|
| Cautious Saver | Low spend, infrequent purchases | 1.35× | More sensitive — flags smaller deviations |
| Steady Spender | Balanced, moderate spend | 1.0× | Baseline |
| Digital-First Shopper | High web activity | 0.85× | Slightly more tolerant of online activity |
| Premium High-Spender | High income, high spend | 0.65× | Large transactions are normal, less sensitive |

### 3.3 Explainability
Every scored transaction ships with its **top 5 contributing features** ranked by
`importance × magnitude` — a lightweight, dependency-free stand-in for full SHAP
values that's fast enough for a live demo and easy to explain to judges: *"the
model isn't a black box — here's exactly what made it suspicious."*

### 3.4 Live dashboard (5 views)
1. **Overview** — session stats, live precision/recall, how the pipeline works
2. **Live Feed** — simulated real-time transaction stream (from real held-out test
   data, so frauds actually appear), click any row for a full explanation
3. **Customer Personas** — the four segments with their stats and risk profile
4. **Model Performance** — ROC curve, confusion matrix, feature importance chart
5. **Try It Yourself** — build a transaction with sliders, pick a persona, and see
   the fraud probability + verdict change live. Includes one-click "known fraud
   pattern" and "typical purchase" presets for a fast, reliable demo.

---

## 4. Architecture

```
finguard/
├── README.md                  <- this file
├── train_fraud_model.py       <- trains the RandomForest fraud model
├── train_personas.py          <- trains the K-Means persona model
├── dataset/                   <- put raw CSVs here ONLY if retraining
└── backend/
    ├── app.py                 <- Flask API + serves the frontend
    ├── requirements.txt
    ├── model/                 <- pre-trained model artifacts (already built)
    │   ├── fraud_model.pkl
    │   ├── persona_model.pkl
    │   ├── metrics.json
    │   ├── personas.json
    │   ├── feature_importance.json
    │   └── ...
    ├── data/
    │   └── sample_transactions.json   <- 523 real held-out transactions for the live demo
    └── static/                <- the dashboard (HTML/CSS/JS, no build step)
        ├── index.html
        ├── style.css
        └── app.js
```

**Stack:** Python + Flask (backend/API) · scikit-learn (RandomForest + KMeans) ·
vanilla HTML/CSS/JS + Chart.js (frontend, zero build step — just open a browser).

Everything needed to run is already trained and included. You do **not** need to
re-run the training scripts unless you want to customize the model.

---

## 5. How to run it

```bash
cd finguard/backend
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser. That's it — no database, no
build step, no API keys.

To retrain the models yourself (optional — only if you want to tweak them):
```bash
# put creditcard.csv and marketing_campaign.csv in finguard/dataset/ first
cd finguard
python train_fraud_model.py
python train_personas.py
```

---

## 6. Other datasets — evaluated but not used, and why

- **NIFTY-50 snapshot** (`stocks/`) — a single day's closing prices for 50
  stocks. No time series, no history — not enough signal to build a real
  feature on for a 30-hour hackathon. Mentioning this evaluation in your pitch
  shows judges you made a deliberate data choice, not a default one.
- **Bank statement** (`transactions/bank.xlsx`) — real transaction descriptions
  and running balances for one account, but no fraud labels, so it can't train
  or evaluate a supervised model. It's a good candidate for a *future* anomaly-
  detection extension (see §7), which you can mention if judges ask "what's
  next."

---

## 7. If you have extra time — good "future work" talking points

- Link personas to real transactions using an actual customer ID (removes the
  synthetic-assignment caveat from §2)
- Add an unsupervised anomaly-detection layer on the bank statement data for
  accounts with no fraud labels at all
- Swap the lightweight explanation heuristic for real SHAP values
- Add email/SMS alert simulation for flagged transactions

---

## 8. Presentation tips for judges

1. **Open with the problem, not the tech**: "Every fraud system treats a
   university student and a business owner the same way. We don't."
2. **Live-demo the persona switch**: in "Try It Yourself," score the exact same
   transaction as a Cautious Saver, then switch only the persona dropdown to
   Premium High-Spender, and re-score. Same numbers, different verdict — that's
   your "wow" moment.
3. **Show real numbers, not just a demo**: 85.7% precision / 78% recall / 0.973
   ROC-AUC on 71,202 held-out transactions is a legitimate, defensible model —
   say the numbers out loud.
4. **Be upfront about the persona-linkage caveat (§2)** before a judge asks —
   it reads as rigor, not a flaw.
5. Keep the live feed running in the background on a second screen/tab while
   you talk — a moving dashboard is far more convincing than a static one.

Good luck — go win this. 🏆
