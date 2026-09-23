"""
generate_data.py
-----------------
Creates a small SAMPLE credit-card transaction dataset using REAL,
human-understandable features (no anonymized "V" columns):

    Amount               - transaction amount in dollars
    Hour                 - hour of the day the transaction happened (0-23)
    Distance_From_Home   - distance in km between the cardholder's home
                            and where the transaction took place
    Is_Foreign           - 1 if the transaction happened in a foreign
                            country, 0 otherwise
    Is_Online            - 1 if it was an online/card-not-present
                            purchase, 0 if in-person/card-present
    Class                - 1 = fraud, 0 = normal (the label we predict)

Each feature has an obvious, explainable connection to fraud risk:
large amounts, unusual hours, transactions far from home, foreign
transactions, and online purchases are all classic real-world fraud
indicators.

Run once to (re)create data/creditcard_sample.csv:
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_NORMAL = 450
N_FRAUD = 50          # ~10% fraud rate -> easy for a beginner model to learn
N_TOTAL = N_NORMAL + N_FRAUD

# ---- Normal transactions ----
# Small-ish amounts, daytime hours, close to home, mostly domestic,
# mostly in-person.
normal = pd.DataFrame({
    "Amount": np.round(np.random.gamma(2.0, 40, N_NORMAL), 2),
    "Hour": np.clip(np.random.normal(14, 4, N_NORMAL).astype(int), 0, 23),
    "Distance_From_Home": np.round(np.random.gamma(1.2, 5, N_NORMAL), 1),
    "Is_Foreign": np.random.binomial(1, 0.05, N_NORMAL),
    "Is_Online": np.random.binomial(1, 0.30, N_NORMAL),
    "Class": 0,
})

# ---- Fraudulent transactions ----
# Bigger amounts, odd/late-night hours, far from home, more often
# foreign and online (classic "card-not-present" fraud pattern).
fraud = pd.DataFrame({
    "Amount": np.round(np.random.gamma(3.0, 150, N_FRAUD), 2),
    "Hour": np.random.choice(
        list(range(0, 5)) + list(range(22, 24)), size=N_FRAUD
    ),
    "Distance_From_Home": np.round(np.random.gamma(2.5, 60, N_FRAUD), 1),
    "Is_Foreign": np.random.binomial(1, 0.65, N_FRAUD),
    "Is_Online": np.random.binomial(1, 0.80, N_FRAUD),
    "Class": 1,
})

# ---- Add realistic overlap so the classes aren't perfectly separable ----
# Real fraud detection is never 100% clear-cut. A handful of "hard cases"
# in each class keep the model honest and make the threshold slider (and
# the precision/recall trade-off) actually meaningful to explore.

# Some fraud is subtle: normal-looking amount/hour/distance (e.g. a stolen
# card used carefully, in-person, nearby).
n_subtle_fraud = int(N_FRAUD * 0.30)
subtle_idx = np.random.choice(fraud.index, size=n_subtle_fraud, replace=False)
fraud.loc[subtle_idx, "Amount"] = np.round(np.random.gamma(2.0, 45, n_subtle_fraud), 2)
fraud.loc[subtle_idx, "Hour"] = np.clip(np.random.normal(14, 4, n_subtle_fraud).astype(int), 0, 23)
fraud.loc[subtle_idx, "Distance_From_Home"] = np.round(np.random.gamma(1.2, 6, n_subtle_fraud), 1)
fraud.loc[subtle_idx, "Is_Foreign"] = np.random.binomial(1, 0.10, n_subtle_fraud)
fraud.loc[subtle_idx, "Is_Online"] = np.random.binomial(1, 0.35, n_subtle_fraud)

# Some normal transactions look risky on paper: a big purchase while
# traveling abroad, late at night, bought online (e.g. genuine vacation
# shopping) -- these are the realistic "false alarm" candidates.
n_risky_normal = int(N_NORMAL * 0.08)
risky_idx = np.random.choice(normal.index, size=n_risky_normal, replace=False)
normal.loc[risky_idx, "Amount"] = np.round(np.random.gamma(3.0, 100, n_risky_normal), 2)
normal.loc[risky_idx, "Hour"] = np.random.choice(list(range(21, 24)) + list(range(0, 3)), size=n_risky_normal)
normal.loc[risky_idx, "Distance_From_Home"] = np.round(np.random.gamma(2.0, 45, n_risky_normal), 1)
normal.loc[risky_idx, "Is_Foreign"] = np.random.binomial(1, 0.55, n_risky_normal)
normal.loc[risky_idx, "Is_Online"] = np.random.binomial(1, 0.60, n_risky_normal)

df = pd.concat([normal, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

df.to_csv("data/creditcard_sample.csv", index=False)
print(f"Saved data/creditcard_sample.csv with {len(df)} rows "
      f"({df['Class'].sum()} fraud / {len(df) - df['Class'].sum()} normal)")
