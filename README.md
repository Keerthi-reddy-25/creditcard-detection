# Credit Card Fraud Detection (Beginner ML + FastAPI Project)

A simple, self-contained fraud detection demo using **real, explainable
transaction features** (no anonymized "V" columns):
- **Models:** Logistic Regression **and** Random Forest (scikit-learn) — switchable in the UI
- **Backend:** FastAPI
- **Frontend:** Plain HTML/CSS/JavaScript (no framework)
- **Dataset:** Small synthetic sample included (`data/creditcard_sample.csv`), 500 rows, ~10% fraud

## Features Used
Unlike the real Kaggle "Credit Card Fraud Detection" dataset, which uses
anonymized `V1`-`V28` PCA columns (privacy-scrambled and not human-readable),
this version uses features anyone can understand and reason about:

| Feature | Meaning | Typical fraud pattern in this dataset |
|---|---|---|
| `Amount` | Transaction amount ($) | Fraud tends to be larger |
| `Hour` | Hour of day (0-23) | Fraud skews toward late night / early morning |
| `Distance_From_Home` | km between cardholder's home and transaction location | Fraud tends to happen further away |
| `Is_Foreign` | 1 if a foreign-country transaction | Fraud is more often foreign |
| `Is_Online` | 1 if online/card-not-present | Fraud is more often online |

Real fraud isn't perfectly separable on these alone, so the sample data
includes some deliberately "hard" cases (subtle fraud that looks normal,
and normal transactions that look risky, e.g. genuine travel purchases) —
this keeps the threshold slider's precision/recall trade-off meaningful
instead of trivial.

## Project Structure
```
fraud_detection_v3/
├── data/
│   └── creditcard_sample.csv    # sample dataset (500 rows)
├── model/                       # created by train.py
│   ├── model.pkl                # trained models + scaler
│   └── test_predictions.csv     # held-out test set probabilities (per model)
├── static/
│   ├── index.html               # frontend page
│   ├── style.css                # styling
│   └── script.js                # slider + model dropdown + fetch logic
├── generate_data.py             # (optional) regenerates the sample dataset
├── train.py                     # trains both models
├── main.py                      # FastAPI app (serves API + frontend)
├── requirements.txt
└── README.md
```

## How It Works
1. `train.py` trains Logistic Regression and Random Forest on the sample
   dataset (same train/test split for both, for a fair comparison), and
   saves each model's **test-set predicted probabilities** to
   `model/test_predictions.csv`.
2. `main.py` loads that file at startup. When you move the threshold slider
   or switch models, the browser calls `GET /metrics?threshold=X&model=Y`,
   and the backend instantly recomputes accuracy, precision, recall, and
   the confusion matrix from those stored probabilities — **no retraining
   needed**.
3. Submitting the transaction form calls `POST /predict` for both models at
   once, scaling your inputs the same way as training and returning each
   model's fraud probability.

## Setup & Run

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Regenerate the sample dataset
A dataset is already included, but you can recreate it any time:
```bash
python generate_data.py
```

### 4. Train the models
```bash
python train.py
```
This creates `model/model.pkl` and `model/test_predictions.csv`. Expected
output looks like:
```
--- Logistic Regression ---
Test accuracy @0.5: 0.913
Test recall   @0.5: 0.467
Confusion matrix @0.5:
 [[130   5]
 [  8   7]]

--- Random Forest ---
Test accuracy @0.5: 0.907
Test recall   @0.5: 0.467
...
Saved model/model.pkl and model/test_predictions.csv
```

### 5. Start the API server
```bash
uvicorn main:app --reload
```

### 6. Open the app
Go to: **http://127.0.0.1:8000**

## Using the App
1. Fill in the transaction fields — Amount, Hour, Distance from home, and
   the two checkboxes (Foreign / Online) — or use the pre-filled "risky"
   sample values, then click **"Check Transaction"**.
2. Pick a **model** (Logistic Regression or Random Forest) from the
   dropdown.
3. Drag the **threshold slider** (0.1–0.9). Watch how, in real time:
   - The **prediction** (Fraud / Not Fraud) for your transaction may flip
   - **Accuracy**, **Precision**, and **Recall** on the test set change
   - The **confusion matrix** counts shift

Switching the model dropdown re-evaluates everything for the newly
selected model, using the same held-out test set and the same transaction
you submitted — so you can directly compare Logistic Regression and
Random Forest at the same threshold.

## Understanding the Threshold Trade-off
The model outputs a probability (0–1), not a hard label. The threshold
decides the cutoff for calling something "fraud":

| Threshold | Effect |
|---|---|
| **Lower** (e.g. 0.1) | More transactions flagged as fraud → **higher recall** (catches more real fraud), but **more false positives** (genuine transactions wrongly blocked) → lower precision |
| **Higher** (e.g. 0.9) | Fewer transactions flagged as fraud → **higher precision** (fewer false alarms), but **more false negatives** (real fraud missed) → lower recall |

On this dataset, at threshold 0.1 recall is around 53% with several false
positives; at threshold 0.9 precision hits 100% but recall drops to about
33% (several real fraud cases missed). There's no universally "correct"
threshold — it's a business decision about which mistake costs more.

## API Reference

### `GET /models`
Response:
```json
{"models": [
  {"key": "logistic_regression", "name": "Logistic Regression"},
  {"key": "random_forest", "name": "Random Forest"}
]}
```

### `POST /predict`
Request body:
```json
{
  "Amount": 500,
  "Hour": 2,
  "Distance_From_Home": 180,
  "Is_Foreign": 1,
  "Is_Online": 1,
  "threshold": 0.5,
  "model": "logistic_regression"
}
```
Response:
```json
{
  "probability": 0.9445,
  "prediction": 1,
  "label": "FRAUD",
  "threshold_used": 0.5,
  "model_used": "Logistic Regression"
}
```

### `GET /metrics?threshold=0.5&model=logistic_regression`
Response:
```json
{
  "threshold": 0.5,
  "model": "logistic_regression",
  "model_name": "Logistic Regression",
  "accuracy": 0.9133,
  "precision": 0.5833,
  "recall": 0.4667,
  "confusion_matrix": {
    "true_negative": 130,
    "false_positive": 5,
    "false_negative": 8,
    "true_positive": 7
  },
  "total_test_samples": 150
}
```

## Notes for Beginners
- The dataset is **synthetic** (generated with numpy) but designed around
  real, explainable fraud indicators, rather than anonymized/PCA columns.
- Logistic Regression was chosen because it's simple to understand: it
  learns a weight for each feature and outputs a probability. Random
  Forest is included so you can compare a simple linear model against an
  ensemble of decision trees on the same data.
- Feel free to swap in real transaction data — just keep the same column
  names (`Amount`, `Hour`, `Distance_From_Home`, `Is_Foreign`, `Is_Online`,
  `Class`) or update `FEATURES` in `train.py` and the `Transaction` model
  in `main.py` to match your own columns.
