"""
main.py
-------
FastAPI backend for the Credit Card Fraud Detection demo.

Endpoints:
  GET  /                 -> serves the frontend (static/index.html)
  GET  /models           -> lists available models (for the dropdown)
  POST /predict          -> predicts fraud probability for a single
                             transaction, using the chosen model
  GET  /metrics?threshold=0.5&model=logistic_regression
                          -> recomputes accuracy, recall, precision and the
                             confusion matrix on the held-out test set for
                             the given threshold + model (instant, no
                             retraining)

Run:
    uvicorn main:app --reload
Then open:
    http://127.0.0.1:8000
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

app = FastAPI(title="Credit Card Fraud Detection API")

# ---------------------------------------------------------------------------
# Load the trained models + scaler, and the pre-computed test-set
# predictions. These are created by running `python train.py` first.
# ---------------------------------------------------------------------------
artifact = joblib.load("model/model.pkl")
MODELS = artifact["models"]              # {"logistic_regression": est, "random_forest": est}
MODEL_NAMES = artifact["model_names"]    # {"logistic_regression": "Logistic Regression", ...}
scaler = artifact["scaler"]
FEATURES = artifact["features"]

test_df = pd.read_csv("model/test_predictions.csv")
Y_TRUE = test_df["Class"].values


def probability_column(model_key: str) -> str:
    return f"Probability_{model_key}"


def validate_model_key(model_key: str):
    if model_key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Valid options: {list(MODELS.keys())}",
        )


class Transaction(BaseModel):
    Amount: float = Field(..., example=500.0, description="Transaction amount in dollars")
    Hour: int = Field(..., ge=0, le=23, example=2, description="Hour of day (0-23) the transaction occurred")
    Distance_From_Home: float = Field(..., example=180.0, description="Distance in km from the cardholder's home")
    Is_Foreign: int = Field(..., ge=0, le=1, example=1, description="1 if a foreign-country transaction, else 0")
    Is_Online: int = Field(..., ge=0, le=1, example=1, description="1 if an online/card-not-present purchase, else 0")
    threshold: float = Field(0.5, ge=0.0, le=1.0, example=0.5)
    model: str = Field("logistic_regression", example="logistic_regression")


@app.get("/models")
def list_models():
    """Returns the available models, for populating the frontend dropdown."""
    return {"models": [{"key": k, "name": v} for k, v in MODEL_NAMES.items()]}


@app.post("/predict")
def predict(txn: Transaction):
    """Predict fraud probability for a single transaction, using the chosen
    model, and classify it using the given threshold."""
    validate_model_key(txn.model)
    model = MODELS[txn.model]

    row = pd.DataFrame([[getattr(txn, f) for f in FEATURES]], columns=FEATURES)
    row_scaled = scaler.transform(row)
    probability = float(model.predict_proba(row_scaled)[0, 1])
    prediction = int(probability >= txn.threshold)

    return {
        "probability": round(probability, 4),
        "prediction": prediction,
        "label": "FRAUD" if prediction == 1 else "NOT FRAUD",
        "threshold_used": txn.threshold,
        "model_used": MODEL_NAMES[txn.model],
    }


@app.get("/metrics")
def metrics(threshold: float = 0.5, model: str = Query("logistic_regression")):
    """Recompute accuracy / precision / recall / confusion matrix on the
    held-out test set, for the given threshold AND model. Lets the frontend
    slider + dropdown update results instantly without retraining."""
    validate_model_key(model)
    probs = test_df[probability_column(model)].values
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(Y_TRUE, preds)
    prec = precision_score(Y_TRUE, preds, zero_division=0)
    rec = recall_score(Y_TRUE, preds, zero_division=0)
    cm = confusion_matrix(Y_TRUE, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "model": model,
        "model_name": MODEL_NAMES[model],
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "total_test_samples": int(len(Y_TRUE)),
    }


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
