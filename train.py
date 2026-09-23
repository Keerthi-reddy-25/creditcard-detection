"""
train.py
--------
Trains TWO beginner-friendly models to detect credit-card fraud, using
REAL, explainable features (Amount, Hour, Distance_From_Home, Is_Foreign,
Is_Online) instead of anonymized "V" columns:
  1. Logistic Regression
  2. Random Forest

Both models are trained on the exact same train/test split, so the
comparison between them is fair.

Steps:
1. Load data/creditcard_sample.csv
2. Split into train/test sets (same split used for both models)
3. Scale features (mainly needed for Logistic Regression)
4. Train Logistic Regression and Random Forest
5. Save:
     - model/model.pkl -> {"models": {...}, "scaler": ..., "features": [...]}
     - model/test_predictions.csv -> true labels + each model's predicted
       probability for the held-out test set (used later by the API to
       recompute accuracy / recall / confusion matrix at any threshold,
       for either model, instantly and without retraining).

Run:
    python train.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Amount", "Hour", "Distance_From_Home", "Is_Foreign", "Is_Online"]
TARGET = "Class"

# Model registry: key -> (display name, estimator)
MODEL_REGISTRY = {
    "logistic_regression": ("Logistic Regression", LogisticRegression(random_state=42)),
    "random_forest": ("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )),
}


def main():
    # 1. Load data
    df = pd.read_csv("data/creditcard_sample.csv")
    X = df[FEATURES]
    y = df[TARGET]

    # 2. Train/test split (stratify keeps the same fraud ratio in both sets)
    #    Both models use this exact same split -> fair comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train each model and collect its test-set probabilities
    trained_models = {}
    test_out = X_test.copy()
    test_out["Class"] = y_test.values

    for key, (display_name, estimator) in MODEL_REGISTRY.items():
        estimator.fit(X_train_scaled, y_train)
        probs = estimator.predict_proba(X_test_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        print(f"\n--- {display_name} ---")
        print("Test accuracy @0.5:", round(accuracy_score(y_test, preds), 3))
        print("Test recall   @0.5:", round(recall_score(y_test, preds), 3))
        print("Confusion matrix @0.5:\n", confusion_matrix(y_test, preds))

        trained_models[key] = estimator
        test_out[f"Probability_{key}"] = probs

    # 5a. Save all models + scaler together
    joblib.dump(
        {"models": trained_models, "scaler": scaler, "features": FEATURES,
         "model_names": {k: v[0] for k, v in MODEL_REGISTRY.items()}},
        "model/model.pkl",
    )

    # 5b. Save test set probabilities (per model) so the API can recompute
    #     metrics for ANY threshold, for ANY model, instantly.
    test_out.to_csv("model/test_predictions.csv", index=False)

    print("\nSaved model/model.pkl and model/test_predictions.csv")


if __name__ == "__main__":
    main()
