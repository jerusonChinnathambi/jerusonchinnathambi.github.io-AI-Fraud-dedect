import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATASET_PATH = "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = "fraud_model.pkl"
METRICS_PATH = "model_metrics.json"
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

X = df[FEATURES]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision_w = precision_score(y_test, predictions, average="weighted")
recall_w = recall_score(y_test, predictions, average="weighted")
f1_w = f1_score(y_test, predictions, average="weighted")
roc_auc = roc_auc_score(y_test, proba)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)

print(f"\nXGBoost Accuracy:  {accuracy:.4f}")
print(f"Precision (weighted): {precision_w:.4f}")
print(f"Recall (weighted):    {recall_w:.4f}")
print(f"F1-score (weighted):  {f1_w:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision_weighted": round(float(precision_w), 4),
    "recall_weighted": round(float(recall_w), 4),
    "f1_weighted": round(float(f1_w), 4),
    "roc_auc": round(float(roc_auc), 4),
    "confusion_matrix": cm.tolist(),
    "per_class": {
        "legitimate": {
            "precision": round(float(report["0"]["precision"]), 4),
            "recall": round(float(report["0"]["recall"]), 4),
            "f1_score": round(float(report["0"]["f1-score"]), 4),
            "support": int(report["0"]["support"]),
        },
        "fraud": {
            "precision": round(float(report["1"]["precision"]), 4),
            "recall": round(float(report["1"]["recall"]), 4),
            "f1_score": round(float(report["1"]["f1-score"]), 4),
            "support": int(report["1"]["support"]),
        },
    },
}

joblib.dump(model, MODEL_PATH)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Metrics saved to {METRICS_PATH}")
