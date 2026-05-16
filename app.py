import json
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "fraud_model.pkl"
METRICS_PATH = "model_metrics.json"
FRAUD_THRESHOLD = 0.20
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


st.set_page_config(page_title="AI Fraud Detection", page_icon="🔎", layout="centered")
st.title("AI Fraud Detection System")
st.write("Enter transaction values to predict fraud using the trained XGBoost model.")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    with st.expander("Model Performance Metrics (XGBoost)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col2.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")
        col3.metric("F1-Score", f"{metrics['f1_weighted'] * 100:.2f}%")
        col4.metric("Precision", f"{metrics['precision_weighted'] * 100:.2f}%")

        st.markdown("**Per-Class Metrics**")
        per_class_df = pd.DataFrame(
            {
                "Class": ["Legitimate (0)", "Fraud (1)"],
                "Precision": [
                    f"{metrics['per_class']['legitimate']['precision'] * 100:.2f}%",
                    f"{metrics['per_class']['fraud']['precision'] * 100:.2f}%",
                ],
                "Recall": [
                    f"{metrics['per_class']['legitimate']['recall'] * 100:.2f}%",
                    f"{metrics['per_class']['fraud']['recall'] * 100:.2f}%",
                ],
                "F1-Score": [
                    f"{metrics['per_class']['legitimate']['f1_score'] * 100:.2f}%",
                    f"{metrics['per_class']['fraud']['f1_score'] * 100:.2f}%",
                ],
                "Support": [
                    f"{metrics['per_class']['legitimate']['support']:,}",
                    f"{metrics['per_class']['fraud']['support']:,}",
                ],
            }
        )
        st.dataframe(per_class_df, use_container_width=True, hide_index=True)

        st.markdown("**Confusion Matrix**")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Legitimate", "Actual: Fraud"],
            columns=["Predicted: Legitimate", "Predicted: Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Run `python3 train_xgboost.py` to refresh metrics after retraining.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("XGBoost model file not found.")
    st.info("Run `python3 train_xgboost.py` first to create `fraud_model.pkl`.")
    st.stop()


model = load_model()

with st.form("fraud_prediction_form"):
    amount = st.number_input("Transaction Amount", min_value=0.0, step=100.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, step=100.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, step=100.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, step=100.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, step=100.0)

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_data = pd.DataFrame(
        [[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
        columns=FEATURES,
    )

    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(input_data)[0][1]
    else:
        fraud_probability = float(model.predict(input_data)[0])

    prediction = int(fraud_probability >= FRAUD_THRESHOLD)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")

    st.metric("Fraud Probability", f"{fraud_probability * 100:.2f}%")
    st.caption(f"Fraud threshold: {FRAUD_THRESHOLD * 100:.0f}%")
    st.dataframe(input_data, use_container_width=True)
