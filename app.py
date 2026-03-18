import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Risk Intelligence Engine", layout="wide")

st.title("🔍 AI Risk Intelligence Engine")
st.write("**Pre-transaction anomaly detection for public finance and payment disbursement systems.**")

st.info(
    "This engine demonstrates the methodology applicable to public sector "
    "payment and disbursement systems — including conditional cash transfer "
    "programmes, government procurement systems, and fintech payment gateways. "
    "It uses the Isolation Forest algorithm to detect behavioural anomalies "
    "prior to transaction execution, shifting financial governance from "
    "reactive audit to proactive risk mitigation."
)

st.write("---")
st.write("Upload a structured transaction CSV file to generate fraud risk scores and anomaly flags.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Separate labels if present
    if "Class" in data.columns:
        labels = data["Class"].values
        features = data.drop(columns=["Class"])
    else:
        labels = None
        features = data.copy()

    # Keep only numeric columns
    features = features.select_dtypes(include=[np.number]).fillna(0)

    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Train Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(scaled)
    anomaly_flag = iso.predict(scaled)
    anomaly_scores = iso.decision_function(scaled)
    risk_score = (
        (1 - (anomaly_scores - anomaly_scores.min()) /
         (anomaly_scores.max() - anomaly_scores.min())) * 100
    ).round(2)
    anomaly_label = ["Anomaly" if x == -1 else "Normal" for x in anomaly_flag]

    # Build results dataframe
    results = features.copy()
    results["Risk Score"] = risk_score
    results["Anomaly Flag"] = anomaly_label
    if labels is not None:
        results["Actual Class"] = labels
    results = results.sort_values("Risk Score", ascending=False)

    # ── METRICS ──
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(results):,}")
    col2.metric("Anomalies Detected", f"{(results['Anomaly Flag'] == 'Anomaly').sum():,}")
    col3.metric("Avg Risk Score", f"{results['Risk Score'].mean():.1f}")

    # ── FILTER ──
    st.subheader("Filter High-Risk Cases")
    threshold = st.slider("Show cases with Risk Score above:", 0, 100, 70)
    filtered = results[results["Risk Score"] >= threshold]
    st.write(f"**{len(filtered):,} records** above threshold")
    st.dataframe(filtered.head(50))

    # ── RISK DISTRIBUTION ──
    st.subheader("Risk Score Distribution")
    st.bar_chart(results["Risk Score"].value_counts().sort_index())

    # ── MODEL PERFORMANCE ──
    st.write("---")
    st.subheader("📊 Model Performance Analysis")
    st.write(
        "Where ground truth labels are available (column named 'Class'), "
        "the engine evaluates model performance using standard classification metrics."
    )

    if labels is not None:
        # Convert Isolation Forest output to binary (1 = anomaly, 0 = normal)
        predicted = [1 if x == -1 else 0 for x in anomaly_flag]
        actual = labels

        # Classification report
        report = classification_report(
            actual, predicted,
            target_names=["Normal", "Fraud"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose().round(3)

        st.write("#### Classification Report")
        st.dataframe(report_df)

        # Key metrics
        precision = report["Fraud"]["precision"]
        recall = report["Fraud"]["recall"]
        f1 = report["Fraud"]["f1-score"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (Fraud)", f"{precision:.1%}",
                    help="Of all cases flagged as fraud, how many were actually fraud")
        col2.metric("Recall (Fraud)", f"{recall:.1%}",
                    help="Of all actual fraud cases, how many did the model catch")
        col3.metric("F1 Score (Fraud)", f"{f1:.1%}",
                    help="Harmonic mean of precision and recall")

        st.write("#### Confusion Matrix")
        cm = confusion_matrix(actual, predicted)
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Normal", "Predicted Fraud"])
        ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=12)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)

        st.write("#### Interpretation")
        st.write(
            f"The model identified **{recall:.1%} of actual fraud cases** in this dataset. "
            f"In a public finance disbursement context, high recall is the priority — "
            f"it is more costly to miss a fraudulent transaction than to flag a legitimate "
            f"one for review. Precision of **{precision:.1%}** means that of every 100 "
            f"cases flagged, approximately {int(precision*100)} are genuine anomalies."
        )

    else:
        st.info(
            "No ground truth labels detected in this dataset. "
            "To enable model performance analysis, include a column named 'Class' "
            "where 1 = fraudulent transaction and 0 = normal transaction."
        )

    # ── DOWNLOAD ──
    st.write("---")
    csv_output = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download High-Risk Cases as CSV",
        data=csv_output,
        file_name="high_risk_cases.csv",
        mime="text/csv"
    )

else:
    st.write("---")
    st.write("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.write("**1. Upload**\nUpload any structured transaction CSV file")
    col2.write("**2. Analyse**\nIsolation Forest detects anomalies in real time")
    col3.write("**3. Export**\nDownload flagged high-risk cases for investigation")

    st.write("---")
    st.write("### Applicable Contexts")
    col1, col2, col3 = st.columns(3)
    col1.write("**Public Finance**\nConditional cash transfer programmes, government disbursements")
    col2.write("**Fintech**\nMobile banking, payment gateways, digital wallets")
    col3.write("**Healthcare Finance**\nBeneficiary payment systems, procurement fraud detection")
