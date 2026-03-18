import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Risk Intelligence Engine", layout="wide")

st.title("🔍 AI Risk Intelligence Engine")
st.write("Upload a CSV file to generate fraud risk scores and anomaly flags using live machine learning.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Remove target column if present
    if "Class" in data.columns:
        labels = data["Class"]
        features = data.drop(columns=["Class"])
    else:
        labels = None
        features = data.copy()

    # Keep only numeric columns
    features = features.select_dtypes(include=[np.number]).fillna(0)

    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Train Isolation Forest for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(scaled)
    anomaly_flag = iso.predict(scaled)
    anomaly_scores = iso.decision_function(scaled)
    risk_score = ((1 - (anomaly_scores - anomaly_scores.min()) /
                   (anomaly_scores.max() - anomaly_scores.min())) * 100).round(2)
    anomaly_label = ["Anomaly" if x == -1 else "Normal" for x in anomaly_flag]

    # Add results to dataframe
    results = features.copy()
    results["Risk Score"] = risk_score
    results["Anomaly Flag"] = anomaly_label
    if labels is not None:
        results["Actual Class"] = labels.values

    # Sort by highest risk
    results = results.sort_values("Risk Score", ascending=False)

    # Quick stats
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(results))
    col2.metric("Anomalies Detected", (results["Anomaly Flag"] == "Anomaly").sum())
    col3.metric("Avg Risk Score", f"{results['Risk Score'].mean():.1f}")

    # Filter section
    st.subheader("Filter High-Risk Cases")
    threshold = st.slider("Show cases with Risk Score above:", 0, 100, 70)
    filtered = results[results["Risk Score"] >= threshold]
    st.write(f"**{len(filtered)} records** above threshold")
    st.dataframe(filtered.head(50))

    # Risk distribution chart
    st.subheader("Risk Score Distribution")
    st.bar_chart(results["Risk Score"].value_counts().sort_index())

    # Download
    csv_output = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download High-Risk Cases as CSV",
        data=csv_output,
        file_name="high_risk_cases.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
    st.write("---")
    st.write("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.write("**1. Upload**\nUpload any transaction CSV file")
    col2.write("**2. Analyse**\nIsolation Forest model detects anomalies in real time")
    col3.write("**3. Export**\nDownload flagged high-risk cases")
