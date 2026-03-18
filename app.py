import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Risk Intelligence Engine", layout="wide")

st.title("AI Risk Intelligence Engine")
st.write("Upload a CSV file to generate fraud risk scores and anomaly flags.")

# Load trained models
model = joblib.load("model.pkl")
iso = joblib.load("iso.pkl")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Remove target column if present
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Generate risk scores
    risk_prob = model.predict_proba(data)[:, 1]
    risk_score = (risk_prob * 100).round(2)

    # Generate anomaly flags
    anomaly_flag = iso.predict(data)
    anomaly_flag = ["Anomaly" if x == -1 else "Normal" for x in anomaly_flag]

    # Add results
    data["Risk Score"] = risk_score
    data["Anomaly Flag"] = anomaly_flag

    # Sort by highest risk
    data = data.sort_values("Risk Score", ascending=False)

    # Filter section
    st.subheader("Filter High-Risk Cases")

    threshold = st.slider("Show cases with Risk Score above:", 0, 100, 70)

    filtered = data[data["Risk Score"] >= threshold]

    # Quick stats
    st.write("### Quick Stats")
    st.write(f"Total records: {len(data)}")
    st.write(f"High-risk (>= {threshold}): {len(filtered)}")
    st.write(f"Anomalies flagged: {(data['Anomaly Flag'] == 'Anomaly').sum()}")

    st.dataframe(filtered.head(50))

    # Download filtered results
    csv_output = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered results as CSV",
        data=csv_output,
        file_name="risk_results_filtered.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")
