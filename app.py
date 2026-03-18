import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="AI Risk Intelligence Engine", layout="wide")

st.title("🔍 AI Risk Intelligence Engine")
st.write("**Pre-transaction anomaly detection for public finance and payment disbursement systems.**")

st.info(
    "This engine demonstrates the methodology applicable to public sector payment and "
    "disbursement systems — including conditional cash transfer programmes, government "
    "procurement systems, and fintech payment gateways. It uses the Isolation Forest "
    "algorithm to detect behavioural anomalies prior to transaction execution, shifting "
    "financial governance from reactive audit to proactive risk mitigation."
)

st.write("---")
st.write("Upload a structured transaction or beneficiary CSV file to generate risk scores and anomaly flags.")
st.write("**Supported formats:** Standard transaction CSV | SURE-P CCT Beneficiary Log")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def is_surep(content_bytes):
    """Check if file is SURE-P CCT format by scanning raw text."""
    try:
        text = content_bytes.decode('utf-8', errors='replace').upper()
    except Exception:
        return False
    return any(k in text for k in ['SURE-P', 'BENEFICIARY LOG', 'CO-RESPONSIBILITY', 'ANC 1', 'REG. CARD'])

def process_surep(content_bytes):
    """Parse SURE-P CCT log and return feature matrix + display df."""
    df_raw = pd.read_csv(io.BytesIO(content_bytes), header=None, skiprows=7, on_bad_lines='skip')
    # Keep only rows where first column is a number (actual data rows)
    df_raw = df_raw[df_raw[0].astype(str).str.strip().str.isdigit()].reset_index(drop=True)

    def yes_col(col_idx):
        if col_idx in df_raw.columns:
            return df_raw[col_idx].astype(str).str.strip().str.upper().eq('YES').astype(int)
        return pd.Series(0, index=df_raw.index)

    features = pd.DataFrame()
    features['ANC1']         = yes_col(12)
    features['ANC2']         = yes_col(14)
    features['ANC3']         = yes_col(16)
    features['ANC4']         = yes_col(18)
    features['SBD']          = yes_col(20)
    features['PNC']          = yes_col(22)
    features['IMM']          = yes_col(24)
    features['CoResp_Total'] = features[['ANC1','ANC2','ANC3','ANC4','SBD','PNC','IMM']].sum(axis=1)
    features['Has_Phone']    = df_raw[8].astype(str).str.strip().ne('').ne('nan').astype(int) if 8 in df_raw.columns else 0
    features['Age']          = pd.to_numeric(df_raw[6], errors='coerce').fillna(24) if 6 in df_raw.columns else 24
    features['No_Phone_With_Attendance'] = ((features['Has_Phone']==0) & (features['ANC2']==1)).astype(int)

    # Build display dataframe
    display = pd.DataFrame()
    if 4 in df_raw.columns: display['Surname']  = df_raw[4]
    if 5 in df_raw.columns: display['First Name'] = df_raw[5]
    if 6 in df_raw.columns: display['Age']      = df_raw[6]
    if 7 in df_raw.columns: display['Village']  = df_raw[7]
    if 3 in df_raw.columns: display['Card No']  = df_raw[3]
    if 28 in df_raw.columns: display['Notes']   = df_raw[28]
    display['CoResp Score'] = features['CoResp_Total']

    return features, display

def process_standard(content_bytes):
    """Parse standard numeric CSV."""
    df = pd.read_csv(io.BytesIO(content_bytes))
    if 'Class' in df.columns:
        labels = df['Class'].values
        features = df.drop(columns=['Class'])
    else:
        labels = None
        features = df.copy()
    features = features.select_dtypes(include=[np.number]).fillna(0)
    return features, features.copy(), labels

if uploaded_file is not None:

    # Read all bytes once
    content_bytes = uploaded_file.read()

    surep_format = is_surep(content_bytes)

    if surep_format:
        features, display_df = process_surep(content_bytes)
        labels = None
        data_type = 'surep'
        st.success("✅ SURE-P CCT Beneficiary Log detected — running CCT risk analysis")
    else:
        features, display_df, labels = process_standard(content_bytes)
        data_type = 'standard'

    if features.empty or len(features) == 0:
        st.error("Could not extract data from this file. Please check the format and try again.")
        st.stop()

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(scaled)
    anomaly_flag   = iso.predict(scaled)
    anomaly_scores = iso.decision_function(scaled)
    risk_score = (
        (1 - (anomaly_scores - anomaly_scores.min()) /
         (anomaly_scores.max() - anomaly_scores.min())) * 100
    ).round(2)
    anomaly_label = ["Anomaly" if x == -1 else "Normal" for x in anomaly_flag]

    # Results
    results = display_df.copy()
    results["Risk Score"]   = risk_score
    results["Anomaly Flag"] = anomaly_label
    if labels is not None:
        results["Actual Class"] = labels
    results = results.sort_values("Risk Score", ascending=False).reset_index(drop=True)

    # SURE-P context
    if data_type == 'surep':
        st.write("---")
        st.write("### 📋 SURE-P CCT Analysis Context")
        st.write(
            "This analysis applies Isolation Forest anomaly detection to beneficiary "
            "co-responsibility completion patterns. High risk scores indicate beneficiaries "
            "whose attendance patterns deviate significantly from the programme baseline — "
            "potential indicators of duplicate registration, payment irregularity, or data "
            "entry anomalies requiring field verification."
        )

    # Metrics
    st.write("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",      f"{len(results):,}")
    col2.metric("Anomalies Detected", f"{(results['Anomaly Flag']=='Anomaly').sum():,}")
    col3.metric("Avg Risk Score",     f"{results['Risk Score'].mean():.1f}")

    # Filter
    st.subheader("Filter High-Risk Cases")
    threshold = st.slider("Show cases with Risk Score above:", 0, 100, 70)
    filtered  = results[results["Risk Score"] >= threshold]
    st.write(f"**{len(filtered):,} records** above threshold")
    st.dataframe(filtered.head(50))

    # Distribution
    st.subheader("Risk Score Distribution")
    st.bar_chart(results["Risk Score"].value_counts().sort_index())

    # Model performance
    st.write("---")
    st.subheader("📊 Model Performance Analysis")

    if labels is not None:
        predicted = [1 if x == -1 else 0 for x in anomaly_flag]
        report    = classification_report(labels, predicted, target_names=["Normal","Fraud"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.write("#### Classification Report")
        st.dataframe(report_df)

        precision = report["Fraud"]["precision"]
        recall    = report["Fraud"]["recall"]
        f1        = report["Fraud"]["f1-score"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (Fraud)", f"{precision:.1%}", help="Of all flagged cases, how many were actual fraud")
        col2.metric("Recall (Fraud)",    f"{recall:.1%}",    help="Of all actual fraud cases, how many were detected")
        col3.metric("F1 Score (Fraud)",  f"{f1:.1%}",        help="Harmonic mean of precision and recall")

        st.write("#### Confusion Matrix")
        cm  = confusion_matrix(labels, predicted)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Predicted Normal","Predicted Fraud"])
        ax.set_yticklabels(["Actual Normal","Actual Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="black", fontsize=12)
        plt.title("Confusion Matrix"); plt.tight_layout()
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
            "No ground truth labels detected. To enable model performance metrics, "
            "include a column named 'Class' where 1 = fraud/anomaly and 0 = normal."
        )

    # Download
    st.write("---")
    csv_out = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download High-Risk Cases as CSV",
        data=csv_out,
        file_name="high_risk_cases.csv",
        mime="text/csv"
    )

else:
    st.write("---")
    st.write("### How it works")
    col1, col2, col3 = st.columns(3)
    col1.write("**1. Upload**\nUpload any transaction or beneficiary CSV file")
    col2.write("**2. Analyse**\nIsolation Forest detects anomalies in real time")
    col3.write("**3. Export**\nDownload flagged high-risk cases for investigation")

    st.write("---")
    st.write("### Applicable Contexts")
    col1, col2, col3 = st.columns(3)
    col1.write("**Public Finance**\nConditional cash transfer programmes, government disbursements")
    col2.write("**Fintech**\nMobile banking, payment gateways, digital wallets")
    col3.write("**Healthcare Finance**\nBeneficiary payment systems, procurement fraud detection")
