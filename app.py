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
    "This engine demonstrates the methodology applicable to public sector payment and "
    "disbursement systems — including conditional cash transfer programmes, government "
    "procurement systems, and fintech payment gateways. It uses the Isolation Forest "
    "algorithm to detect behavioural anomalies prior to transaction execution, shifting "
    "financial governance from reactive audit to proactive risk mitigation."
)

st.write("---")
st.write("Upload a structured transaction or beneficiary CSV file to generate risk scores and anomaly flags.")
st.write("**Supported formats:** Standard transaction CSV (numeric) | SURE-P CCT Beneficiary Log")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def detect_surep_format(df_raw):
    """Detect if this is a SURE-P CCT beneficiary log."""
    text = df_raw.to_string().upper()
    return 'SURE-P' in text or 'ANC' in text or 'BENEFICIARY LOG' in text or 'CO-RESPONSIBILITY' in text

def process_surep(df):
    """Convert SURE-P CCT log into numeric risk features."""
    df = df.copy()
    df.columns = range(len(df.columns))

    # Map columns based on known structure
    col_names = {
        0: 'SN', 2: 'DATE_REG', 3: 'CARD_NO', 4: 'SURNAME', 5: 'FIRST_NAME',
        6: 'AGE', 7: 'VILLAGE', 8: 'PHONE', 9: 'PHONE_OWNER', 10: 'PAYMENT_METHOD',
        12: 'ANC1', 13: 'DATE_ANC1', 14: 'ANC2', 15: 'DATE_ANC2',
        16: 'ANC3', 17: 'DATE_ANC3', 18: 'ANC4', 19: 'DATE_ANC4',
        20: 'SBD', 21: 'DATE_SBD', 22: 'PNC', 23: 'DATE_PNC',
        24: 'IMM', 25: 'DATE_IMM', 28: 'NOTES'
    }
    df = df.rename(columns={k: v for k, v in col_names.items() if k in df.columns})

    def yes_to_int(col):
        if col in df.columns:
            return df[col].astype(str).str.strip().str.upper().eq('YES').astype(int)
        return pd.Series(0, index=df.index)

    # Build numeric feature matrix
    features = pd.DataFrame()
    features['ANC1_Complete']     = yes_to_int('ANC1')
    features['ANC2_Complete']     = yes_to_int('ANC2')
    features['ANC3_Complete']     = yes_to_int('ANC3')
    features['ANC4_Complete']     = yes_to_int('ANC4')
    features['SBD_Complete']      = yes_to_int('SBD')
    features['PNC_Complete']      = yes_to_int('PNC')
    features['IMM_Complete']      = yes_to_int('IMM')
    features['CoResp_Score']      = (
        features['ANC1_Complete'] + features['ANC2_Complete'] +
        features['ANC3_Complete'] + features['ANC4_Complete'] +
        features['SBD_Complete']  + features['PNC_Complete']  +
        features['IMM_Complete']
    )
    features['Has_Phone']         = df['PHONE'].astype(str).str.strip().ne('').ne('nan').astype(int) if 'PHONE' in df.columns else 0
    features['Age']               = pd.to_numeric(df['AGE'], errors='coerce').fillna(features['Age'].mean() if 'Age' in features else 24) if 'AGE' in df.columns else 24

    # Anomaly indicators
    features['Missing_Phone_With_Attendance'] = (
        (features['Has_Phone'] == 0) &
        (features['ANC2_Complete'] == 1)
    ).astype(int)

    features['High_Attendance_No_Phone'] = (
        (features['CoResp_Score'] >= 3) &
        (features['Has_Phone'] == 0)
    ).astype(int)

    # Display dataframe
    display_cols = {}
    for c in ['SURNAME', 'FIRST_NAME', 'AGE', 'VILLAGE', 'PAYMENT_METHOD', 'NOTES']:
        if c in df.columns:
            display_cols[c] = df[c]
    display_df = pd.DataFrame(display_cols)

    return features, display_df

if uploaded_file is not None:

    # Try reading as standard CSV first
    try:
        raw = pd.read_csv(uploaded_file)
        is_surep = detect_surep_format(raw)
    except Exception:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, header=None)
        is_surep = True

    if is_surep:
        # Re-read with SURE-P format
        uploaded_file.seek(0)
        raw_full = pd.read_csv(uploaded_file, header=None)
        is_surep_confirm = detect_surep_format(raw_full)

        if is_surep_confirm:
            # Skip header rows and read data
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, header=None, skiprows=7)
            df_raw = df_raw[df_raw[0].astype(str).str.strip().str.isdigit()].reset_index(drop=True)

            st.success("✅ SURE-P CCT Beneficiary Log detected — running CCT risk analysis")

            features, display_df = process_surep(df_raw)
            labels = None
            data_type = 'surep'
        else:
            features = raw.select_dtypes(include=[np.number]).fillna(0)
            labels = raw['Class'].values if 'Class' in raw.columns else None
            if 'Class' in raw.columns:
                features = raw.drop(columns=['Class'])
            display_df = features.copy()
            data_type = 'standard'
    else:
        if 'Class' in raw.columns:
            labels = raw['Class'].values
            features = raw.drop(columns=['Class'])
        else:
            labels = None
            features = raw.copy()
        features = features.select_dtypes(include=[np.number]).fillna(0)
        display_df = features.copy()
        data_type = 'standard'

    # Scale and run Isolation Forest
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(scaled)
    anomaly_flag = iso.predict(scaled)
    anomaly_scores = iso.decision_function(scaled)
    risk_score = (
        (1 - (anomaly_scores - anomaly_scores.min()) /
         (anomaly_scores.max() - anomaly_scores.min())) * 100
    ).round(2)
    anomaly_label = ["Anomaly" if x == -1 else "Normal" for x in anomaly_flag]

    # Build results
    results = display_df.copy()
    results["Risk Score"] = risk_score
    results["Anomaly Flag"] = anomaly_label
    if labels is not None:
        results["Actual Class"] = labels
    results = results.sort_values("Risk Score", ascending=False).reset_index(drop=True)

    # SURE-P context note
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
    col1.metric("Total Records", f"{len(results):,}")
    col2.metric("Anomalies Detected", f"{(results['Anomaly Flag'] == 'Anomaly').sum():,}")
    col3.metric("Avg Risk Score", f"{results['Risk Score'].mean():.1f}")

    # Filter
    st.subheader("Filter High-Risk Cases")
    threshold = st.slider("Show cases with Risk Score above:", 0, 100, 70)
    filtered = results[results["Risk Score"] >= threshold]
    st.write(f"**{len(filtered):,} records** above threshold")
    st.dataframe(filtered.head(50))

    # Risk distribution
    st.subheader("Risk Score Distribution")
    st.bar_chart(results["Risk Score"].value_counts().sort_index())

    # Model performance (only for labelled data)
    st.write("---")
    st.subheader("📊 Model Performance Analysis")

    if labels is not None:
        predicted = [1 if x == -1 else 0 for x in anomaly_flag]
        report = classification_report(labels, predicted, target_names=["Normal","Fraud"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.write("#### Classification Report")
        st.dataframe(report_df)

        precision = report["Fraud"]["precision"]
        recall = report["Fraud"]["recall"]
        f1 = report["Fraud"]["f1-score"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (Fraud)", f"{precision:.1%}", help="Of all flagged cases, how many were actual fraud")
        col2.metric("Recall (Fraud)", f"{recall:.1%}", help="Of all actual fraud cases, how many were detected")
        col3.metric("F1 Score (Fraud)", f"{f1:.1%}", help="Harmonic mean of precision and recall")

        st.write("#### Confusion Matrix")
        cm = confusion_matrix(labels, predicted)
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
    col1.write("**1. Upload**\nUpload any transaction or beneficiary CSV file")
    col2.write("**2. Analyse**\nIsolation Forest detects anomalies in real time")
    col3.write("**3. Export**\nDownload flagged high-risk cases for investigation")

    st.write("---")
    st.write("### Applicable Contexts")
    col1, col2, col3 = st.columns(3)
    col1.write("**Public Finance**\nConditional cash transfer programmes, government disbursements")
    col2.write("**Fintech**\nMobile banking, payment gateways, digital wallets")
    col3.write("**Healthcare Finance**\nBeneficiary payment systems, procurement fraud detection")
