# AI Risk Intelligence Engine

An AI-powered pre-transaction anomaly detection system for financial risk 
modelling in public sector payment and disbursement systems.

🔗 **Live Demo:** https://financial-anomaly-detection-engine.streamlit.app

---

## What It Does

- Ingests structured transaction data via CSV upload
- Generates fraud risk scores (0–100) using unsupervised machine learning
- Flags behavioural anomalies using the Isolation Forest algorithm
- Filters and displays high-risk cases above a user-defined threshold
- Exports flagged results as a downloadable CSV

---

## Background

This engine was developed from direct experience designing and deploying 
The Beneficiary Data and Payment Management System for Nigeria's SURE-P 
Maternal and Child Health Conditional Cash Transfer Programme — a World 
Bank and Bill & Melinda Gates Foundation-funded initiative covering 36 
states and over 1,300 Primary Healthcare Centres.

Deploying that system at a national scale exposed structural vulnerabilities 
common across large government disbursement programmes: reactive error 
detection, manual reconciliation, post-transaction oversight, and limited 
behavioural anomaly detection. This engine was built to address those gaps 
by shifting financial governance from reactive audit to pre-transaction 
anomaly detection.

---

## How It Works

The system uses the **Isolation Forest** algorithm — an unsupervised 
machine learning approach that identifies anomalous transactions by 
measuring how quickly an observation can be isolated from the rest of the 
dataset. Anomalies require fewer splits to isolate, resulting in shorter 
path lengths and higher risk scores.

This approach is particularly effective in public finance environments 
where labelled fraud data is limited or unavailable.

---

## How to Use

1. Visit the live app at the link above
2. Upload any structured transaction CSV file
3. Review the risk scores, anomaly flags, and distribution chart
4. Adjust the risk threshold slider to filter high-risk cases
5. Download flagged cases as CSV for further investigation

---

## Built With

- Python
- Streamlit
- Scikit-learn (Isolation Forest)
- Pandas
- NumPy

---

## Author

**Abimbola Eunice Ajayi**  
Independent Researcher, UK  
AI Systems & Public Finance Technology
