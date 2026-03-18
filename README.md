# AI Risk Intelligence Engine
AI-driven pre-transaction anomaly detection engine for financial risk modelling


An AI-powered pre-transaction anomaly detection system for financial 
risk modelling in public sector payment and disbursement systems.

## What It Does
- Ingests structured transaction data via CSV upload
- Generates fraud risk scores (0–100) using a trained classification model
- Flags behavioural anomalies using the Isolation Forest algorithm
- Filters and displays high-risk cases above a user-defined threshold
- Exports flagged results as a downloadable CSV

## Background
This engine was developed from direct experience designing and deploying 
The Beneficiary Data and Payment Management System for Nigeria's SURE-P 
Maternal and Child Health Conditional Cash Transfer programme — a 
World Bank and Bill & Melinda Gates Foundation-funded initiative covering 
36 states and over 1,300 Primary Healthcare Centres.

## Built With
- Python
- Streamlit
- Scikit-learn (Isolation Forest)
- Pandas
- Joblib

## Author
Abimbola Eunice Ajayi — Independent Researcher, UK
