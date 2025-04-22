# Customer Churn Prediction.py
import pickle

import pandas as pd
import streamlit as st

# Load model, scaler, and feature names
model = pickle.load(open('churn_model (1).pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

st.title("Telco Customer Churn Prediction")
st.write("Fill out the customer details below to predict if they are likely to churn.")

# Input fields
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    submit = st.form_submit_button("Predict")

if submit:
    # Raw input dict
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert TotalCharges to numeric
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

    # One-hot encode with drop_first=True
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training set
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]

    # Output
    st.markdown("---")
    if pred == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")
