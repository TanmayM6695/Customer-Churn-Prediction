import streamlit as st
import joblib
import numpy as np
import os

# Load the scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")


# App title
st.title("🔍 Customer Churn Prediction App")

st.markdown("Provide customer details to predict whether they are likely to churn.")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)

# Convert categorical to numeric
gender_binary = 1 if gender == "Female" else 0

# Prepare and scale input
input_data = np.array([[age, gender_binary, tenure, monthly_charges]])
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("🔮 Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    churn_proba = model.predict_proba(scaled_input)[0][1]  # Probability of class 1 (churn)

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

    st.markdown(f"**Churn Probability:** `{churn_proba:.2%}`")
