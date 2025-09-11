import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")

st.write("This application predicts the likelihood of heart disease based on patient health data.")

# Collect user input
st.header("Patient Information")

age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert inputs to dataframe
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                   'restecg', 'thalach', 'exang', 'oldpeak', 
                                   'slope', 'ca', 'thal'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[1]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 The model predicts a HIGH likelihood of heart disease. (Risk: {probability:.2f})")
    else:
        st.success(f"✅ The model predicts a LOW likelihood of heart disease. (Risk: {probability:.2f})")

