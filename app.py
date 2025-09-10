# streamlit_app.py (save this as app.py)
# Streamlit app for Heart Disease ML pipeline (Cleveland dataset)
# Features: upload dataset, EDA, preprocessing, train models, evaluate, predict new patient.
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# ---------------------
# Sidebar
# ---------------------
st.sidebar.title("Heart Disease ML App")
st.sidebar.info("Upload dataset, preprocess, train, evaluate and predict heart disease.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------------
# Load dataset
# ---------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    st.warning("No dataset uploaded. Using sample toy dataset...")
    df = pd.DataFrame({
        "age": [63, 67, 67, 37, 41],
        "sex": [1, 1, 1, 1, 0],
        "cp": [1, 4, 4, 3, 2],
        "trestbps": [145, 160, 120, 130, 130],
        "chol": [233, 286, 229, 250, 204],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 0, 2],
        "thalach": [150, 108, 129, 187, 172],
        "exang": [0, 1, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
        "slope": [3, 2, 2, 3, 1],
        "ca": [0, 3, 2, 0, 0],
        "thal": [6, 3, 7, 3, 3],
        "condition": [1, 1, 1, 0, 0]
    })

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------
# Exploratory Data Analysis
# ---------------------
if st.checkbox("Show EDA and dataset preview"):
    st.write("Dataset info:")
    st.write(df.describe())
    st.write("Missing values:", df.isnull().sum().sum())
    st.write("Duplicates:", df.duplicated().sum())

    fig, ax = plt.subplots()
    sns.countplot(x="condition", data=df, ax=ax)
    st.pyplot(fig)

# ---------------------
# Preprocessing
# ---------------------
if st.button("Preprocess & Split (80/20)"):
    df = df.drop_duplicates()

    categorical_cols = ["cp", "restecg", "slope", "thal"]
    encoders = {}
    for col in categorical_cols:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df.drop("condition", axis=1)
    y = df["condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.session_state.X_train = X_train_scaled
    st.session_state.X_test = X_test_scaled
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler
    st.session_state.encoders = encoders

    st.success("Preprocessing done!")

# ---------------------
# Train Models
# ---------------------
if st.button("Train models"):
    if "X_train" not in st.session_state:
        st.error("Preprocess data first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(random_state=42)
        svm = SVC(probability=True, random_state=42)
        xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        xg.fit(X_train, y_train)

        st.session_state.models = {
            "Logistic Regression": lr,
            "Random Forest": rf,
            "SVM": svm,
            "XGBoost": xg
        }

        st.success("Models trained successfully!")

# ---------------------
# Evaluate Models
# ---------------------
if st.button("Evaluate models"):
    if "models" not in st.session_state:
        st.error("Train models first.")
    else:
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        models = st.session_state.models

        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            results[name] = {"Accuracy": acc, "F1": f1, "ROC-AUC": auc}

            st.write(f"### {name}")
            st.text(classification_report(y_test, y_pred))

            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(results).T)

# ---------------------
# Predict New Patient
# ---------------------
st.subheader("Predict on New Patient")
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=20, max_value=100, value=55)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=140)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)", [1, 0])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=220, value=160)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.2)
    slope = st.selectbox("Slope", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [3, 6, 7])

    submitted = st.form_submit_button("Predict")

if submitted:
    if "models" not in st.session_state:
        st.error("Please train models first.")
    else:
        new_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal
        ]], columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])

        scaler = st.session_state.scaler
        new_scaled = scaler.transform(new_data)

        st.write("### Predictions:")
        models = st.session_state.models
        for name, model in models.items():
            pred = model.predict(new_scaled)[0]
            proba = model.predict_proba(new_scaled)[0][1]
            st.write(f"{name}: Predicted = {pred}, Probability = {proba:.2f}")
