# Heart_Disease_Prediction
# About the Project
This project applies machine learning to detect heart disease early, aiming to improve patient outcomes. Using the UCI Heart Disease dataset, the model analyzes patient data to classify the presence of heart disease. The workflow includes data cleaning, preprocessing, feature selection, and model development. By leveraging predictive analytics, this project demonstrates how data-driven insights can support clinical decision-making and enhance diagnostic accuracy.

# Problem Statement
Cardiovascular disease is a leading cause of death globally, yet traditional diagnostic methods are often invasive, costly, and limited in capturing complex patient data. This project leverages machine learning to predict heart disease early using the Cleveland Heart Disease dataset. Data preprocessing included cleaning duplicates, encoding categorical features, and scaling numerical values. Models such as Logistic Regression, Random Forest, SVM, and XGBoost were applied, with evaluation using accuracy, F1-score, ROC-AUC, and confusion matrices. The approach demonstrates how predictive analytics can identify complex patterns, reduce reliance on invasive testing, and support clinicians in timely, data-driven decision-making, ultimately improving patient outcomes and resource allocation.

# Dataset Source
This project uses the Cleveland Heart Disease dataset from UCI, publicly available on Kaggle. The dataset contains real clinical and psychological patient data, including features such as cholesterol, resting blood pressure, chest pain type, maximum heart rate, and thalassemia key status indicators of heart disease risk. Its structured mix of numerical and categorical variables makes it suitable for machine learning, allowing models to explore complex relationships between risk factors. The dataset’s size supports efficient experimentation while enabling reproducible research and benchmarking against prior studies, making it an ideal choice for predictive modeling of heart disease.

# Methodology
EDA on the Cleveland Heart Disease dataset (297 records, 14 features) revealed patterns in numerical (age, cholesterol, blood pressure) and categorical (chest pain, sex, thalassemia) variables, guiding preprocessing and feature selection. Models implemented include Logistic Regression, Random Forest, SVM, and XGBoost. Evaluation using accuracy, recall, F1-score, and ROC-AUC showed Random Forest and XGBoost performed best, supporting early and accurate heart disease detection.

# Model Development
https://colab.research.google.com/drive/1Wre-yxgQkYg8Pyu8w6tyHhDkzPmY5coK?usp=sharing (Google Colab Code)

# Model Evaluation
Logistic Regression Results:
Accuracy: 0.9166666666666666
F1 Score: 0.9019607843137255
ROC-AUC: 0.953125

Random Forest Results:
Accuracy: 0.8833333333333333
F1 Score: 0.8627450980392157
ROC-AUC: 0.9447544642857144

SVM Results:
Accuracy: 0.9
F1 Score: 0.88
ROC-AUC: 0.9397321428571428

XGBoost Results:
Accuracy: 0.85
F1 Score: 0.8301886792452831
ROC-AUC: 0.9441964285714286

# Demonstration
https://heartdiseaseapp-tqqzet2dp7aarmhpzr6wny.streamlit.app/ (App)
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/data?select=heart_cleveland_upload.csv (Model Code Dataset) 
File Name: heart_cleveland_upload.csv

# Acknowledgement
Dataset: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/data?select=heart_cleveland_upload.csv

Team Members:
1. Kristy Jade Luther 202307010100 BCSSE
2. Muhammad Aiman Hakimi Bin Khairul Hafti 202309010149 BCSSE
3. Venush A/L Kumar 202409010618 BCSSE
4. Natasha Najwa Abdullah 202409010558 BCSSE
5. Muhammad Alham Bin Jamain 202309010265 BCSSE

Lecturer: SIR NAZMIRUL IZZAD BIN NASSIR
Subject Code: BIT4333
Subject Name: INTRODUCTION TO MACHINE LEARNING
Submission: 11th SEPTEMBER 2025
