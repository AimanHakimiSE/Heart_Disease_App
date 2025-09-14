# Heart Disease Prediction: A Machine Learning Approach 
# About the Project
<div align="justify">
This project extends my degree coursework by developing and evaluating machine learning models to predict heart disease using the Cleveland Heart Disease dataset from the UCI repository. It emphasizes the technical aspects of model selection, training, and performance evaluation, highlighting the effectiveness of algorithms such as Logistic Regression, Support Vector Machine, Random Forest, and XGBoost in clinical prediction tasks.
</div>

# Background Overview
<div align="justify">
Cardiovascular disease is a leading cause of death globally, yet traditional diagnostic methods are often invasive, costly, and limited in capturing complex patient data. This project leverages machine learning to predict heart disease early using the Cleveland Heart Disease dataset. Data preprocessing included cleaning duplicates, encoding categorical features, and scaling numerical values. Models such as Logistic Regression, Random Forest, SVM, and XGBoost were applied, with evaluation using accuracy, F1-score, ROC-AUC, and confusion matrices. The approach demonstrates how predictive analytics can identify complex patterns, reduce reliance on invasive testing, and support clinicians in timely, data-driven decision-making, ultimately improving patient outcomes and resource allocation.
</div>

# Problem Statement
<div align="justify">
Despite medical advances, the timely and accurate diagnosis of cardiovascular disease remains a challenge. Traditional methods rely on costly, invasive tests and fragmented health records that hinder holistic analysis. Conventional approaches also depend on rigid thresholds, overlooking subtle risk patterns and patient variability. These gaps highlight the need for advanced, data-driven solutions capable of handling multidimensional clinical data and supporting more reliable cardiovascular risk prediction.
</div>

# Objective
This project aims to develop and evaluate predictive models using machine learning techniques on the Cleveland Heart Disease dataset to improve the early detection of cardiovascular risk. The objective encompasses:
* Preparing and preprocessing clinical data for quality and consistency.
* Implementing models such as Logistic Regression, Random Forest, SVM, and XGBoost.
* Evaluating performance using accuracy, F1-score, ROC-AUC, and confusion matrices.
* Enhancing generalizability through hyperparameter tuning and cross-validation.

By leveraging clinical data, this project seeks to demonstrate the potential of machine learning as a decision-support tool, complementing traditional diagnostics and advancing data-driven healthcare.


# Built With
Google Collab

# Dataset Source
<div align="justify">
This project uses the Cleveland Heart Disease dataset from UCI, publicly available on Kaggle. The dataset contains real clinical and psychological patient data, including features such as cholesterol, resting blood pressure, chest pain type, maximum heart rate, and thalassemia key status indicators of heart disease risk. Its structured mix of numerical and categorical variables makes it suitable for machine learning, allowing models to explore complex relationships between risk factors. The dataset’s size supports efficient experimentation while enabling reproducible research and benchmarking against prior studies, making it an ideal choice for predictive modelling of heart disease.
</div>

# Methodology
* Data Collection: The Cleveland Heart Disease dataset from the UCI repository was used, consisting of 303 patient records with 14 clinical attributes. After removing duplicates, 297 unique records were retained for analysis.
* Preprocessing: Exploratory data analysis (EDA) was conducted to examine distributions, detect anomalies, and study relationships between features. Categorical variables were encoded, numerical features were scaled, and the data was split into training and testing sets. Hyperparameter tuning and cross-validation were applied to improve generalizability.
* Model Development: Multiple supervised machine learning algorithms were implemented, including Logistic Regression, Support Vector Machine (SVM), Random Forest, and XGBoost, selected for their balance between interpretability and predictive performance.
* Model Evaluation: Performance was assessed using accuracy, precision, recall, F1-score, and ROC-AUC, with special emphasis on recall and ROC-AUC due to the clinical importance of minimizing false negatives.
* Deployment: The best-performing model was deployed through a Streamlit web application, allowing users to input patient details and receive real-time predictions with probability scores to support clinical decision-making.

# Result and Impact
<div align="justify>
The evaluation showed clear differences in performance among models. Logistic Regression worked as a baseline but struggled with non-linear data, while SVM improved with scaling yet underperformed compared to ensembles. Random Forest and XGBoost achieved the best results, with XGBoost leading across accuracy, F1 score, and ROC-AUC. Both minimized false negatives, highlighting clinical reliability. Feature importance confirmed known risk factors such as chest pain type, thalassemia, and age. These findings emphasize machine learning’s role in supporting proactive care, reducing hospital admissions, improving patient outcomes, and shifting healthcare toward prevention rather than treatment.
  </div>

# Challenges and Solution
* Class Imbalance in Dataset: Addressed using stratified train–test split and evaluation metrics like F1 score and ROC AUC instead of accuracy, ensuring balanced detection of both classes.
* Categorical Variables and Scaling: Handled with label and one-hot encoding to preserve clinical meaning, while StandardScaler normalized numerical features for fair model training.
* Overfitting Risk: Mitigated through cross-validation and hyperparameter tuning, allowing ensemble models such as Random Forest and XGBoost to generalize effectively.
* Model Interpretability: Enhanced by feature importance analysis, highlighting medically relevant predictors and bridging the gap between accuracy and clinical trust.

# Model Development
https://colab.research.google.com/drive/1Wre-yxgQkYg8Pyu8w6tyHhDkzPmY5coK?usp=sharing (Google Colab Code)

# How to Use
How to Use
To test the application, click this link:
https://heartdiseaseapp-tqqzet2dp7aarmhpzr6wny.streamlit.app/
(Video of the App Tutorial in Notebook)

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

# Acknowledgement
<div align="justify"
We would like to express our gratitude to our lecturer, Sir Nazmirul Izzad Bin Nassir, for his invaluable guidance and support throughout this project. We also extend our appreciation to our team members for their collaboration and dedication in completing this work. The dataset used in this study was obtained from Kaggle, which served as the foundation for our analysis and experimentation.
  </div>

Team Members
* Kristy Jade Luther (202307010100, BCSSE)
* Muhammad Aiman Hakimi Bin Khairul Hafti (202309010149, BCSSE)
* Venush A/L Kumar (202409010618, BCSSE)
* Natasha Najwa Abdullah (202409010558, BCSSE)
* Muhammad Alham Bin Jamain (202309010265, BCSSE)

Course Information
* Subject Code: BIT4333
* Subject Name: Introduction to Machine Learning
* Submission Date: 11th September 2025

# References
* Bryan Chulde-Fernández, D., et al. (2025, March 19). Classification of Heart Failure Using Machine Learning: A Comparative Study. Retrieved from MDPI: https://www.mdpi.com/2075-1729/15/3/496
* Mattingly, Q. (2025). Cardiovascular Diseases. Retrieved from World Health Organization: https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1
* Prof Peder L. Myhre, MD, PhD, & J. T. (2024, October). Digital tools in heart failure: addressing unmet needs. Retrieved from Science Direct: https://www.sciencedirect.com/science/article/pii/S2589750024001584
* Siming Wan, F. W.-j. (2025, May 29). Machine learning approaches for cardiovascular disease prediction: A review. Retrieved from Science Direct: https://www.sciencedirect.com/science/article/abs/pii/S1875213625003201
* World Health Organization. (2025, July 31). Cardiovascular diseases (CVDs). Retrieved from World Health Organization: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)






