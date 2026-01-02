# Customer-Churn-Prediction
## 📌 Project Overview

Customer churn refers to customers leaving a service or discontinuing a subscription. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project focuses on predicting customer churn using supervised machine learning and presenting insights through an interactive Streamlit dashboard. The aim is to help businesses understand churn behavior and identify key factors that influence customer retention.

## 🎯 Objectives

-Predict whether a customer is likely to churn

-Analyze patterns and factors contributing to churn

-Visualize insights in an interactive and easy-to-understand dashboard

-Build a project suitable for real-world business analysis and demonstrations

## 📂 Dataset

• Name: Telco Customer Churn Dataset

• Source: Kaggle

• Records: ~7,000 customers

• Target Variable: Churn (Yes / No)

## Dataset Features Include:

• Customer demographics (gender, senior citizen, dependents)

• Service usage details (internet service, streaming services, tech support)

• Contract and payment information

• Billing and tenure details

## 🧠 Machine Learning Approach

• Learning Type: Supervised Learning

• Problem Type: Binary Classification

• Algorithm Used: Logistic Regression

## Logistic Regression was chosen because:

• It is well-suited for binary classification problems

• It provides interpretable coefficients

• It helps understand which features influence churn


## 🔧 Project Workflow

1️⃣ Data Loading & Inspection

• Loaded dataset using Pandas

• Examined structure, data types, and missing values

2️⃣ Data Cleaning & Preprocessing

• Converted TotalCharges to numeric format

• Removed rows with invalid or missing values

• Dropped irrelevant identifier (customerID)

• Encoded the target variable (Churn)

• Applied One-Hot Encoding to categorical features

• Scaled numerical features using StandardScaler

3️⃣ Train–Test Split

• Split dataset into 80% training and 20% testing data

• Used stratified sampling to maintain churn ratio

4️⃣ Model Training

• Trained a Logistic Regression model on scaled training data

5️⃣ Model Evaluation

• Evaluated the model using:

    • Accuracy score

    • Confusion matrix

    • Classification report

    • ROC curve and AUC score

## 📈 Model Performance

• Accuracy: ~80%

• The model demonstrates reasonable predictive performance for churn detection

• Recall is emphasized to reduce missed churn cases, which is critical in business scenarios


## 📊 Streamlit Dashboard

An interactive dashboard was developed using Streamlit to visualize:

    • Dataset overview

    • Customer churn distribution

    • Churn vs customer tenure analysis

    • Logistic Regression model performance

    • Confusion matrix visualization

    • Feature importance analysis

The dashboard allows users to explore churn patterns without needing to interact with the code.


## 🚀 Deployment

This project is deployed as a locally runnable Streamlit dashboard.
The dashboard allows users to interactively explore customer churn analysis, model performance, and key insights through a browser interface.

Run the Application Locally:

git clone https://github.com/aayushrj-tech/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py


The application will open in the browser at:

• http://localhost:8501

## Deployment Notes

• The application runs completely on a local machine

• No external APIs or API keys are required

• Suitable for demonstrations, analysis, and learning purposes

## 🛠 Tools & Technologies

• Python

• Pandas & NumPy

• Matplotlib & Seaborn

• Scikit-learn

• Streamlit


## 👤 Author

Aayush Raj

B.Tech – Computer Science & Engineering (AI & ML)    
