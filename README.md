# Customer Churn Prediction – ChurnGuard AI

This project focuses on predicting telecom customer churn using machine learning. The objective is to identify customers who are likely to leave the service so that businesses can take preventive action and improve retention.

The project covers the complete ML workflow including data preprocessing, model training, evaluation, explainability, dashboard development, and cloud deployment.

## 🚀 Live Application

Streamlit Web App:  
https://customerchurnprediction-ohctytlym3cst3blssgwxf.streamlit.app

## 📊 Project Overview

Customer churn is a major challenge in the telecom industry. In this project:

- Customer data was cleaned and preprocessed  
- Class imbalance was handled using SMOTE  
- Multiple models were trained and compared  
- Random Forest was selected as the best-performing model  
- SHAP was used for model explainability  
- The solution was deployed as a live web application

- 📁 Dataset

This project uses the IBM Telco Customer Churn Dataset, which contains customer account information, service usage details, and churn labels. The dataset is publicly available and widely used for churn prediction modeling. 

## 🧠 Machine Learning Workflow

1. Data Cleaning and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Handling Imbalanced Data (SMOTE)  
5. Model Training and Evaluation  
6. Model Selection (Random Forest)  
7. Explainability using SHAP  
8. Streamlit Integration  
9. Cloud Deployment  

## 📈 Model Performance

- Best Model: Random Forest  
- ROC-AUC Score: 0.8338  
- F1 Score: 0.6225  
- Dataset balanced using SMOTE  

## 🖥 Web Application Features

- Individual customer churn prediction  
- Probability score display  
- Interactive risk visualization  
- Batch prediction using CSV upload  
- Model performance and information section  
- Clean and responsive UI  

## 🛠 Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Imbalanced-learn (SMOTE)  
- Plotly  
- Streamlit  
- Power BI  
- GitHub  
- Streamlit Community Cloud  

## 📂 Project Structure
Customer_Churn_Prediction/
├── app.py
├── requirements.txt
├── models/
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ └── feature_cols.json
├── Churn_Predictions_PowerBI.csv
├── Customer_Churn_Prediction.ipynb
├── Customer_Churn_Prediction.pbix
└── Customer_Churn_Project_Report.pdf


## 🎯 Business Relevance

This system helps telecom companies:

- Identify high-risk customers early  
- Reduce churn rate  
- Improve retention strategies  
- Support data-driven decision making  

## 👤 Author

J. Charan Reddy  
B.Tech – Data Analytics & Machine Learning  
GitHub: https://github.com/okmjunhb-maker
