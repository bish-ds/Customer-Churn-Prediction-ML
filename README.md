# Telco Customer Churn Prediction & Retention Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green.svg?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces)

An end-to-end Machine Learning project to predict telecommunication customer churn and recommend retention strategies. This project comprises a fully executed Jupyter notebook exploring the data science lifecycle and a modern, deployable **Streamlit web application** that can be hosted on **Hugging Face Spaces**.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Interactive Dashboard Features](#-interactive-dashboard-features)
- [Project Structure](#-project-structure)
- [Dataset Details](#-dataset-details)
- [Methodology & ML Lifecycle](#-methodology--ml-lifecycle)
- [Tech Stack](#-tech-stack)
- [Installation & Local Setup](#-installation--local-setup)
- [Hugging Face Spaces Deployment](#-hugging-face-spaces-deployment)
- [Business Insights & Recommendations](#-business-insights--recommendations)

---

## 📋 Project Overview

Customer churn is one of the most critical metrics for contract-based businesses. Acquiring new customers can cost up to 5x more than retaining existing ones. This project addresses the churn problem using the **Telco Customer Churn** dataset (7,043 customers). It showcases a comprehensive machine learning approach involving exploratory analysis, advanced resampling (SMOTE), cross-validation, hyperparameter tuning, model persistence, and deployment.

---

## 🖥️ Interactive Dashboard Features

The project includes `app.py`, a premium Streamlit web application styled with a glassmorphism dark mode:
1. **Dynamic Form Input**: Enter customer demographics, subscribed services, and billing contracts.
2. **Real-time Scoring**: Instantly calculates the customer's churn risk probability using the serialized XGBoost model.
3. **Risk Profile Classification**: Labels customers as **HIGH CHURN RISK** or **LOW CHURN RISK**.
4. **Actionable Recommendations**: Generates dynamic business tactics tailored specifically to the customer's profile (e.g., contract upgrade offers, payment method switch discounts).

---

## 📂 Project Structure

```
Customer_Churn_Prediction_using_ML/
├── Customer_Churn_Prediction_using_ML.ipynb   # Main notebook with analysis and outputs
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Churn dataset (7043 rows, 21 columns)
├── app.py                                      # Streamlit application dashboard
├── customer_churn_model.pkl                    # Pickled final trained XGBoost model
├── encoders.pkl                                # Pickled LabelEncoders for text fields
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Git ignore rules
└── README.md                                   # Project documentation
```

---

## 📊 Dataset Details

- **Source**: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features**: 21 columns including customer demographics (gender, partner), account settings (tenure, contract, billing), and subscribed services (internet, security, tech support).
- **Target Variable**: `Churn` (Yes / No) — highly imbalanced (73.5% No Churn vs. 26.5% Churn).

---

## 🔬 Methodology & ML Lifecycle

The development workflow in the notebook is organized as follows:

1. **Data Cleaning**: Handled type conversions for `TotalCharges` (converted empty strings to 0 and recast to float).
2. **Exploratory Data Analysis (EDA)**: Conducted univariate and bivariate visual analysis with boxplots, histograms, and heatmaps to discover churn drivers.
3. **Feature Engineering & Preprocessing**:
   - Encoded categorical variables using `LabelEncoder`.
   - Performed Stratified Train-Test Split (80/20) to preserve label distributions.
   - Applied **Synthetic Minority Over-sampling Technique (SMOTE)** to handle target class imbalance.
4. **Model Comparison & Evaluation**:
   - Compared Logistic Regression, Decision Tree, Random Forest, and XGBoost.
   - Used **Stratified 5-Fold Cross-Validation** to assess model generalization.
   - Evaluated models using F1 Score (primary metric for class imbalance) and ROC-AUC.
5. **Hyperparameter Tuning**:
   - Executed `GridSearchCV` on Random Forest.
   - Executed `RandomizedSearchCV` (50 iterations) on XGBoost to optimize performance.
6. **Final Selection**: Exported the best XGBoost estimator along with encoders into pickle files for deployment.

---

## 🛠️ Tech Stack

- **Data Processing**: `Pandas`, `NumPy`
- **Visualization**: `Matplotlib`, `Seaborn`
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `Imbalanced-Learn` (SMOTE)
- **Deployment & App**: `Streamlit`
- **Model Serialization**: `Pickle`

---

## 🚀 Installation & Local Setup

To run this project on your machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Customer-Churn-Prediction-ML.git
   cd Customer-Churn-Prediction-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   The app will automatically launch in your default web browser at `http://localhost:8501`.


## 📈 Business Insights & Recommendations

The model identified several top predictors of customer churn, which translate to actionable retention strategies:

1. **Contract Type**: Month-to-month contracts are the strongest predictor of churn.
   - *Recommendation*: Offer promotional incentives (e.g., 10% discount) to upgrade month-to-month customers to annual or two-year contracts.
2. **Tenure**: Customers with shorter tenure (< 12 months) are highly prone to churn.
   - *Recommendation*: Implement focused customer onboarding and check-in programs during the first year of service.
3. **Add-on Services**: The absence of `OnlineSecurity` and `TechSupport` services correlates with higher churn.
   - *Recommendation*: Bundle security and support services at reduced prices to drive adoption.
4. **Payment Method**: Customers paying via `Electronic Check` show significantly higher churn.
   - *Recommendation*: Encourage enrollment in auto-pay (Credit Card or Bank Transfer) by offering a one-time bill credit.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
