# CrediPredict – Loan Approval Prediction System

**CrediPredict** is an end-to-end Data Science and Machine Learning project designed to predict whether a credit application is likely to be **Approved (`Y`)** or **Rejected (`N`)** based on applicant financial and demographic details. 

This repository is built as a **B.Tech Data Science placement project**, focusing on clean data engineering, thorough exploratory data analysis (EDA), explainable machine learning modeling, an interactive **Streamlit** user interface, and Power BI analytics.

---

## 📌 Problem Statement

Automating loan approval workflows is critical for financial institutions to reduce risk and process applications efficiently. Manual evaluation can be slow, inconsistent, or prone to subjective bias. By building an explainable binary classification model, lenders can obtain instant risk assessments based on historical applicant data (income, credit score history, education, loan amount, and property location).

---

## 🎯 Objectives

1. Establish a structured Data Science pipeline from raw SQL/CSV storage to machine learning model deployment.
2. Handle missing values and extreme financial outliers using domain-specific heuristics.
3. Engineer meaningful household financial features (`Total_Income`, `Loan_to_Income_Ratio`).
4. Train and compare **Logistic Regression**, **Decision Tree**, and **Random Forest** algorithms using **5-Fold Stratified Cross-Validation**.
5. Hyperparameter tune candidate models via **GridSearchCV** to optimize the F1 Score and ROC-AUC metrics.
6. Serve real-time predictions via an intuitive Python-based **Streamlit Application**.
7. Provide dataset insights via **Power BI** dashboards and structured **SQL queries**.

---

## 🛠️ Project Architecture & Data Flow

```text
MySQL / CSV Data Source
       │
       ▼
Python / Pandas Data Loader (src/data_loader.py)
       │
       ▼
Data Cleaning & Missing Value Imputation (02_data_cleaning.ipynb & src/preprocessing.py)
       │
       ▼
Exploratory Data Analysis & Statistics (03_eda.ipynb)
       │
       ▼
Feature Engineering & Preprocessing (04_preprocessing.ipynb & src/preprocessing.py)
       │
       ▼
Train/Test Split (Stratified 80/20)
       │
       ▼
Machine Learning Training (Logistic Regression, Decision Tree, Random Forest)
       │
       ▼
5-Fold Stratified Cross Validation & GridSearchCV Tuning (src/train.py)
       │
       ▼
Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC) -> reports/model_comparison.csv
       │
       ▼
Save Best Model Pipeline (.pkl) -> models/loan_approval_model.pkl
       │
       ▼
Streamlit Application (app.py & src/predict.py)
       │
       ▼
Loan Prediction & Analytics UI
```

---

## 📂 Project Structure

```text
CrediPredict/
├── data/
│   ├── raw/
│   │   └── loan_data.csv                  # Raw dataset (614 rows, 13 features)
│   └── processed/
│       ├── cleaned_loan_data.csv          # Cleaned dataset without nulls
│       └── processed_loan_data.csv        # Processed dataset with engineered features
├── notebooks/
│   ├── 01_data_loading.ipynb              # Loading and initial data inspection
│   ├── 02_data_cleaning.ipynb             # Null imputation & outlier auditing
│   ├── 03_eda.ipynb                       # Seaborn/Matplotlib visual charts
│   ├── 04_preprocessing.ipynb             # Statistics & feature engineering
│   ├── 05_model_training.ipynb            # ML training & GridSearchCV tuning
│   └── 06_model_evaluation.ipynb          # Metrics reporting & model comparison
├── src/
│   ├── data_loader.py                     # Data loading module (MySQL + CSV fallback & logging)
│   ├── preprocessing.py                  # Cleaning, Feature Engineering & ColumnTransformer
│   ├── train.py                           # Training & model selection pipeline
│   └── predict.py                         # Inference engine for trained model
├── models/
│   └── loan_approval_model.pkl            # Saved Joblib Pipeline (Preprocessor + Best Model)
├── sql/
│   ├── database.sql                       # Database creation script
│   ├── tables.sql                         # Table schema DDL script (applicants & prediction_history)
│   └── queries.sql                        # 11 Analytical SQL queries
├── powerbi/
│   └── loan_approval_dashboard_data.csv   # Dataset for Power BI reporting
├── reports/
│   └── model_comparison.csv               # Model performance metrics summary
├── app.py                                 # Main Streamlit web application
├── requirements.txt                       # Project python dependencies
├── README.md                              # Documentation
└── .gitignore                             # Git ignore configuration
```

---

## 📊 Dataset Overview

The dataset contains historical loan application records with 13 features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Loan_ID` | Categorical | Unique loan application identifier |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Applicant marital status (Yes / No) |
| `Dependents` | Categorical | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Self-employed status (Yes / No) |
| `ApplicantIncome` | Numerical | Primary applicant monthly income ($) |
| `CoapplicantIncome`| Numerical | Co-applicant monthly income ($) |
| `LoanAmount` | Numerical | Requested loan amount ($ in Thousands) |
| `Loan_Amount_Term` | Numerical | Term of loan in months (120, 180, 240, 360, etc.) |
| `Credit_History` | Numerical | Credit history meets guidelines (1.0 = Good, 0.0 = Poor) |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | Target | Binary target: `Y` (Approved) or `N` (Rejected) |

---

## 🧹 Data Cleaning & Preprocessing

1. **Missing Value Imputation**:
   - **Categorical Columns** (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`): Imputed using **Mode** (most frequent class).
   - **Numerical Columns** (`LoanAmount`, `Loan_Amount_Term`): Imputed using **Median** (robust against right-skewed extreme values).
2. **Feature Engineering**:
   - `Total_Income = ApplicantIncome + CoapplicantIncome`: Captures combined household purchasing and repayment power.
   - `Loan_to_Income_Ratio = (LoanAmount * 1000) / Total_Income`: Quantifies debt burden relative to annual earnings.
3. **Pipeline Transformers**:
   - `StandardScaler` for numerical scaling.
   - `OneHotEncoder(drop='first')` for categorical encoding.
   - Combined via `ColumnTransformer` to prevent data leakage.

---

## 🤖 Machine Learning Model Performance

All models were evaluated on an unseen **20% Test Set** using **5-Fold Stratified Cross-Validation**:

| Model | CV Mean Accuracy | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.7923** | **0.7317** | **0.7222** | **0.9630** | **0.8254** | **0.6329** |
| **Random Forest** | 0.7618 | 0.7236 | 0.7196 | 0.9506 | 0.8191 | 0.6700 |
| **Decision Tree** | 0.6559 | 0.7154 | 0.7170 | 0.9383 | 0.8128 | 0.5977 |

> **Selected Model:** **Logistic Regression** achieved the highest overall **F1 Score (0.8254)** and **Recall (0.9630)**, making it the most reliable model for identifying creditworthy applicants while maintaining clear explainability.

---

## 🚀 How to Run the Project

### 1. Clone & Setup Environment
```bash
# Navigate to project directory
cd CrediPredict

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure MySQL Database (Optional)
If MySQL is available, execute `sql/database.sql` and `sql/tables.sql` to initialize the database and tables.

### 3. Run Data Pipeline & Train Model
```bash
# Run training script (preprocesses data, tunes models, saves models/loan_approval_model.pkl)
python src/train.py
```

### 4. Run Streamlit Web Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🎓 Interview Explainability & Q&A Guide

When presenting this project in an interview, be prepared to answer:

1. **Why is this a classification problem?**
   - The target variable `Loan_Status` is binary (`Y` or `N`).
2. **Why fill missing numerical values with Median instead of Mean?**
   - Financial variables like income and loan amount exhibit right skewness. The mean is sensitive to extreme values, whereas the median reflects the true central tendency.
3. **Why use ColumnTransformer & Pipeline?**
   - Fitting scalers and encoders inside a Pipeline ensures transformations are fit **only on training folds** during cross-validation, preventing data leakage.
4. **Why use Stratified K-Fold CV?**
   - Stratification maintains the target label proportions (`68% Approved / 32% Rejected`) across each of the 5 cross-validation folds.
5. **Why was Logistic Regression selected over Random Forest?**
   - Logistic Regression yielded higher test set Recall (96.3%) and F1 Score (0.8254) while maintaining maximum model transparency and coefficient explainability.

---

## ⚖️ Responsible AI Disclaimer

*This project was created strictly for educational, research, and technical placement portfolio purposes. The predictions generated by these models should not be used as the sole basis for real-world automated financial or lending decisions without human review, bias checks, and regulatory compliance validation.*
