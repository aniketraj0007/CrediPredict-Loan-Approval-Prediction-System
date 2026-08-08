"""
CrediPredict - Data Preprocessing & Feature Engineering Module
Provides reusable data cleaning, feature engineering, and scikit-learn preprocessing pipeline.
Written for B.Tech Data Science project.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw loan application dataframe:
    - Fills categorical missing values with Mode.
    - Fills numerical missing values with Median.
    """
    df_clean = df.copy()

    # Categorical columns imputation using mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # Credit History is categorical in nature (1.0 / 0.0), fill with mode
    if 'Credit_History' in df_clean.columns and df_clean['Credit_History'].isnull().sum() > 0:
        ch_mode = df_clean['Credit_History'].mode()[0]
        df_clean['Credit_History'] = df_clean['Credit_History'].fillna(ch_mode)

    # Numerical columns imputation using median
    num_cols = ['LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-relevant features:
    1. Total_Income = ApplicantIncome + CoapplicantIncome
    2. Loan_to_Income_Ratio = LoanAmount / Total_Income (LoanAmount in thousands vs Total_Income)
    """
    df_fe = df.copy()

    # Combined income of applicant and coapplicant
    df_fe['Total_Income'] = df_fe['ApplicantIncome'] + df_fe['CoapplicantIncome']

    # Loan to Income ratio (LoanAmount is in thousands, convert to actual amount for ratio)
    # Ratio = (LoanAmount * 1000) / (Total_Income + 1e-5)
    df_fe['Loan_to_Income_Ratio'] = np.where(
        df_fe['Total_Income'] > 0,
        (df_fe['LoanAmount'] * 1000) / df_fe['Total_Income'],
        0.0
    )
    df_fe['Loan_to_Income_Ratio'] = np.round(df_fe['Loan_to_Income_Ratio'], 4)

    return df_fe


def build_preprocessor():
    """
    Returns scikit-learn ColumnTransformer for categorical and numerical features.
    Categorical features: OneHotEncoder
    Numerical features: StandardScaler
    """
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Loan_to_Income_Ratio']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    return preprocessor, num_features, cat_features


if __name__ == "__main__":
    from data_loader import load_raw_data
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    fe_df = engineer_features(clean_df)
    print("Cleaned & Engineered Data Shape:", fe_df.shape)
    print("Engineered Columns:", ['Total_Income', 'Loan_to_Income_Ratio'])
    print(fe_df[['ApplicantIncome', 'CoapplicantIncome', 'Total_Income', 'LoanAmount', 'Loan_to_Income_Ratio']].head(3))
