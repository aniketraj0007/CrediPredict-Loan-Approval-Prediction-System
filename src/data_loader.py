"""
CrediPredict - Data Loader Module
Handles loading raw loan application dataset from MySQL database or fallback CSV file.
Written for B.Tech Data Science project.
"""

import os
import pandas as pd
from sqlalchemy import create_engine

# Database Connection Defaults
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "credipredict_db")
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "loan_data.csv")


def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine for MySQL.
    """
    connection_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_uri)


def load_raw_data(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw loan dataset from local CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Loaded {len(df)} records from CSV: {csv_path}")
    return df


def load_data_from_db(query: str = "SELECT * FROM applicants") -> pd.DataFrame:
    """
    Loads dataset from MySQL database. Falls back to CSV if database connection fails.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            print(f"[DataLoader] Successfully fetched {len(df)} records from MySQL database.")
            return df
    except Exception as e:
        print(f"[DataLoader Warning] MySQL connection failed ({e}). Falling back to local CSV...")
        return load_raw_data()


def save_prediction_history(pred_data: dict) -> tuple:
    """
    Attempts to insert a prediction record into MySQL prediction_history table.
    Returns tuple: (success: bool, message: str)
    """
    try:
        engine = get_db_engine()
        df_row = pd.DataFrame([{
            'gender': pred_data.get('gender'),
            'married': pred_data.get('married'),
            'dependents': str(pred_data.get('dependents')),
            'education': pred_data.get('education'),
            'self_employed': pred_data.get('self_employed'),
            'applicant_income': float(pred_data.get('applicant_income', 0)),
            'coapplicant_income': float(pred_data.get('coapplicant_income', 0)),
            'loan_amount': float(pred_data.get('loan_amount', 0)),
            'loan_amount_term': float(pred_data.get('loan_amount_term', 360)),
            'credit_history': float(pred_data.get('credit_history', 1.0)),
            'property_area': pred_data.get('property_area'),
            'prediction': pred_data.get('prediction'),
            'probability': float(pred_data.get('probability', 0.0))
        }])
        df_row.to_sql('prediction_history', con=engine, if_exists='append', index=False)
        return True, "Prediction successfully saved to database."
    except Exception as e:
        return False, f"MySQL Connection Error: {str(e)}"


if __name__ == "__main__":
    print("Testing Data Loader Module...")
    df = load_raw_data()
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))

