"""
CrediPredict - Inference Prediction Module
Loads saved joblib pipeline and generates loan approval predictions and probabilities.
Written for B.Tech Data Science project.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "loan_approval_model.pkl")


class LoanPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model file not found at: {model_path}. Please run src/train.py first.")
        
        self.pipeline = joblib.load(model_path)
        print(f"[LoanPredictor] Loaded model pipeline from: {model_path}")

    def predict(self, applicant_data: dict) -> dict:
        """
        Accepts dictionary of applicant information and returns prediction and probability.
        """
        # Map input keys to dataframe format
        df_input = pd.DataFrame([{
            'Gender': applicant_data.get('gender', 'Male'),
            'Married': applicant_data.get('married', 'Yes'),
            'Dependents': str(applicant_data.get('dependents', '0')),
            'Education': applicant_data.get('education', 'Graduate'),
            'Self_Employed': applicant_data.get('self_employed', 'No'),
            'ApplicantIncome': float(applicant_data.get('applicant_income', 5000)),
            'CoapplicantIncome': float(applicant_data.get('coapplicant_income', 0)),
            'LoanAmount': float(applicant_data.get('loan_amount', 150)),
            'Loan_Amount_Term': float(applicant_data.get('loan_amount_term', 360)),
            'Credit_History': float(applicant_data.get('credit_history', 1.0)),
            'Property_Area': applicant_data.get('property_area', 'Urban')
        }])

        # Feature Engineering for incoming payload
        df_input['Total_Income'] = df_input['ApplicantIncome'] + df_input['CoapplicantIncome']
        df_input['Loan_to_Income_Ratio'] = np.where(
            df_input['Total_Income'] > 0,
            (df_input['LoanAmount'] * 1000) / df_input['Total_Income'],
            0.0
        )
        df_input['Loan_to_Income_Ratio'] = np.round(df_input['Loan_to_Income_Ratio'], 4)

        # Predict status and probability
        prediction_class = self.pipeline.predict(df_input)[0]
        prediction_prob = self.pipeline.predict_proba(df_input)[0][1]

        status = "Approved" if prediction_class == 1 else "Rejected"

        return {
            "prediction": status,
            "probability": round(float(prediction_prob), 4),
            "confidence_pct": round(float(prediction_prob if prediction_class == 1 else (1 - prediction_prob)) * 100, 2)
        }


# Global singleton instance for API import
_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LoanPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    sample_applicant = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 6000,
        "coapplicant_income": 2000,
        "loan_amount": 140,
        "loan_amount_term": 360,
        "credit_history": 1.0,
        "property_area": "Semiurban"
    }
    result = predictor.predict(sample_applicant)
    print("\nSample Prediction Test Result:")
    print(result)
