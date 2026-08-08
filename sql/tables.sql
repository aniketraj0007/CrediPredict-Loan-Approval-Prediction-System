-- ========================================================
-- CrediPredict - Applicants Table Schema
-- Description: Table schema to store loan applicant records
-- ========================================================

USE credipredict_db;

-- Drop table if exists
DROP TABLE IF EXISTS applicants;

-- Create applicants table
CREATE TABLE applicants (
    applicant_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10),
    married VARCHAR(10),
    dependents VARCHAR(10),
    education VARCHAR(20),
    self_employed VARCHAR(10),
    applicant_income INT,
    coapplicant_income DOUBLE,
    loan_amount DOUBLE,
    loan_amount_term DOUBLE,
    credit_history DOUBLE,
    property_area VARCHAR(20),
    loan_status VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for common query performance
CREATE INDEX idx_loan_status ON applicants(loan_status);
CREATE INDEX idx_credit_history ON applicants(credit_history);

-- ========================================================
-- CrediPredict - Prediction History Table Schema
-- Description: Table schema to store online Streamlit predictions
-- ========================================================

DROP TABLE IF EXISTS prediction_history;

CREATE TABLE prediction_history (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    gender VARCHAR(10),
    married VARCHAR(10),
    dependents VARCHAR(10),
    education VARCHAR(20),
    self_employed VARCHAR(10),
    applicant_income DOUBLE,
    coapplicant_income DOUBLE,
    loan_amount DOUBLE,
    loan_amount_term DOUBLE,
    credit_history DOUBLE,
    property_area VARCHAR(20),
    prediction VARCHAR(20),
    probability DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_created_at ON prediction_history(created_at);
CREATE INDEX idx_pred_result ON prediction_history(prediction);

