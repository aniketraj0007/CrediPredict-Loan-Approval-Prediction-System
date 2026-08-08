-- ========================================================
-- CrediPredict - Analytical SQL Queries
-- Description: 10 simple and effective SQL queries for data analysis
-- Author: B.Tech Data Science Student
-- ========================================================

USE credipredict_db;

-- Query 1: Total Applications
SELECT COUNT(*) AS total_applications
FROM applicants;

-- Query 2: Approved Applications Count
SELECT COUNT(*) AS approved_applications
FROM applicants
WHERE loan_status = 'Y';

-- Query 3: Rejected Applications Count
SELECT COUNT(*) AS rejected_applications
FROM applicants
WHERE loan_status = 'N';

-- Query 4: Overall Approval Percentage
SELECT 
    COUNT(*) AS total_applications,
    SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) AS approved,
    ROUND((SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS approval_percentage
FROM applicants;

-- Query 5: Average Applicant Income
SELECT ROUND(AVG(applicant_income), 2) AS avg_applicant_income
FROM applicants;

-- Query 6: Average Loan Amount
SELECT ROUND(AVG(loan_amount), 2) AS avg_loan_amount_thousands
FROM applicants
WHERE loan_amount IS NOT NULL;

-- Query 7: Approval Rate by Education Level
SELECT 
    education,
    COUNT(*) AS total_applicants,
    SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) AS approved_count,
    ROUND((SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS approval_rate_pct
FROM applicants
GROUP BY education;

-- Query 8: Approval Rate by Property Area
SELECT 
    property_area,
    COUNT(*) AS total_applicants,
    SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) AS approved_count,
    ROUND((SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS approval_rate_pct
FROM applicants
GROUP BY property_area;

-- Query 9: Approval Rate by Self Employment Status
SELECT 
    COALESCE(self_employed, 'Unknown') AS self_employed_status,
    COUNT(*) AS total_applicants,
    SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) AS approved_count,
    ROUND((SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS approval_rate_pct
FROM applicants
GROUP BY self_employed;

-- Query 10: Approval Rate by Credit History
SELECT 
    CASE 
        WHEN credit_history = 1.0 THEN 'Good Credit (1.0)'
        WHEN credit_history = 0.0 THEN 'Bad Credit (0.0)'
        ELSE 'Missing Credit History'
    END AS credit_status,
    COUNT(*) AS total_applicants,
    SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) AS approved_count,
    ROUND((SUM(CASE WHEN loan_status = 'Y' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2) AS approval_rate_pct
FROM applicants
GROUP BY credit_history;

-- Query 11: Recent Live Streamlit Predictions Log
SELECT 
    prediction_id,
    gender,
    married,
    education,
    applicant_income,
    loan_amount,
    credit_history,
    property_area,
    prediction,
    probability,
    created_at
FROM prediction_history
ORDER BY created_at DESC
LIMIT 10;

