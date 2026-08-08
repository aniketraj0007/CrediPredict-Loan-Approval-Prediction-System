"""
CrediPredict - Loan Approval Prediction System
Streamlit Web Application
Written for B.Tech Data Science Placement Project
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_raw_data, save_prediction_history
from predict import get_predictor

# Page configuration
st.set_page_config(
    page_title="CrediPredict - Loan Approval System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Clean, Human-Designed Student UI
st.markdown("""
<style>
    /* Main Font & Page Styling */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
    }
    
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 16px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    .sidebar-brand {
        font-size: 22px;
        font-weight: 700;
        color: #1D4ED8;
        margin-bottom: 0px;
        letter-spacing: -0.3px;
    }
    
    .sidebar-subbrand {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 16px;
    }
    
    .sidebar-footer {
        font-size: 12px;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Sidebar Buttons Custom Design */
    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        border: none !important;
        background-color: transparent !important;
        color: #334155 !important;
        padding: 10px 14px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }
    
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        transform: translateX(3px);
    }
    
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-weight: 600 !important;
        border-left: 4px solid #1D4ED8 !important;
        border-radius: 4px 6px 6px 4px !important;
    }
    
    /* Primary Buttons Styling */
    div.stButton > button[kind="primary"]:not([data-testid="stSidebar"] button) {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button[kind="primary"]:not([data-testid="stSidebar"] button):hover {
        background-color: #1E40AF !important;
        transform: translateY(-1px);
    }
    
    /* Form Submit Button */
    div.stFormSubmitButton > button {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    div.stFormSubmitButton > button:hover {
        background-color: #1E40AF !important;
    }
    
    /* Result Cards */
    .result-card-approved {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    
    .result-card-rejected {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    
    .result-status-approved {
        font-size: 26px;
        font-weight: 700;
        color: #16A34A;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    
    .result-status-rejected {
        font-size: 26px;
        font-weight: 700;
        color: #DC2626;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    
    .result-prob {
        font-size: 18px;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State for Page Navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown('<div class="sidebar-brand">CrediPredict</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subbrand">Loan Approval Prediction</div>', unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 12px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

nav_items = [
    ("Home", "🏠  Home"),
    ("Loan Prediction", "💰  Loan Prediction"),
    ("About", "ℹ️  About")
]

for page_key, label in nav_items:
    is_active = (st.session_state['current_page'] == page_key)
    if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state['current_page'] = page_key
        st.rerun()




# ==========================================
# PAGE 1: HOME
# ==========================================
if st.session_state['current_page'] == "Home":
    st.markdown('<div class="main-title">CrediPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Loan Approval Prediction System</div>', unsafe_allow_html=True)
    st.write("Enter applicant details and check whether the loan is likely to be approved or rejected.")
    
    st.write("")
    if st.button("Check Loan Eligibility", key="home_cta", type="primary"):
        st.session_state['current_page'] = "Loan Prediction"
        st.rerun()

    st.divider()

    st.subheader("Loan Application Overview")
    
    try:
        df = load_raw_data()
        df_clean = df.copy()
        df_clean['Loan_Status_Full'] = df_clean['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected'})
        df_clean['Credit_History_Label'] = df_clean['Credit_History'].map({1.0: 'Good (1.0)', 0.0: 'Poor (0.0)'}).fillna('Unknown')

        total_apps = len(df_clean)
        approved_apps = (df_clean['Loan_Status'] == 'Y').sum()
        rejected_apps = (df_clean['Loan_Status'] == 'N').sum()
        approval_rate = (approved_apps / total_apps) * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Applications", f"{total_apps}")
        m2.metric("Approved", f"{approved_apps}")
        m3.metric("Rejected", f"{rejected_apps}")
        m4.metric("Approval Rate", f"{approval_rate:.1f}%")

        st.write("")
        st.subheader("Application Insights")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Loan Status")
            fig, ax = plt.subplots(figsize=(5, 3))
            status_counts = df_clean['Loan_Status_Full'].value_counts()
            ax.bar(status_counts.index, status_counts.values, color=['#16A34A', '#DC2626'], width=0.4)
            ax.set_ylabel("Applications")
            sns.despine()
            st.pyplot(fig)

        with col_right:
            st.markdown("##### Credit History")
            fig, ax = plt.subplots(figsize=(5, 3))
            credit_approval = df_clean.groupby('Credit_History_Label')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
            ax.bar(credit_approval.index, credit_approval.values, color='#1D4ED8', width=0.4)
            ax.set_ylabel("Approval Rate (%)")
            ax.set_ylim(0, 100)
            sns.despine()
            st.pyplot(fig)

        col_bottom_left, col_bottom_right = st.columns(2)

        with col_bottom_left:
            st.markdown("##### Approval by Education")
            fig, ax = plt.subplots(figsize=(5, 3))
            edu_approval = df_clean.groupby('Education')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
            ax.bar(edu_approval.index, edu_approval.values, color='#1D4ED8', width=0.4)
            ax.set_ylabel("Approval Rate (%)")
            ax.set_ylim(0, 100)
            sns.despine()
            st.pyplot(fig)

        with col_bottom_right:
            st.markdown("##### Approval by Property Area")
            fig, ax = plt.subplots(figsize=(5, 3))
            area_approval = df_clean.groupby('Property_Area')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
            ax.bar(area_approval.index, area_approval.values, color='#1D4ED8', width=0.4)
            ax.set_ylabel("Approval Rate (%)")
            ax.set_ylim(0, 100)
            sns.despine()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Unable to load analytics data: {str(e)}")


# ==========================================
# PAGE 2: LOAN PREDICTION
# ==========================================
elif st.session_state['current_page'] == "Loan Prediction":
    st.markdown('<div class="main-title">Loan Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter the applicant details below.</div>', unsafe_allow_html=True)
    st.divider()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Applicant Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            
        with col2:
            st.write("")
            st.write("")
            applicant_income = st.number_input("Applicant Income ($/month)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income ($/month)", min_value=0, value=1500, step=500)
            loan_amount = st.number_input("Loan Amount ($ in Thousands)", min_value=1, value=150, step=10)
            loan_amount_term = st.selectbox("Loan Amount Term (Months)", [360, 180, 120, 240, 300, 480, 84, 60], index=0)
            property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

        st.write("")
        st.subheader("Credit Information")
        credit_history_label = st.selectbox(
            "Credit History",
            ["1 - Good Credit History (Meets Guidelines)", "0 - Poor / No Credit History (Does Not Meet Guidelines)"]
        )
        credit_history = 1.0 if "1 -" in credit_history_label else 0.0

        st.write("")
        submit_button = st.form_submit_button("Check Loan Eligibility")

    if submit_button:
        if applicant_income < 0 or coapplicant_income < 0:
            st.error("Income amounts cannot be negative.")
        elif loan_amount <= 0 or loan_amount_term <= 0:
            st.error("Loan amount and term must be greater than zero.")
        else:
            try:
                predictor = get_predictor()
                applicant_payload = {
                    "gender": gender,
                    "married": married,
                    "dependents": dependents,
                    "education": education,
                    "self_employed": self_employed,
                    "applicant_income": float(applicant_income),
                    "coapplicant_income": float(coapplicant_income),
                    "loan_amount": float(loan_amount),
                    "loan_amount_term": float(loan_amount_term),
                    "credit_history": float(credit_history),
                    "property_area": property_area
                }
                
                result = predictor.predict(applicant_payload)
                prediction_status = result["prediction"]
                approval_probability = result["probability"]

                st.write("")
                st.divider()
                st.subheader("Prediction Result")

                if prediction_status == "Approved":
                    st.markdown(f'''
                    <div class="result-card-approved">
                        <div style="font-size: 13px; color: #64748B; font-weight: 500;">Loan Status</div>
                        <div class="result-status-approved">✓ APPROVED</div>
                        <div class="result-prob">Approval Probability: <b>{approval_probability * 100:.1f}%</b></div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="result-card-rejected">
                        <div style="font-size: 13px; color: #64748B; font-weight: 500;">Loan Status</div>
                        <div class="result-status-rejected">✕ REJECTED</div>
                        <div class="result-prob">Approval Probability: <b>{approval_probability * 100:.1f}%</b></div>
                    </div>
                    ''', unsafe_allow_html=True)

                st.write("")
                st.subheader("Application Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.write(f"**Applicant Income:** ${applicant_income:,.0f}")
                    st.write(f"**Loan Amount:** ${loan_amount:,.0f}k")
                    st.write(f"**Credit History:** {'Good (1.0)' if credit_history == 1.0 else 'Poor (0.0)'}")
                with summary_col2:
                    st.write(f"**Education:** {education}")
                    st.write(f"**Property Area:** {property_area}")
                    st.write(f"**Loan Term:** {loan_amount_term} months")

                # Database Save
                db_payload = {**applicant_payload, "prediction": prediction_status, "probability": approval_probability}
                saved, db_msg = save_prediction_history(db_payload)
                if not saved:
                    st.caption(f"Database status: {db_msg}")

            except Exception as e:
                st.error(f"Error during prediction execution: {str(e)}")


# ==========================================
# PAGE 3: ABOUT
# ==========================================
elif st.session_state['current_page'] == "About":
    st.title("About CrediPredict")
    st.divider()

    st.markdown(
        "CrediPredict is a machine learning application that predicts whether a loan application is likely to be "
        "<span style='color: #16A34A; font-weight: 600;'>Approved</span> or "
        "<span style='color: #DC2626; font-weight: 600;'>Rejected</span> based on applicant and financial information.",
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "**How it works**\n\n"
        "You provide the applicant details. The system processes the information and uses a trained machine learning model "
        "to predict the loan approval status along with the probability."
    )

    st.write("")
    st.subheader("Key Points")

    st.markdown("""
    ✓ **Predicts loan approval** (Approved / Rejected)  
    ✓ **Uses machine learning models** trained on historical data  
    ✓ **Uses applicant and financial information** for prediction
    """)





