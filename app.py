# app.py - ENHANCED VERSION with Complete 5Cs of Credit
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Advanced Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load the trained model
try:
    with open("loan_prediction.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    
    # Handle different possible structures in the pickle file
    if len(loaded_data) == 4:
        model, scaler, label_encoders, target_encoder = loaded_data
        st.sidebar.success("✅ Model Loaded Successfully!")
    elif len(loaded_data) == 3:
        model, label_encoders, target_encoder = loaded_data
        scaler = None
        st.sidebar.warning("⚠️ Scaler not found in model file")
    else:
        st.error(f"❌ Unexpected number of objects in pickle file: {len(loaded_data)}")
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.title("🏦 Advanced Loan Approval Prediction App")
st.markdown("Comprehensive loan assessment based on the **5 Cs of Credit**: Character, Capacity, Capital, Collateral, Conditions")

# Input form with ALL required criteria
with st.form("loan_application"):
    st.header("📋 Applicant Information - 5 Cs of Credit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Character & Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        
        # Age input
        age = st.number_input("Age", min_value=18, max_value=70, value=30, 
                             help="Applicants typically between 21-60 years")
        
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        
    with col2:
        st.subheader("💼 Employment & Stability")
        employment_type = st.selectbox(
            "Employment Type", 
            ["Salaried", "Self Employed"],
            help="Select 'Salaried' if you work for an employer, 'Self Employed' if you run your own business"
        )
        
        # Employment stability
        employment_years = st.number_input("Years with Current Employer", 
                                         min_value=0.0, max_value=40.0, value=2.0, step=0.5,
                                         help="Lenders prefer 1-2+ years stability")
        
        self_employed_value = "Yes" if employment_type == "Self Employed" else "No"
        
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    st.header("💰 Financial Capacity & Capital")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💵 Income Details")
        applicant_income = st.number_input("Applicant Monthly Income ($)", 
                                         min_value=0, value=5000, step=100,
                                         help="Minimum typically $2,500+ depending on location")
        coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", 
                                           min_value=0, value=0, step=100)
        
        # Existing debt obligations
        existing_emi = st.number_input("Existing Monthly Debt Payments ($)", 
                                     min_value=0, value=0, step=50,
                                     help="Current loan EMIs, credit card payments")
        
    with col4:
        st.subheader("🏠 Collateral & Assets")
        loan_amount = st.number_input("Loan Amount Requested ($)", 
                                    min_value=0, value=15000, step=100)
        loan_amount_term = st.number_input("Loan Term (months)", 
                                         min_value=12, max_value=360, value=60, step=12)
        
        # Collateral value
        collateral_value = st.number_input("Collateral Value ($ - if any)", 
                                         min_value=0, value=0, step=1000,
                                         help="Property, vehicle, or other asset value")
        
        # Savings/Assets
        total_savings = st.number_input("Total Savings & Investments ($)", 
                                      min_value=0, value=5000, step=500,
                                      help="Demonstrates financial discipline")
    
    st.header("🔐 Credit History & Risk Assessment")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("📊 Credit Profile")
        # Detailed credit score instead of just Good/Bad
        credit_score = st.slider("Credit Score", 
                               min_value=300, max_value=850, value=750, step=10,
                               help="Excellent: 750+, Good: 650-749, Poor: <650")
        
        # Map credit score to the model's expected format
        if credit_score >= 650:
            credit_history = 1.0  # Good
            credit_category = "Good"
        else:
            credit_history = 0.0  # Bad  
            credit_category = "Poor"
            
        st.write(f"**Credit Category:** {credit_category}")
        
    with col6:
        st.subheader("📈 Risk Indicators")
        # Calculate Debt-to-Income Ratio
        total_income = applicant_income + coapplicant_income
        if total_income > 0:
            dti_ratio = (existing_emi / total_income) * 100
        else:
            dti_ratio = 0
            
        st.metric("Debt-to-Income Ratio", f"{dti_ratio:.1f}%")
        
        # DTI assessment
        if dti_ratio <= 36:
            dti_status = "✅ Excellent"
        elif dti_ratio <= 40:
            dti_status = "⚠️ Acceptable"
        elif dti_ratio <= 50:
            dti_status = "⚠️ High"
        else:
            dti_status = "❌ Very High"
            
        st.write(f"**DTI Status:** {dti_status}")
    
    submitted = st.form_submit_button("🚀 Comprehensive Loan Assessment", use_container_width=True)

if submitted:
    # Create input dataframe for ML model (maintaining compatibility)
    input_data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed_value],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],  # Using mapped value
        'Property_Area': [property_area]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Display comprehensive assessment
    st.header("🎯 Comprehensive Loan Assessment Results")
    
    try:
        # Get column order and preprocess for ML model
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        categorical_columns = list(label_encoders.keys())
        expected_columns = categorical_columns + numerical_columns
        
        # Reorder input dataframe
        input_df_ordered = input_df[expected_columns].copy()
        
        # Preprocess input data
        encoded_df = input_df_ordered.copy()
        for column in label_encoders:
            if column in encoded_df.columns:
                encoded_df[column] = label_encoders[column].transform(encoded_df[column])
        
        # Scale features if scaler available
        if scaler is not None:
            input_scaled = scaler.transform(encoded_df)
        else:
            input_scaled = encoded_df.values
        
        # ML Model Prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        result = target_encoder.inverse_transform(prediction)
        ml_confidence = prediction_proba[0][prediction[0]]
        
        # ENHANCED: Comprehensive Risk Scoring System
        risk_score = 0
        approval_factors = []
        rejection_factors = []
        
        # 1. Credit Score Assessment (Character - 30% weight)
        if credit_score >= 750:
            risk_score += 30
            approval_factors.append("Excellent credit score (750+)")
        elif credit_score >= 650:
            risk_score += 20
            approval_factors.append("Good credit score (650-749)")
        else:
            risk_score += 5
            rejection_factors.append("Poor credit score (<650)")
        
        # 2. Income Stability (Capacity - 25% weight)
        total_monthly_income = applicant_income + coapplicant_income
        if total_monthly_income >= 2500:  # Minimum threshold
            risk_score += 15
            approval_factors.append("Sufficient monthly income")
        else:
            risk_score += 5
            rejection_factors.append("Insufficient monthly income")
        
        if employment_years >= 2:
            risk_score += 10
            approval_factors.append("Stable employment history (2+ years)")
        elif employment_years >= 1:
            risk_score += 7
            approval_factors.append("Moderate employment stability (1-2 years)")
        else:
            risk_score += 3
            rejection_factors.append("Limited employment history (<1 year)")
        
        # 3. Debt-to-Income Ratio (Capacity - 20% weight)
        if dti_ratio <= 36:
            risk_score += 20
            approval_factors.append("Excellent debt-to-income ratio (<36%)")
        elif dti_ratio <= 40:
            risk_score += 15
            approval_factors.append("Acceptable debt-to-income ratio (36-40%)")
        elif dti_ratio <= 50:
            risk_score += 8
            rejection_factors.append("High debt-to-income ratio (40-50%)")
        else:
            risk_score += 0
            rejection_factors.append("Very high debt-to-income ratio (>50%)")
        
        # 4. Collateral & Assets (Collateral/Capital - 15% weight)
        if collateral_value >= loan_amount * 0.8:  # 80% collateral coverage
            risk_score += 15
            approval_factors.append("Strong collateral coverage")
        elif collateral_value >= loan_amount * 0.5:
            risk_score += 10
            approval_factors.append("Adequate collateral")
        else:
            risk_score += 5
            if loan_amount > 10000:  # Only mention for larger loans
                rejection_factors.append("Limited collateral for loan amount")
        
        if total_savings >= loan_amount * 0.2:
            risk_score += 5
            approval_factors.append("Strong personal capital/savings")
        
        # 5. Age & Conditions (Conditions - 10% weight)
        if 25 <= age <= 55:
            risk_score += 10
            approval_factors.append("Ideal age for loan tenure")
        elif 21 <= age <= 60:
            risk_score += 7
            approval_factors.append("Acceptable age range")
        else:
            risk_score += 3
            rejection_factors.append("Age outside preferred range")
        
        # FINAL DECISION with Enhanced Rules
        final_confidence = (ml_confidence + (risk_score / 100)) / 2
        confidence_threshold = 0.70
        
        # Enhanced decision rules
        if risk_score >= 70 and final_confidence >= confidence_threshold:
            final_decision = 'Y'
            decision_confidence = "High"
        elif risk_score >= 60 and final_confidence >= confidence_threshold * 0.9:
            final_decision = 'Y' 
            decision_confidence = "Moderate"
        else:
            final_decision = 'N'
            decision_confidence = "Low"
        
        # Override: Automatic rejections
        if credit_score < 600:
            final_decision = 'N'
            rejection_factors.append("Very poor credit score (automatic rejection)")
        if dti_ratio > 60:
            final_decision = 'N'
            rejection_factors.append("Extremely high debt-to-income ratio (automatic rejection)")
        if age < 21 or age > 65:
            final_decision = 'N'
            rejection_factors.append("Age outside lending criteria")
        
        # Display Results
        st.subheader("📊 Risk Assessment Score")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Comprehensive Risk Score", f"{risk_score}/100")
        
        with col2:
            st.metric("ML Model Confidence", f"{ml_confidence:.2%}")
        
        with col3:
            st.metric("Final Confidence", f"{final_confidence:.2%}")
        
        # Final Decision
        st.subheader("🎯 Final Decision")
        
        if final_decision == 'Y':
            st.success(f"## ✅ LOAN APPROVED - {decision_confidence} Confidence")
            st.balloons()
            
            st.info("### ✅ Approval Factors:")
            for factor in approval_factors:
                st.write(f"• {factor}")
                
        else:
            st.error(f"## ❌ LOAN NOT APPROVED - {decision_confidence} Confidence")
            
            if rejection_factors:
                st.warning("### ❌ Primary Concerns:")
                for factor in rejection_factors:
                    st.write(f"• {factor}")
            
            if approval_factors:
                st.info("### ✅ Positive Factors:")
                for factor in approval_factors:
                    st.write(f"• {factor}")
        
        # Detailed Breakdown
        st.subheader("📈 Detailed Assessment Breakdown")
        
        assessment_data = {
            'Criteria': [
                'Credit Score (Character)',
                'Income & Employment (Capacity)', 
                'Debt-to-Income Ratio (Capacity)',
                'Collateral & Assets (Collateral/Capital)',
                'Age & Conditions (Conditions)',
                'ML Model Prediction'
            ],
            'Score': [30, 25, 20, 20, 10, 'N/A'],
            'Your Score': [
                f"{min(30, max(5, (credit_score-300)//15))}/30",
                f"{min(25, 15 + min(10, employment_years*5))}/25", 
                f"{min(20, max(0, 20 - max(0, dti_ratio-36)//2))}/20",
                f"{min(20, 5 + min(15, collateral_value//1000))}/20",
                f"{min(10, max(3, 10 - abs(age-40)//10))}/10",
                f"{ml_confidence:.2%}"
            ],
            'Status': [
                "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Poor",
                "Stable" if employment_years >= 2 else "Moderate" if employment_years >= 1 else "Limited",
                "Excellent" if dti_ratio <= 36 else "Good" if dti_ratio <= 40 else "High" if dti_ratio <= 50 else "Very High",
                "Strong" if collateral_value >= loan_amount*0.8 else "Adequate" if collateral_value >= loan_amount*0.5 else "Limited",
                "Ideal" if 25 <= age <= 55 else "Acceptable" if 21 <= age <= 60 else "Outside Range",
                "High" if ml_confidence >= 0.7 else "Moderate" if ml_confidence >= 0.5 else "Low"
            ]
        }
        
        assessment_df = pd.DataFrame(assessment_data)
        st.dataframe(assessment_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations & Next Steps")
        
        if final_decision == 'Y':
            st.success("""
            **Next Steps for Approval:**
            • Submit required documentation (income proof, identity, address)
            • Complete KYC verification
            • Review and accept loan terms
            • Funds will be disbursed within 3-5 business days
            """)
        else:
            st.info("""
            **Suggestions for Improvement:**
            • Improve credit score by paying existing debts on time
            • Reduce debt-to-income ratio by paying down existing loans
            • Maintain stable employment for 6-12 months
            • Consider adding a co-applicant with good credit
            • Provide additional collateral if available
            • Reapply after addressing these factors
            """)
            
    except Exception as e:
        st.error(f"❌ Error in assessment: {str(e)}")

# Enhanced sidebar information
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 5 Cs of Credit Assessment")
    st.markdown("""
    **Character** - Credit history & score
    **Capacity** - Income stability & DTI ratio  
    **Capital** - Savings & net worth
    **Collateral** - Asset security
    **Conditions** - Age, loan purpose, terms
    """)
    
    st.markdown("### 🎯 Approval Guidelines")
    st.markdown("""
    **Excellent**: 750+ credit, <36% DTI, 2+ years employment
    **Good**: 650-749 credit, <40% DTI, 1+ year employment  
    **Poor**: <650 credit, >50% DTI, unstable income
    """)
    
    st.markdown("### ⚠️ Automatic Rejections")
    st.markdown("""
    • Credit score < 600
    • DTI ratio > 60%
    • Age < 21 or > 65
    • Insufficient income
    """)
