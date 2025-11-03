# app.py 
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load the trained model with better error handling
try:
    with open("loan_prediction.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    
    # Handle different possible structures in the pickle file
    if len(loaded_data) == 4:
        model, scaler, label_encoders, target_encoder = loaded_data
        st.sidebar.success("✅ Model Loaded Successfully!")
    elif:
        st.error(f"❌ Unexpected number of objects in pickle file: {len(loaded_data)}")
        st.stop()
    
#     # Display debug info in sidebar
    with st.sidebar:
        st.write("**Model Performance:**")
        st.metric("Accuracy", "84.55%")
        st.write("**Features:**", 11)
        st.write("**Scaler Available:**", "Yes" if scaler is not None else "No")
        
        # Show available encoders
        with st.expander("🔍 Model Info"):
            st.write("**Available Encoders:**", list(label_encoders.keys()))
            for col, encoder in label_encoders.items():
                st.write(f"**{col}:** {list(encoder.classes_)}")
        
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

st.title("🏦 Loan Approval Prediction App")
st.markdown("Predict whether a loan application will be approved based on applicant information")

# Input form
with st.form("loan_application"):
    st.header("📋 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        
    with col2:
        st.subheader("Employment & Property")
        employment_type = st.selectbox(
            "Employment Type", 
            ["Salaried", "Self Employed"],
            help="Select 'Salaried' if you work for an employer, 'Self Employed' if you run your own business"
        )
        
        # Map the user-friendly options to model-compatible values
        self_employed_value = "Yes" if employment_type == "Self Employed" else "No"
        
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        credit_history = st.selectbox("Credit History", [1.0, 0.0], 
                                    format_func=lambda x: "Good" if x == 1.0 else "Bad")
    
    st.header("💰 Financial Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Income Details")
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=100)
        
    with col4:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150, step=10)
        loan_amount_term = st.number_input("Loan Term (months)", min_value=0, value=24, step=12)
    
    submitted = st.form_submit_button("🚀 Predict Loan Approval", use_container_width=True)

if submitted:
    # Create input dataframe
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
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Display the raw input
    with st.expander("📊 Input Data Summary"):
        st.write("**Your Application Details:**")
        st.dataframe(input_df, use_container_width=True)
        st.write(f"**Employment Type:** {employment_type} → Self_Employed: {self_employed_value}")
        
    try:
        # FIX 1: Get the EXACT column order from the model or scaler
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            # Best option: get column order from scaler
            expected_columns = list(scaler.feature_names_in_)
            st.sidebar.info(f"📋 Using scaler feature order")
        elif hasattr(model, 'feature_names_in_'):
            # Alternative: get from model
            expected_columns = list(model.feature_names_in_)
            st.sidebar.info(f"📋 Using model feature order")
        else:
            # Fallback: reconstruct order from encoders + numerical columns
            numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            categorical_columns = list(label_encoders.keys())
            expected_columns = categorical_columns + numerical_columns
            st.sidebar.warning("⚠️ Reconstructed column order")
        
        st.sidebar.write(f"**Expected columns:** {expected_columns}")
        
        # FIX 2: Check for missing or extra columns
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        extra_columns = [col for col in input_df.columns if col not in expected_columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            st.stop()
        if extra_columns:
            st.warning(f"⚠️ Extra columns (will be removed): {extra_columns}")
        
        # FIX 3: Reorder columns to EXACTLY match training order
        input_df_ordered = input_df[expected_columns].copy()
        
        # Display column order info
        with st.expander("🔍 Column Order Debug"):
            st.write("**Input DataFrame Columns:**", list(input_df.columns))
            st.write("**Expected Columns:**", expected_columns)
            st.write("**Final Ordered Columns:**", list(input_df_ordered.columns))
        
        # Preprocess input data using saved encoders
        encoded_df = input_df_ordered.copy()
        encoding_info = {}
        
        for column in label_encoders:
            if column in encoded_df.columns:
                original_value = encoded_df[column].iloc[0]
                try:
                    encoded_value = label_encoders[column].transform([original_value])[0]
                    encoded_df[column] = encoded_value
                    encoding_info[column] = {
                        'original': original_value,
                        'encoded': encoded_value,
                        'mapping': dict(zip(label_encoders[column].classes_, 
                                          label_encoders[column].transform(label_encoders[column].classes_)))
                    }
                except ValueError as e:
                    st.error(f"❌ Encoding error for {column}: '{original_value}' not in {list(label_encoders[column].classes_)}")
                    st.stop()
        
        # Display encoding info
        with st.expander("🔍 Encoding Information"):
            st.write("**Encoding Transformations:**")
            for col, info in encoding_info.items():
                st.write(f"**{col}:** '{info['original']}' → {info['encoded']}")
        
        # Scale features if scaler is available
        if scaler is not None:
            input_scaled = scaler.transform(encoded_df)
        else:
            # If no scaler, use the encoded data directly
            input_scaled = encoded_df.values
            st.warning("⚠️ Using unscaled features (scaler not available)")
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Convert prediction back to original label
        result = target_encoder.inverse_transform(prediction)
        probability = prediction_proba[0][prediction[0]]
        
        # Apply business rules
        final_decision = result[0]
        confidence_threshold = 0.70  # 70% confidence threshold
        credit_history_bad = credit_history == 0.0
        
        # Rule 1: Reject if confidence level is below 70%
        if probability < confidence_threshold:
            final_decision = 'N'
            rejection_reason = f"Confidence level ({probability:.2%}) below required threshold ({confidence_threshold:.0%})"
        
        # Rule 2: Reject if confidence above 70% but credit history is bad
        elif probability >= confidence_threshold and credit_history_bad:
            final_decision = 'N'
            rejection_reason = "Credit history is poor despite high confidence"
        
        else:
            rejection_reason = None
        
        # Display results
        st.header("🎯 Prediction Results")
        
        result_col, prob_col = st.columns(2)
        
        with result_col:
            if final_decision == 'Y':
                st.success(f"## ✅ Loan Approved!")
                st.balloons()
            else:
                st.error(f"## ❌ Loan Not Approved")
                if rejection_reason:
                    st.warning(f"**Reason:** {rejection_reason}")
        
        with prob_col:
            confidence_level = "High" if probability > 0.7 else "Medium" if probability > 0.5 else "Low"
            status_color = "normal"
            
            # Color code based on final decision
            if final_decision == 'Y':
                status_color = "normal"
            else:
                if probability < confidence_threshold:
                    status_color = "off"
                else:
                    status_color = "inverse"
            
            st.metric(
                label="Confidence Level",
                value=f"{probability:.2%}",
                delta=confidence_level,
                delta_color=status_color
            )
        
        # Show detailed probabilities
        st.subheader("Detailed Probabilities")
        prob_df = pd.DataFrame({
            'Class': target_encoder.classes_,
            'Probability': prediction_proba[0],
            'Description': ['Loan Rejected', 'Loan Approved']
        })
        st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}).hide(axis='index'), 
                    use_container_width=True)
        
        # Show business rules applied
        st.subheader("📋 Business Rules Applied")
        rules_df = pd.DataFrame({
            'Rule': [
                'Confidence Threshold (70%)',
                'Credit History Check',
                'Final Decision'
            ],
            'Status': [
                f"{'✅ Met' if probability >= confidence_threshold else '❌ Not Met'} ({probability:.2%})",
                f"{'✅ Good' if not credit_history_bad else '❌ Poor'}",
                f"{'✅ Approved' if final_decision == 'Y' else '❌ Rejected'}"
            ],
            'Description': [
                f"Model confidence must be ≥ {confidence_threshold:.0%}",
                "Credit history must be good for approval",
                "Based on all rules and model prediction"
            ]
        })
        st.dataframe(rules_df, use_container_width=True)
        
        # Show feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("📈 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': encoded_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(feature_importance, use_container_width=True)
        
        # Interpretation
        st.subheader("💡 Interpretation")
        if final_decision == 'Y':
            st.info(f"""
            **Factors that contributed to approval:**
            - Good credit history
            - {'Stable salaried employment' if employment_type == 'Salaried' else 'Self-employed business'}
            - Favorable property area
            - Strong financial profile
            - High confidence level ({probability:.2%})
            """)
        else:
            if probability < confidence_threshold:
                st.info(f"""
                **Reasons for rejection:**
                - Confidence level ({probability:.2%}) below required threshold ({confidence_threshold:.0%})
                - Model prediction is not sufficiently certain
                - Consider providing additional documentation
                """)
            elif credit_history_bad:
                st.info(f"""
                **Reasons for rejection:**
                - Poor credit history (automatic rejection)
                - Despite high model confidence ({probability:.2%})
                - Improve credit score before reapplying
                """)
            else:
                st.info(f"""
                **Factors that may need improvement:**
                - Consider improving credit history
                - {'Provide additional employment documentation' if employment_type == 'Salaried' else 'Provide business financial statements'}
                - Adjust loan amount relative to income
                - Increase income stability
                """)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        
        # Enhanced debugging information
        with st.expander("🔧 Technical Debug Information"):
            st.write(f"**Error type:** {type(e).__name__}")
            st.write(f"**Error message:** {str(e)}")
            
            # Show what we have available
            st.write("**Available objects:**")
            st.write(f"- Model: {type(model)}")
            st.write(f"- Scaler: {type(scaler) if scaler else 'None'}")
            st.write(f"- Label Encoders: {list(label_encoders.keys())}")
            
            if 'input_df_ordered' in locals():
                st.write("**Input DataFrame Columns:**", list(input_df_ordered.columns))
            if 'expected_columns' in locals():
                st.write("**Expected Columns:**", expected_columns)

# Add help section in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 How to Use:")
    st.markdown("1. Fill all applicant details")
    st.markdown("2. Select employment type: 'Salaried' or 'Self Employed'")
    st.markdown("3. Enter financial information")
    st.markdown("4. Click 'Predict Loan Approval'")
    st.markdown("5. Review results and insights")
    
    st.markdown("### 📊 Model Info:")
    st.markdown("- **Algorithm**: Random Forest")
    st.markdown("- **Training Data**: 480 records")
    st.markdown("- **Features**: 11 variables")
    st.markdown("- **Accuracy**: 84.55%")
    
    st.markdown("### 🛡️ Business Rules:")
    st.markdown("- **Confidence**: ≥70% required")
    st.markdown("- **Credit History**: Must be good")
    st.markdown("- **Final Decision**: Based on all criteria")


