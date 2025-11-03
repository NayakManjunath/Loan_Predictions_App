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

# Load the trained model
try:
    with open("loan_prediction.pkl", "rb") as f:
        model, scaler, label_encoders, target_encoder = pickle.load(f)
    
    # Display debug info in sidebar
    with st.sidebar:
        st.success("✅ Model Loaded Successfully!")
        st.write("**Model Performance:**")
        st.metric("Accuracy", "84.55%")
        st.write("**Available Encoders:**", list(label_encoders.keys()))
        
        # Show encoding mappings for debugging
        with st.expander("🔍 Model Encoding Info"):
            for col, encoder in label_encoders.items():
                st.write(f"**{col}:** {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
            st.write(f"**Target:** {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
        
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
        
        # FIX: Check what education options the model was trained on
        education_options = list(label_encoders['Education'].classes_) if 'Education' in label_encoders else ["Graduate", "Not Graduate"]
        education = st.selectbox("Education", education_options)
        
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        
    with col2:
        st.subheader("Employment & Property")
        
        # FIX: Get actual Self_Employed options from the model
        self_employed_options = list(label_encoders['Self_Employed'].classes_) if 'Self_Employed' in label_encoders else ["No", "Yes"]
        
        if len(self_employed_options) == 2 and set(self_employed_options) == {"No", "Yes"}:
            # Use the mapping approach for better UX
            employment_type = st.selectbox(
                "Employment Type", 
                ["Salaried", "Self Employed"],
                help="Select 'Salaried' if you work for an employer, 'Self Employed' if you run your own business"
            )
            self_employed_value = "Yes" if employment_type == "Self Employed" else "No"
        else:
            # Use direct model options
            self_employed_value = st.selectbox("Self Employed", self_employed_options)
        
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
        loan_amount_term = st.number_input("Loan Term (months)", min_value=0, value=360, step=12)
    
    submitted = st.form_submit_button("🚀 Predict Loan Approval", use_container_width=True)

if submitted:
    # Create input dataframe with EXACTLY the same columns as training
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
        
    try:
        # FIX: Ensure column order matches training data
        expected_columns = list(label_encoders.keys()) + ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        input_df = input_df[expected_columns]
        
        # Preprocess input data using saved encoders
        encoded_df = input_df.copy()
        encoding_debug = {}
        
        for column in label_encoders:
            if column in encoded_df.columns:
                original_value = encoded_df[column].iloc[0]
                try:
                    encoded_df[column] = label_encoders[column].transform(encoded_df[column])
                    encoding_debug[column] = {
                        'original': original_value,
                        'encoded': encoded_df[column].iloc[0],
                        'mapping': dict(zip(label_encoders[column].classes_, range(len(label_encoders[column].classes_))))
                    }
                except ValueError as e:
                    st.error(f"❌ Encoding error for {column}: '{original_value}' not in {list(label_encoders[column].classes_)}")
                    st.stop()
        
        # Display encoding debug info
        with st.expander("🔍 Encoding Debug Info"):
            st.write("**Encoding Transformations:**")
            for col, debug_info in encoding_debug.items():
                st.write(f"**{col}:** '{debug_info['original']}' → {debug_info['encoded']}")
            st.write("**Full Encoding Mapping:**")
            for col, debug_info in encoding_debug.items():
                st.write(f"{col}: {debug_info['mapping']}")
        
        # Display final encoded data
        with st.expander("🔍 Final Encoded Data"):
            st.write("**Data sent to model:**")
            st.dataframe(encoded_df, use_container_width=True)
            st.write(f"**Data shape:** {encoded_df.shape}")
            st.write(f"**Data types:** {encoded_df.dtypes.to_dict()}")
        
        # Scale features
        input_scaled = scaler.transform(encoded_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Convert prediction back to original label
        result = target_encoder.inverse_transform(prediction)
        probability = prediction_proba[0][prediction[0]]
        
        # Display results
        st.header("🎯 Prediction Results")
        
        result_col, prob_col = st.columns(2)
        
        with result_col:
            if result[0] == 'Y':
                st.success(f"## ✅ Loan Approved!")
                st.balloons()
            else:
                st.error(f"## ❌ Loan Not Approved")
        
        with prob_col:
            confidence_level = "High" if probability > 0.7 else "Medium" if probability > 0.5 else "Low"
            st.metric(
                label="Confidence Level",
                value=f"{probability:.2%}",
                delta=confidence_level
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
        
        # Show feature importance if available
        if hasattr(model, 'feature_importances_'):
            st.subheader("📈 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': encoded_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(feature_importance, use_container_width=True)
        
        # Interpretation
        st.subheader("💡 Interpretation")
        if result[0] == 'Y':
            st.info(f"""
            **Factors that likely contributed to approval:**
            - Good credit history
            - {'Stable salaried employment' if self_employed_value == 'No' else 'Self-employed business'}
            - Favorable property area
            - Strong financial profile
            """)
        else:
            st.info(f"""
            **Factors that may need improvement:**
            - Consider improving credit history
            - {'Provide additional employment documentation' if self_employed_value == 'No' else 'Provide business financial statements'}
            - Adjust loan amount relative to income
            - Increase income stability
            """)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("""
        **Common Issues:**
        - Input values don't match training data encodings
        - Column order mismatch
        - Data type inconsistencies
        """)
        
        # Additional debug info
        with st.expander("🔧 Technical Debug Info"):
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
            st.write("**Input DataFrame:**")
            st.write(input_df)
            st.write("**Available encoders:**")
            for col, encoder in label_encoders.items():
                st.write(f"{col}: {list(encoder.classes_)}")

# Add help section in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 How to Use:")
    st.markdown("1. Fill all applicant details")
    st.markdown("2. Select employment type")
    st.markdown("3. Enter financial information") 
    st.markdown("4. Click 'Predict Loan Approval'")
    st.markdown("5. Review results and debug info if needed")
    
    st.markdown("### 📊 Model Info:")
    st.markdown("- **Algorithm**: Random Forest")
    st.markdown("- **Features**: 11 variables")
    st.markdown("- **Accuracy**: 84.55%")

  
