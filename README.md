# 🏦 Loan Approval Prediction App

A comprehensive machine learning project that predicts loan approval decisions using MySQL, Python, and Streamlit.

## 📊 Project Overview

This end-to-end data science project demonstrates:
- **Data Extraction**: Loading data from MySQL database
- **Exploratory Data Analysis**: Comprehensive data analysis and visualization
- **Machine Learning**: Random Forest model training and evaluation
- **Web Application**: Interactive Streamlit app for predictions

## 🚀 Features

- **Database Integration**: MySQL for data storage and retrieval
- **Data Analysis**: Comprehensive EDA with visualizations
- **ML Model**: Random Forest Classifier with 82%+ accuracy
- **Web Interface**: Streamlit-based prediction app
- **Real-time Predictions**: Instant loan approval results

## 📁 Project Structure
loan-approval-prediction/
│
├── 📁 data/ # Raw and processed data
│ ├── applicant_info.json # Applicant personal information
│ ├── financial_info.json # Financial details
│ ├── loan_info.json # Loan application data
│ │
├── 📁 notebooks/ # Jupyter notebooks
│ ├── MySQL_Data_Loading.ipynb # Database connection & data loading
│ ├── EDA_Loan_Prediction.ipynb # Exploratory Data Analysis
│ ├── Mini_Project_Loan_Prediction.ipynb # Complete project workflow
│
├── 📁 models/ # Trained ML models
│ ├── loan_approval.pkl # Main trained model
│ ├── loan_prediction.pkl # Alternative model
│ 
├── 📁 src/ # Source code
│ ├── app.py # Streamlit web application
│ 
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # Project documentation (this file)
├── 📄 .gitignore # Git ignore rules
└── 📄 LICENSE # MIT License


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- MySQL Server
- Jupyter Lab

### 1. Clone the Repository
bash
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction

### 2. Install Dependencies
bash
pip install -r requirements.txt
### 3. Database Setup
Configure your MySQL database and update connection details in the notebooks.
### 4. Run the Application
bash
streamlit run src/app.py

### 📊 Model Performance
### Accuracy: 82.29%

Algorithm: Random Forest Classifier

Training Data: 480 records

Key Features: Credit History, Income, Loan Amount, Property Area

### Classification Report
text
              precision    recall  f1-score   support

           N       0.82      0.50      0.62        28
           Y       0.82      0.96      0.88        68

    accuracy                           0.82        96
   macro avg       0.82      0.73      0.75        96
weighted avg       0.82      0.82      0.81        96

### 🎯 Usage
For Development:
Explore notebooks/ for data analysis and model development

Use src/train_model.py to retrain the model

Run src/app.py for the web interface

For End Users:
Launch the Streamlit app

Fill in applicant details

Enter financial information

Get instant loan approval prediction

### 📋 Notebooks Overview
MySQL_Data_Loading.ipynb: Database connection and data extraction

EDA_Loan_Prediction.ipynb: Exploratory data analysis and visualization

Mini_Project_Loan_Prediction.ipynb: Complete project workflow

### 🔧 Technical Stack
### Backend
Python: Primary programming language

MySQL: Database management

Scikit-learn: Machine learning

Pandas & NumPy: Data manipulation

### Frontend
Streamlit: Web application framework

Matplotlib & Seaborn: Data visualization

Machine Learning
Random Forest: Classification algorithm

Label Encoding: Categorical data processing

Standard Scaling: Feature normalization

### 📈 Project Workflow
Data Collection: Extract data from MySQL database

Data Preprocessing: Handle missing values, encode categorical variables

Exploratory Analysis: Understand data patterns and relationships

Model Training: Train and evaluate Random Forest classifier

Web Deployment: Create interactive prediction interface

Model Serialization: Save trained model for production use

### 👥 Contributors
Manjunath Nayak

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

