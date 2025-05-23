import streamlit as st
import pandas as pd
from joblib import load

st.title('Loan Prediction App')

model = load('best_rf_model.pkl')

categorical_features = {
    'person_gender': {'Male': 1, 'Female': 0},
    'person_education': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
    'person_home_ownership': {'Own': 0, 'Rent': 1, 'Mortgage': 2},
    'loan_intent': {'Personal': 0, 'Business': 1, 'Debt Consolidation': 2}
}

def predict_loan_status(features):
    # Susun urutan fitur sesuai training
    encoded_features = [
        features[0],  # person_age
        categorical_features['person_gender'][features[1]],
        categorical_features['person_education'][features[2]],
        features[3],  # person_income
        features[4],  # person_emp_exp
        categorical_features['person_home_ownership'][features[5]],
        features[6],  # loan_amnt
        categorical_features['loan_intent'][features[7]],
        features[8],  # loan_int_rate
        features[9],  # loan_percent_income
        features[10], # cb_person_cred_hist_length
        features[11], # credit_score
        features[12], # previous_loan_defaults_on_file
    ]

    # Pastikan kolom sesuai dengan urutan dan jumlah fitur saat training
    columns = [
        'person_age', 'person_gender', 'person_education', 'person_income', 'person_emp_exp',
        'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file'
    ]
    
    features_df = pd.DataFrame([encoded_features], columns=columns)
    prediction = model.predict(features_df)
    return prediction[0]

# Input form
person_age = st.number_input('Age of the Person', min_value=18, max_value=100, step=1)
person_gender = st.selectbox('Gender of the Person', ['Male', 'Female'])
person_education = st.selectbox('Education Level', ['High School', 'Bachelors', 'Masters', 'PhD'])
person_income = st.number_input('Annual Income', min_value=0, step=1000)
person_emp_exp = st.number_input('Years of Work Experience', min_value=0, step=1)
person_home_ownership = st.selectbox('Home Ownership Status', ['Own', 'Rent', 'Mortgage'])
loan_amnt = st.number_input('Loan Amount', min_value=0, step=500)
loan_intent = st.selectbox('Loan Intent', ['Personal', 'Business', 'Debt Consolidation'])
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, step=0.1)
loan_percent_income = st.number_input('Loan as Percentage of Income', min_value=0.0, max_value=100.0, step=0.1)
cb_person_cred_hist_length = st.number_input('Credit History Length (in years)', min_value=0, step=1)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', [0, 1])

input_features = [
    person_age,
    person_gender,
    person_education,
    person_income,
    person_emp_exp,
    person_home_ownership,
    loan_amnt,
    loan_intent,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    previous_loan_defaults_on_file
]

if st.button('Predict Loan Status'):
    prediction = predict_loan_status(input_features)
    
    if prediction == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Denied')
