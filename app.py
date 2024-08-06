import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

def make_predictions(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    # Ensure to interpret the prediction correctly
    return prediction[0]


def predict_additional_metrics(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    predictions = metric_model.predict(input_data_reshaped)  # Returns an array
    # Adjust based on the actual output structure
    emi, ela, proi = predictions[0]  # Unpack if it's a single array with multiple values
    return emi, ela, proi



st.title('FINANCIAL RISK ANALYSIS')

loaded_model = joblib.load('model_logi.pickle')
metric_model = joblib.load('multi_reg_1.pickle')

st.sidebar.title('Select Input Features')

all_features = ['LoanTenure', 'BorrowerAPR', 'BorrowerRate',
                'LenderYield', 'EstimatedEffectiveYield', 'EstimatedLoss',
                'EstimatedReturn', 'ProsperRating (numeric)', 'ProsperRating (Alpha)',
                'ProsperScore', 'ListingCategory (numeric)', 'BorrowerState',
                'Occupation', 'EmploymentStatus', 'IsBorrowerHomeowner',
                'DateCreditPulled', 'CreditScoreRangeLower', 'CreditScoreRangeUpper',
                'FirstRecordedCreditLine', 'CurrentCreditLines', 'OpenCreditLines',
                'TotalCreditLinespast7years', 'OpenRevolvingAccounts',
                'OpenRevolvingMonthlyPayment', 'TotalInquiries', 'CurrentDelinquencies',
                'AmountDelinquent', 'PublicRecordsLast10Years',
                'PublicRecordsLast12Months', 'RevolvingCreditBalance',
                'BankcardUtilization', 'AvailableBankcardCredit', 'DebtToIncomeRatio',
                'IncomeRange', 'IncomeVerifiable', 'StatedMonthlyIncome',
                'TotalProsperLoans', 'TotalProsperPaymentsBilled',
                'OnTimeProsperPayments', 'ProsperPaymentsOneMonthPlusLate',
                'ProsperPrincipalBorrowed', 'ProsperPrincipalOutstanding',
                'LoanCurrentDaysDelinquent', 'LoanFirstDefaultedCycleNumber',
                'LoanMonthsSinceOrigination', 'LoanOriginalAmount',
                'MonthlyLoanPayment', 'LP_CustomerPayments',
                'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees',
                'LP_CollectionFees', 'LP_GrossPrincipalLoss', 'LP_NetPrincipalLoss',
                'LP_NonPrincipalRecoverypayments', 'InterestAmount',
                'TotalAmount', 'PROI']
selected_features = st.sidebar.multiselect('Select features', all_features,
                                           default=['BorrowerAPR', 'BorrowerRate', 'EstimatedEffectiveYield',
                                                    'EstimatedReturn', 'LenderYield', 'LoanCurrentDaysDelinquent'
                                                    ])
input_dict = {feature: 0 for feature in all_features}

# Collect user input for selected features using sliders
for feature in selected_features:
    min_val, max_val = -1.0000, 2.0000  # Adjust these values based on your feature ranges
    value = st.sidebar.slider(f'Select value for {feature}', min_value=min_val, max_value=max_val, value=0.0000001,
                              step=0.00001)
    input_dict[feature] = value

input_data = np.array([input_dict[feature] for feature in all_features]).reshape(1, -1)

# Make predictions
if st.sidebar.button('Predict'):
    try:
        loan_status_pred = make_predictions(input_data)
        # Display predictions
        st.subheader('Predictions:')
        # Update the logic to handle the prediction result
        loan_status_text = 'Accepted' if loan_status_pred == 1 else 'Rejected'
        st.write(f"Loan Status: {loan_status_text}")

        if loan_status_pred == 1:
            emi, ela, roi = predict_additional_metrics(input_data)

            st.subheader('Predicted Values:')
            st.write(f"EMI: ₹{emi:,.2f}")
            st.write(f"ELA: ₹{ela:,.2f}")
            st.write(f"PROI: ₹{roi:,.2f}")
        else:
            st.write("The loan was rejected. No further predictions are available.")

    except Exception as e:
        st.error(f"Error making prediction: {e}")


