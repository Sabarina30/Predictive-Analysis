import pandas as pd
import numpy as np
import streamlit as st
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv("Final_data_for_model.csv")

# Perform data preprocessing and split into X, y_class, and y_reg
df.drop(['Unnamed: 0', "LoanNumber", 'PercentFunded', 'Recommendations', "ScorexChangeAtTimeOfListing",
         'InvestmentFromFriendsCount', 'InvestmentFromFriendsAmount', 'TotalTrades', "EmploymentStatusDuration",
         'TradesNeverDelinquent (percentage)', 'TradesOpenedLast6Months', "EmploymentStatusDuration",
         "CurrentlyInGroup",
         "InquiriesLast6Months", "DelinquenciesLast7Years", "ProsperPaymentsLessThanOneMonthLate",
         "LoanOriginationQuarter", 'Investors'], axis=1, inplace=True)

X = df.drop(columns=['LoanStatus', 'ELA', 'EMI', 'PROI'], axis=1)
y_class = df['LoanStatus']
y_reg = df[['ELA', 'EMI', 'PROI']]

# Split data into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42
)
# Standardize your features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# Define a custom pipeline to combine classification and regression
class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg

    def fit(self, X, y_class, y_reg):
        self.clf.fit(X_train_std, y_class_train)
        X_filtered = X_train_std[y_class_train == 1]
        y_reg_filtered = y_reg_train[y_class_train == 1]
        self.reg.fit(X_filtered, y_reg_filtered)
        return self

    def predict(self, X):
        y_class_pred = self.clf.predict(X_test_std)
        X_filtered = X_test_std[y_class_pred == 1]
        y_reg_pred = self.reg.predict(X_filtered)
        return y_class_pred, y_reg_pred


# Create an instance of your custom pipeline
pipeline = CustomPipeline(clf=LogisticRegression(random_state=42), reg=Ridge())

# Fit the pipeline
pipeline.fit(X_train_std, y_class_train, y_reg_train)

# Save the combined pipeline using joblib
joblib.dump(pipeline, 'combined_pipeline.pkl')


# Function to load the combined pipeline
def load_combined_pipeline(filename):
    return joblib.load(filename)


# Function to make predictions
def make_predictions(model, input_data):
    y_class_pred, y_reg_pred = model.predict(input_data)

    # Extract specific predictions
    loan_status_pred = int(y_class_pred[0])  # Assuming y_class_pred is binary (0 or 1)
    emi_pred = y_reg_pred[0, 0]
    ela_pred = y_reg_pred[0, 1]
    proi_pred = y_reg_pred[0, 2]

    return loan_status_pred, emi_pred, ela_pred, proi_pred


# Main function to run Streamlit app
def main():
    st.title('Financial Risk Analysis')

    # Load the combined pipeline
    model = load_combined_pipeline('combined_pipeline.pkl')

    # Sidebar for feature selection
    st.sidebar.title('Select Input Features')

    # List of all available features (assuming 60 features)
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
                    'LP_NonPrincipalRecoverypayments', 'EMI', 'ELA', 'InterestAmount',
                    'TotalAmount', 'ROI']

    # User selection for input features
    selected_features = st.sidebar.multiselect('Select features', all_features,
                                               default=['EmploymentStatus','IncomeRange'])

    # Placeholder for input data
    input_data = []

    # Collect user input for selected features using sliders
    for feature in selected_features:
        value = st.sidebar.slider(f'Select value for {feature}', min_value=-0.00001, max_value=2.00000, value=0.000001,
                                  step=0.00001)
        input_data.append(value)

    # Convert input_data to numpy array
    input_data = np.array(input_data).reshape(1, -1)  # Reshape to match expected input shape

    # Make predictions
    if st.sidebar.button('Predict'):
        loan_status_pred, emi_pred, ela_pred, proi_pred = make_predictions(model, input_data)

        # Display predictions for selected features
        st.subheader('Predictions:')
        st.write(f"Loan Status: {'Accepted' if loan_status_pred == 1 else 'Risky'}")
        st.write(f"EMI Prediction: {emi_pred:.2f}")
        st.write(f"ELA Prediction: {ela_pred:.2f}")
        st.write(f"PROI Prediction: {proi_pred:.2f}")


if __name__ == '__main__':
    main()
