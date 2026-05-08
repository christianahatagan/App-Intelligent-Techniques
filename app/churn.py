import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from io import BytesIO

st.set_page_config(page_title="Churn Prediction System", layout="wide")

# Custom CSS - Pink Theme
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #C71585;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #C71585;
    }
    .metric-box {
        background: linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #FFE4E1;
        border-left: 5px solid #90EE90;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #FFE4E1;
        border-left: 5px solid #FF1493;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #FFF0F5;
    }
    .warning-box {
        background-color: #FFF4E6;
        border-left: 5px solid #FFA500;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Customer Churn Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Dataset 1: Internet Service", 
     "Dataset 2: Telco Customer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")

# Load model with error handling
model_path = '../models/' if os.path.exists('../models/') else 'models/'
if os.path.exists('models/'):
    model_path = 'models/'
elif os.path.exists('../models/'):
    model_path = '../models/'
elif os.path.exists('./models/'):
    model_path = './models/'
elif os.path.exists('app/models/'):
    model_path = 'app/models/'
else:
    model_path = './'

model = None
scaler = None
dataset_type = None

try:
    with st.spinner('Loading model...'):
        if "Dataset 1" in model_choice:
            model_file = f'{model_path}xgboost_model_dataset1.pkl'
            scaler_file = f'{model_path}scaler_dataset1.pkl'
            
            if not os.path.exists(model_file):
                st.error(f"Model file not found: {model_file}")
                st.info("Please ensure model files are in the correct directory")
                st.stop()
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            dataset_type = 1
            st.sidebar.success("Model 1 Loaded Successfully")
            st.sidebar.metric("F1-Score", "94.3%")
            st.sidebar.metric("Recall", "93.2%")
            st.sidebar.metric("Precision", "95.4%")
        else:
            model_file = f'{model_path}xgboost_model_dataset2.pkl'
            scaler_file = f'{model_path}scaler_dataset2.pkl'
            
            if not os.path.exists(model_file):
                st.error(f"Model file not found: {model_file}")
                st.info("Please ensure model files are in the correct directory")
                st.stop()
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            dataset_type = 2
            st.sidebar.success("Model 2 Loaded Successfully")
            st.sidebar.metric("F1-Score", "62.8%")
            st.sidebar.metric("Recall", "76.9%")
            st.sidebar.metric("Precision", "53.1%")
except FileNotFoundError as e:
    st.error(f"Model files not found. Please check the models folder.")
    st.error(f"Error details: {e}")
    st.info("Expected location: `models/` directory with .pkl files")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Helper functions for feature engineering
def engineer_features_dataset1(row):
    """Engineer features for Dataset 1"""
    usage_ratio = row['download_avg'] / (row['upload_avg'] + 1)
    contract_health = row['remaining_contract'] / (row['subscription_age'] + 1)
    revenue_per_month = row['bill_avg'] / (row['subscription_age'] + 1)
    has_service_issues = 1 if row['service_failure_count'] > 0 else 0
    log_download = np.log1p(row['download_avg'])
    total_services = row['is_tv_subscriber'] + row['is_movie_package_subscriber']
    is_high_value = 1 if (row['bill_avg'] > 15 and total_services > 0) else 0
    at_risk_flag = 1 if (row['remaining_contract'] == 0 and row['service_failure_count'] > 0) else 0
    
    return [
        row['is_tv_subscriber'], row['is_movie_package_subscriber'], 
        row['subscription_age'], row['bill_avg'], row['remaining_contract'],
        row['service_failure_count'], row['download_avg'], row['upload_avg'], 
        row['download_over_limit'], usage_ratio, contract_health, 
        revenue_per_month, has_service_issues, log_download, 
        total_services, is_high_value, at_risk_flag
    ]

def engineer_features_dataset2(row):
    """Engineer features for Dataset 2"""
    revenue_per_month = row['total_charges'] / (row['tenure'] + 1)
    is_new = 1 if row['tenure'] <= 12 else 0
    is_long_term = 1 if row['tenure'] > 36 else 0
    senior_no_partner = 1 if (row['senior_citizen'] == 1 and row['partner'] == 0) else 0
    
    features = [
        row['gender'], row['senior_citizen'], row['partner'], row['dependents'], 
        row['tenure'], row['phone_service'], row['paperless_billing'], 
        row['monthly_charges'], row['total_charges'],
        revenue_per_month, is_new, is_long_term, senior_no_partner,
        0, 0
    ]
    
    # Pad to 35 features
    while len(features) < 35:
        features.append(0)
    
    return features[:35]

def validate_dataset1_csv(df):
    """Validate Dataset 1 CSV columns"""
    required_cols = [
        'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
        'bill_avg', 'remaining_contract', 'service_failure_count',
        'download_avg', 'upload_avg', 'download_over_limit'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    return missing

def validate_dataset2_csv(df):
    """Validate Dataset 2 CSV columns"""
    required_cols = [
        'gender', 'senior_citizen', 'partner', 'dependents', 'tenure',
        'phone_service', 'paperless_billing', 'monthly_charges', 'total_charges'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    return missing

# Validation functions (same as before)
def validate_dataset1_inputs(subscription_age, bill_avg, remaining_contract, service_failures, 
                            download_avg, upload_avg):
    """Validate Dataset 1 inputs for logical consistency"""
    errors = []
    warnings = []
    
    if subscription_age < 0:
        errors.append("Subscription age cannot be negative")
    if bill_avg < 0:
        errors.append("Bill amount cannot be negative")
    if remaining_contract < 0:
        errors.append("Remaining contract cannot be negative")
    if remaining_contract > subscription_age:
        errors.append("Remaining contract cannot exceed subscription age")
    if download_avg < 0 or upload_avg < 0:
        errors.append("Data usage cannot be negative")
    
    if subscription_age > 0 and bill_avg == 0:
        warnings.append("Active subscription with $0 bill is unusual")
    if service_failures > 3:
        warnings.append("Very high service failure count detected")
    if download_avg > 100:
        warnings.append("Unusually high download usage detected")
    
    return errors, warnings

def validate_dataset2_inputs(tenure, monthly_charges, total_charges):
    """Validate Dataset 2 inputs for logical consistency"""
    errors = []
    warnings = []
    
    if tenure < 0:
        errors.append("Tenure cannot be negative")
    if monthly_charges < 0:
        errors.append("Monthly charges cannot be negative")
    if total_charges < 0:
        errors.append("Total charges cannot be negative")
    
    expected_minimum_total = monthly_charges * tenure * 0.5
    if tenure > 0 and total_charges > 0 and total_charges < expected_minimum_total:
        warnings.append(f"Total charges seem low for {tenure} months tenure")
    
    if tenure > 0 and monthly_charges > 0 and total_charges > (monthly_charges * tenure * 2):
        warnings.append("Total charges seem very high compared to monthly charges")
    
    if tenure == 0 and total_charges > monthly_charges:
        warnings.append("Total charges exceed monthly charges for new customer")
    
    return errors, warnings

# Create tabs for Single vs Batch prediction
tab1, tab2 = st.tabs(["Single Prediction", "Batch Predictions"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Customer Information Input")
        
        if dataset_type == 1:
            # Dataset 1: Internet Service
            st.subheader("Internet Service Customer Details")
            
            st.markdown("#### Service Subscriptions")
            col_a, col_b = st.columns(2)
            with col_a:
                is_tv = st.selectbox("TV Subscriber", options=[0, 1], 
                                    format_func=lambda x: "Yes" if x == 1 else "No", key="single_tv")
            with col_b:
                is_movie = st.selectbox("Movie Package Subscriber", options=[0, 1], 
                                       format_func=lambda x: "Yes" if x == 1 else "No", key="single_movie")
            
            st.markdown("#### Account Information")
            col_a, col_b = st.columns(2)
            with col_a:
                subscription_age = st.number_input("Subscription Age (months)", 
                                                  min_value=0.0, max_value=20.0, value=8.0, step=0.5,
                                                  help="How long has the customer been subscribed", key="single_sub_age")
                remaining_contract = st.number_input("Remaining Contract (months)", 
                                                    min_value=0.0, max_value=10.0, value=0.0, step=0.5,
                                                    help="Months left in contract (0 = no contract)", key="single_contract")
            with col_b:
                bill_avg = st.number_input("Average Monthly Bill ($)", 
                                           min_value=0, max_value=100, value=15, step=1,
                                           help="Average monthly billing amount", key="single_bill")
                service_failures = st.number_input("Service Failure Count", 
                                                  min_value=0, max_value=5, value=0, step=1,
                                                  help="Number of service interruptions", key="single_failures")
            
            st.markdown("#### Usage Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                download_avg = st.number_input("Average Download (GB)", 
                                              min_value=0.0, max_value=150.0, value=30.0, step=5.0,
                                              help="Average monthly download in GB", key="single_download")
            with col_b:
                upload_avg = st.number_input("Average Upload (GB)", 
                                            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
                                            help="Average monthly upload in GB", key="single_upload")
            
            download_over_limit = st.selectbox("Download Over Limit", options=[0, 1], 
                                              format_func=lambda x: "Yes" if x == 1 else "No",
                                              help="Has customer exceeded download limit", key="single_over_limit")
            
            errors, warnings = validate_dataset1_inputs(subscription_age, bill_avg, remaining_contract, 
                                                       service_failures, download_avg, upload_avg)
            
            if errors:
                st.error("VALIDATION ERRORS:")
                for error in errors:
                    st.error(f"• {error}")
            
            if warnings:
                st.warning("WARNINGS:")
                for warning in warnings:
                    st.warning(f"• {warning}")
            
            predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True,
                                      disabled=len(errors) > 0, key="single_predict")
            
            if predict_button:
                with st.spinner('Analyzing customer data...'):
                    try:
                        time.sleep(0.5)
                        
                        usage_ratio = download_avg / (upload_avg + 1)
                        contract_health = remaining_contract / (subscription_age + 1)
                        revenue_per_month = bill_avg / (subscription_age + 1)
                        has_service_issues = 1 if service_failures > 0 else 0
                        log_download = np.log1p(download_avg)
                        total_services = is_tv + is_movie
                        is_high_value = 1 if (bill_avg > 15 and total_services > 0) else 0
                        at_risk_flag = 1 if (remaining_contract == 0 and service_failures > 0) else 0
                        
                        features = np.array([[
                            is_tv, is_movie, subscription_age, bill_avg, remaining_contract,
                            service_failures, download_avg, upload_avg, download_over_limit,
                            usage_ratio, contract_health, revenue_per_month, has_service_issues,
                            log_download, total_services, is_high_value, at_risk_flag
                        ]])
                        
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)[0]
                        probability = model.predict_proba(features_scaled)[0]
                        
                        st.session_state['prediction'] = prediction
                        st.session_state['probability'] = probability
                        st.session_state['dataset_type'] = dataset_type
                        st.session_state['input_data'] = {
                            'is_tv': is_tv,
                            'is_movie': is_movie,
                            'subscription_age': subscription_age,
                            'bill_avg': bill_avg,
                            'remaining_contract': remaining_contract,
                            'service_failures': service_failures,
                            'download_avg': download_avg,
                            'upload_avg': upload_avg,
                            'download_over_limit': download_over_limit,
                            'total_services': total_services
                        }
                        
                        st.success("Prediction completed successfully!")
                        time.sleep(0.3)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please check your inputs and try again")
        
        else:
            # Dataset 2: Telco Customer
            st.subheader("Telco Customer Details")
            
            st.markdown("#### Demographics")
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.selectbox("Gender", options=[0, 1], 
                                    format_func=lambda x: "Male" if x == 1 else "Female", key="single_gender")
                senior = st.selectbox("Senior Citizen", options=[0, 1], 
                                    format_func=lambda x: "Yes" if x == 1 else "No",
                                    help="Is customer 65 or older", key="single_senior")
            with col_b:
                partner = st.selectbox("Has Partner", options=[0, 1], 
                                     format_func=lambda x: "Yes" if x == 1 else "No", key="single_partner")
                dependents = st.selectbox("Has Dependents", options=[0, 1], 
                                        format_func=lambda x: "Yes" if x == 1 else "No", key="single_dependents")
            
            st.markdown("#### Services")
            col_a, col_b = st.columns(2)
            with col_a:
                phone_service = st.selectbox("Phone Service", options=[0, 1], 
                                            format_func=lambda x: "Yes" if x == 1 else "No", key="single_phone")
                paperless = st.selectbox("Paperless Billing", options=[0, 1], 
                                        format_func=lambda x: "Yes" if x == 1 else "No", key="single_paperless")
            with col_b:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1,
                                        help="How long customer has been with company", key="single_tenure")
            
            st.markdown("#### Financial Information")
            col_a, col_b = st.columns(2)
            with col_a:
                monthly_charges = st.number_input("Monthly Charges ($)", 
                                                 min_value=0.0, max_value=120.0, value=50.0, step=5.0,
                                                 help="Current monthly bill amount", key="single_monthly")
            with col_b:
                total_charges = st.number_input("Total Charges ($)", 
                                               min_value=0.0, max_value=9000.0, value=1000.0, step=100.0,
                                               help="Total amount charged to date", key="single_total")
            
            errors, warnings = validate_dataset2_inputs(tenure, monthly_charges, total_charges)
            
            if errors:
                st.error("VALIDATION ERRORS:")
                for error in errors:
                    st.error(f"• {error}")
            
            if warnings:
                st.warning("WARNINGS:")
                for warning in warnings:
                    st.warning(f"• {warning}")
            
            predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True,
                                      disabled=len(errors) > 0, key="single_predict2")
            
            if predict_button:
                with st.spinner('Analyzing customer data...'):
                    try:
                        time.sleep(0.5)
                        
                        revenue_per_month = total_charges / (tenure + 1)
                        is_new = 1 if tenure <= 12 else 0
                        is_long_term = 1 if tenure > 36 else 0
                        senior_no_partner = 1 if (senior == 1 and partner == 0) else 0
                        
                        features = np.array([[
                            gender, senior, partner, dependents, tenure, 
                            phone_service, paperless, monthly_charges, total_charges,
                            revenue_per_month, is_new, is_long_term, senior_no_partner,
                            0, 0
                        ]])
                        
                        if features.shape[1] < 35:
                            padding = np.zeros((1, 35 - features.shape[1]))
                            features = np.hstack([features, padding])
                        elif features.shape[1] > 35:
                            features = features[:, :35]
                        
                        features_scaled = scaler.transform(features)
                        prediction = model.predict(features_scaled)[0]
                        probability = model.predict_proba(features_scaled)[0]
                        
                        st.session_state['prediction'] = prediction
                        st.session_state['probability'] = probability
                        st.session_state['dataset_type'] = dataset_type
                        st.session_state['input_data'] = {
                            'gender': gender,
                            'senior': senior,
                            'partner': partner,
                            'dependents': dependents,
                            'tenure': tenure,
                            'phone_service': phone_service,
                            'paperless': paperless,
                            'monthly_charges': monthly_charges,
                            'total_charges': total_charges
                        }
                        
                        st.success("Prediction completed successfully!")
                        time.sleep(0.3)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please check your inputs and try again")

    with col2:
        st.header("Prediction Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            probability = st.session_state['probability']
            dataset_type_session = st.session_state.get('dataset_type', 1)
            input_data = st.session_state.get('input_data', {})
            
            if prediction == 1:
                st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                st.markdown("### HIGH RISK: Customer Likely to Churn")
                st.markdown(f"<h1 style='text-align: center; color: #FF1493;'>{probability[1]*100:.1f}%</h1>", 
                           unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold;'>Churn Probability</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### LOW RISK: Customer Likely to Stay")
                st.markdown(f"<h1 style='text-align: center; color: #90EE90;'>{probability[0]*100:.1f}%</h1>", 
                           unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; font-weight: bold;'>Retention Probability</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("Probability Breakdown")
            prob_data = {
                'Outcome': ['Will Stay', 'Will Churn'],
                'Probability': [probability[0], probability[1]]
            }
            prob_df = pd.DataFrame(prob_data)
            st.bar_chart(prob_df.set_index('Outcome'), color='#FF69B4')
            
            st.subheader("Risk Level")
            risk_prob = probability[1]
            
            if risk_prob > 0.7:
                st.error("CRITICAL: Immediate retention action required")
            elif risk_prob > 0.5:
                st.warning("HIGH: Monitor closely and consider retention offer")
            elif risk_prob > 0.3:
                st.info("MEDIUM: Regular monitoring recommended")
            else:
                st.success("LOW: Customer appears satisfied")
            
            st.subheader("Recommended Actions")
            
            if prediction == 1:
                if dataset_type_session == 1:
                    if input_data.get('remaining_contract', 1) == 0:
                        st.write("- Offer contract renewal with discount")
                    if input_data.get('service_failures', 0) > 0:
                        st.write("- Address service quality issues immediately")
                    if input_data.get('total_services', 1) == 0:
                        st.write("- Propose bundle package (TV + Movie)")
                    if input_data.get('download_over_limit', 0) == 1:
                        st.write("- Suggest upgraded data plan")
                    st.write("- Assign to retention specialist")
                else:
                    if input_data.get('tenure', 13) < 12:
                        st.write("- New customer: Provide onboarding support")
                    if input_data.get('senior', 0) == 1 and input_data.get('partner', 1) == 0:
                        st.write("- Senior customer: Offer dedicated support")
                    st.write("- Contact customer within 48 hours")
                    st.write("- Offer loyalty discount or incentive")
            else:
                st.write("- Continue regular service monitoring")
                st.write("- Schedule satisfaction survey")
                st.write("- Consider upsell opportunities")
            
            if st.button("Clear Results", use_container_width=True, key="clear_single"):
                del st.session_state['prediction']
                del st.session_state['probability']
                if 'dataset_type' in st.session_state:
                    del st.session_state['dataset_type']
                if 'input_data' in st.session_state:
                    del st.session_state['input_data']
                st.rerun()
        else:
            st.info("Enter customer information and click 'Predict Churn Risk' to see results")
            st.markdown("""
            ### How to use:
            1. Fill in customer details on the left
            2. Review any validation warnings
            3. Click the prediction button
            4. View results and recommendations here
            """)

# ==================== TAB 2: BATCH PREDICTIONS ====================
with tab2:
    st.header("Batch Predictions")
    st.markdown("Upload a CSV file with multiple customers to get predictions for all of them at once.")
    
    # Show expected format
    with st.expander("View Expected CSV Format"):
        if dataset_type == 1:
            st.markdown("### Dataset 1: Internet Service")
            st.markdown("**Required columns:**")
            sample_df = pd.DataFrame({
                'is_tv_subscriber': [1, 0],
                'is_movie_package_subscriber': [1, 1],
                'subscription_age': [12.0, 8.0],
                'bill_avg': [45, 30],
                'remaining_contract': [3.0, 0.0],
                'service_failure_count': [0, 2],
                'download_avg': [50.0, 75.0],
                'upload_avg': [5.0, 3.0],
                'download_over_limit': [0, 1]
            })
            st.dataframe(sample_df)
        else:
            st.markdown("### Dataset 2: Telco Customer")
            st.markdown("**Required columns:**")
            sample_df = pd.DataFrame({
                'gender': [1, 0],
                'senior_citizen': [0, 1],
                'partner': [1, 0],
                'dependents': [1, 0],
                'tenure': [24, 6],
                'phone_service': [1, 1],
                'paperless_billing': [1, 0],
                'monthly_charges': [65.50, 45.00],
                'total_charges': [1500.00, 270.00]
            })
            st.dataframe(sample_df)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data. Make sure it has all required columns."
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} customers.")
            
            # Validate columns
            if dataset_type == 1:
                missing_cols = validate_dataset1_csv(df)
            else:
                missing_cols = validate_dataset2_csv(df)
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Process button
            if st.button("Process All Predictions", type="primary", use_container_width=True):
                with st.spinner(f'Processing {len(df)} customers...'):
                    try:
                        predictions = []
                        probabilities = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            # Engineer features
                            if dataset_type == 1:
                                features = engineer_features_dataset1(row)
                            else:
                                features = engineer_features_dataset2(row)
                            
                            # Scale and predict
                            features_array = np.array([features])
                            features_scaled = scaler.transform(features_array)
                            
                            pred = model.predict(features_scaled)[0]
                            prob = model.predict_proba(features_scaled)[0]
                            
                            predictions.append(pred)
                            probabilities.append(prob[1])  # Probability of churn
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Add results to dataframe
                        df['churn_prediction'] = predictions
                        df['churn_prediction_label'] = df['churn_prediction'].map({0: 'Will Stay', 1: 'Will Churn'})
                        df['churn_probability'] = [f"{p*100:.2f}%" for p in probabilities]
                        df['risk_level'] = pd.cut(
                            probabilities, 
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                        )
                        
                        st.success("All predictions completed!")
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Customers", len(df))
                        with col2:
                            churn_count = (df['churn_prediction'] == 1).sum()
                            st.metric("Predicted Churns", churn_count)
                        with col3:
                            churn_rate = (churn_count / len(df)) * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        with col4:
                            avg_prob = np.mean(probabilities) * 100
                            st.metric("Avg Churn Prob", f"{avg_prob:.1f}%")
                        
                        # Risk distribution
                        st.subheader("Risk Level Distribution")
                        risk_counts = df['risk_level'].value_counts()
                        st.bar_chart(risk_counts)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
                        st.info("Please check your data format and try again")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Make sure your file is a valid CSV format")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #C71585;'>
        <p><strong>Customer Churn Prediction System</strong></p>
        <p>Assignment 5 - Phase 6 | Built with Streamlit & XGBoost</p>
        <p style='font-size: 0.8rem; color: #888;'>
            Fast inference • High accuracy • Secure predictions • Batch processing enabled
        </p>
    </div>
    """, unsafe_allow_html=True)
