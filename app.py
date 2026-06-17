import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Global Card styling */
    .custom-card {
        background: rgba(30, 41, 59, 0.45);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Headings styling */
    .header-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #38bdf8;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    /* Result section metrics styling */
    .metric-value-high {
        font-size: 3.5rem;
        font-weight: 800;
        color: #f87171;
        text-shadow: 0 0 10px rgba(248, 113, 113, 0.4);
    }
    .metric-value-low {
        font-size: 3.5rem;
        font-weight: 800;
        color: #34d399;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.4);
    }
    
    /* Styled lists */
    .rec-item {
        background: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
    }
    
    /* Custom style for buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load pickled models
@st.cache_resource
def load_artifacts():
    if not os.path.exists("customer_churn_model.pkl") or not os.path.exists("encoders.pkl"):
        return None, None
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data, encoders
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model_data, encoders = load_artifacts()

# Title and Layout
st.markdown("<h1 class='header-text'>🔄 Telco Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("##### AI-Powered Retention Analytics & Churn Probability Estimator")
st.write("---")

if model_data is None or encoders is None:
    st.info("👋 Welcome! The machine learning model is currently training in the background. Once the notebook finishes executing, please refresh this page to access the predictor.")
    st.stop()

# Get model details
model = model_data["model"]
feature_names = model_data["features_names"]

# Dynamic input fields construction
st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Customer Information Form</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Tab/Section 1: Demographics
with col1:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", options=list(encoders['gender'].classes_))
    senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])
    partner = st.selectbox("Partner (Married/Cohabiting)", options=list(encoders['Partner'].classes_))
    dependents = st.selectbox("Dependents", options=list(encoders['Dependents'].classes_))

# Tab/Section 2: Services
with col2:
    st.markdown("#### 🛠️ Services Subscribed")
    phone_service = st.selectbox("Phone Service", options=list(encoders['PhoneService'].classes_))
    multiple_lines = st.selectbox("Multiple Lines", options=list(encoders['MultipleLines'].classes_))
    internet_service = st.selectbox("Internet Service Type", options=list(encoders['InternetService'].classes_))
    
    # Dependent services based on Internet Service
    if internet_service != "No":
        online_security = st.selectbox("Online Security Service", options=list(encoders['OnlineSecurity'].classes_))
        online_backup = st.selectbox("Online Backup Service", options=list(encoders['OnlineBackup'].classes_))
        device_protection = st.selectbox("Device Protection Service", options=list(encoders['DeviceProtection'].classes_))
        tech_support = st.selectbox("Tech Support Service", options=list(encoders['TechSupport'].classes_))
        streaming_tv = st.selectbox("Streaming TV Service", options=list(encoders['StreamingTV'].classes_))
        streaming_movies = st.selectbox("Streaming Movies Service", options=list(encoders['StreamingMovies'].classes_))
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

# Tab/Section 3: Account details
with col3:
    st.markdown("#### 💳 Account & Contract")
    contract = st.selectbox("Contract Type", options=list(encoders['Contract'].classes_))
    paperless_billing = st.selectbox("Paperless Billing", options=list(encoders['PaperlessBilling'].classes_))
    payment_method = st.selectbox("Payment Method", options=list(encoders['PaymentMethod'].classes_))
    tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    
    # Auto-calculate/suggest Total Charges
    suggested_total = round(tenure * monthly_charges, 2)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(suggested_total))

st.markdown("</div>", unsafe_allow_html=True)

# Predict trigger
st.write("")
predict_btn = st.button("RUN CHURN RISK PREDICTION", use_container_width=True)

if predict_btn:
    # Prepare input data dict
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Create DataFrame in the same columns order as the model expect
    input_df = pd.DataFrame([input_dict])
    
    # Apply encoders to categorical fields
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except Exception as e:
                # Handle unexpected labels gracefully
                st.error(f"Error encoding column '{col}': {e}")
                st.stop()
                
    # Ensure correct column ordering
    input_df = input_df[feature_names]
    
    # Make prediction
    pred_prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    
    # Display Results
    st.write("---")
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown("<div class='custom-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("### CHURN RISK PROFILE")
        
        prob_percentage = f"{pred_prob * 100:.1f}%"
        
        if pred_prob >= 0.5:
            st.markdown(f"<div class='metric-value-high'>{prob_percentage}</div>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#f87171;'>⚠️ HIGH CHURN RISK</h4>", unsafe_allow_html=True)
            st.write("This customer has a high likelihood of leaving the service. Immediate retention measures are recommended.")
        else:
            st.markdown(f"<div class='metric-value-low'>{prob_percentage}</div>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#34d399;'>✅ LOW CHURN RISK</h4>", unsafe_allow_html=True)
            st.write("This customer is currently stable with a low risk of churn. Maintain regular service standards.")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with res_col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Actionable Retention Recommendations")
        
        recs = []
        
        if contract == "Month-to-month":
            recs.append("🎯 **Upgrade Contract**: Pitch annual or 2-year contracts. Month-to-month contracts are the #1 predictor of churn. Offer a 10% loyalty discount for upgrading.")
            
        if tenure <= 12:
            recs.append("⏳ **Early Tenure Engagement**: The customer is in their critical first year. Schedule a proactive check-in call to address any initial setup or onboarding concerns.")
            
        if internet_service != "No" and (online_security == "No" or tech_support == "No"):
            recs.append("🛡️ **Service Bundling**: Offer a free 3-month trial of Online Security and Tech Support. Customers with these features demonstrate significantly higher retention rates.")
            
        if payment_method == "Electronic check":
            recs.append("💳 **Payment Method Transition**: Offer a one-time $5 statement credit to set up automatic payment (Credit Card or Bank Transfer). Electronic check users churn 3x more frequently.")
            
        if monthly_charges > 80.0:
            recs.append("💰 **Plan Optimization**: The monthly billing is high. Have account managers contact the customer to review their package and optimize it, preventing price-sensitivity churn.")
            
        if not recs:
            recs.append("👍 **Nurture Relationship**: The customer profile is healthy. Send personalized holiday or birthday greetings to keep brand affinity high.")
            
        for rec in recs:
            st.markdown(f"<div class='rec-item'>{rec}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

# Add sidebar details
st.sidebar.markdown("<h3 class='header-text'>About the Model</h3>", unsafe_allow_html=True)
st.sidebar.info("""
This prediction engine is powered by an optimized **Machine Learning Model** trained on the Telco Customer Churn dataset of 7,043 customers.

**Key Technical Details:**
- **Pre-processing:** Synthetic Minority Over-sampling Technique (SMOTE) was used to resolve dataset imbalance.
- **Model Tuning:** Hyperparameter optimization via Stratified K-Fold cross-validation.
- **Top Predictive Features:** Contract Type, Tenure, Payment Method, and Monthly Charges.
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Portfolio Showcase")
