
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Customer Segmentation & Churn Assistant", page_icon="🛍️", layout="centered")

st.title("🛍️ Customer Segmentation & Churn Risk Assistant")
st.markdown("Input customer metrics below to dynamically calculate their RFM segment and evaluate churn risk using our trained Random Forest model.")

# Load original RFM dataset to fit ranges
try:
    rfm = pd.read_csv("customer_rfm_analysis.csv")
except Exception as e:
    st.warning("RFM analysis dataset not found. Using default logic.")

# User Inputs
st.sidebar.header("Customer Purchase Metrics")
recency = st.sidebar.slider("Recency (Days since last purchase)", 1, 375, 45)
frequency = st.sidebar.slider("Frequency (Number of Orders)", 1, 100, 5)
monetary = st.sidebar.number_input("Monetary Value (Total Spend in £)", min_value=1.0, value=1000.0, step=10.0)

# 1. Feature Calculations
aov = monetary / frequency if frequency > 0 else 0

# Approximate R, F, M scores
if recency <= 50: r_score = 4
elif recency <= 120: r_score = 3
elif recency <= 240: r_score = 2
else: r_score = 1

if frequency <= 1: f_score = 1
elif frequency <= 2: f_score = 2
elif frequency <= 5: f_score = 3
else: f_score = 4

if monetary <= 250: m_score = 1
elif monetary <= 500: m_score = 2
elif monetary <= 1200: m_score = 3
else: m_score = 4

rfm_score = r_score + f_score + m_score

# Determine Segment
if rfm_score >= 10:
    segment = "Champions 🏆"
    color = "green"
elif rfm_score >= 8:
    segment = "Loyal Customers ⭐"
    color = "blue"
elif rfm_score >= 6:
    segment = "Potential Customers 📈"
    color = "orange"
else:
    segment = "At Risk ⚠️"
    color = "red"

# Display calculated results
st.subheader("📊 Customer Segmentation Profile")
col1, col2, col3 = st.columns(3)
col1.metric("Calculated AOV", f"£{aov:.2f}")
col2.metric("RFM Score", f"{rfm_score} / 12")
col3.markdown(f"**Segment Group:**\n<h3 style='color:{color}; margin-top:0px;'>{segment}</h3>", unsafe_html=True)

# We will retrain a quick instance of the model within the app using the same hyperparameters
# to keep the app self-contained, or we can load the existing workspace data.
if 'rfm' in locals():
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']
    features_improved = ['Frequency', 'Monetary', 'AOV', 'R', 'F', 'M']
    X = rfm[features_improved]
    y = rfm['Churn']
    
    model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=8, class_weight='balanced')
    model.fit(X, y)
    
    input_features = pd.DataFrame([[frequency, monetary, aov, r_score, f_score, m_score]], columns=features_improved)
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]
    
    st.subheader("🧠 ML Churn Prediction")
    if prediction == 1:
        st.error(f"🔴 **High Risk of Churning!** (Inactivity > 90 days likely)")
    else:
        st.success(f"🟢 **Active Customer**")
    st.progress(float(probability))
    st.write(f"Churn Probability Index: **{probability * 100:.1f}%**")
