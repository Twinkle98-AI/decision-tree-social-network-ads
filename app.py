import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Decision Tree Purchase Predictor",
    page_icon="🌳",
    layout="centered"
)

# Load model & scaler
model = pickle.load(open("dt_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.markdown(
    "<h1 style='text-align:center;'>🌳 Social Network Ads Prediction</h1>",
    unsafe_allow_html=True
)
st.write("Predict whether a user will purchase a product based on Age & Salary")
st.divider()

# Sidebar inputs
st.sidebar.header("🔢 User Inputs")

age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30)
salary = st.sidebar.number_input("Estimated Salary", min_value=15000, max_value=200000, value=50000)

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.success("✅ Customer WILL purchase the product")
    else:
        st.error("❌ Customer will NOT purchase the product")

    st.metric("Purchase Probability", f"{probability:.2%}")

st.divider()
st.caption("Built using Decision Tree (Entropy) | Scikit-learn | Streamlit")
