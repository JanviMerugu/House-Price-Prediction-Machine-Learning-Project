import streamlit as st
import pandas as pd
import numpy as np
from src.train_model import train_and_evaluate
from src.load_data import load_data
from src.preprocess import preprocess_data

# Page config
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.markdown("### 📋 Enter the house details below to estimate the price.")

# Load and preprocess data
df = load_data("data/data.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)
model, mse, r2 = train_and_evaluate(X_train, X_test, y_train, y_test)

# UI Inputs
city = st.text_input("🏙️ City Name", placeholder="Enter your city")

bedrooms = st.slider("🛏️ Number of Bedrooms", 0, 10, 3)
bathrooms = st.slider("🛁 Number of Bathrooms", 0.0, 5.0, 2.0, step=0.25)
sqft_living = st.number_input("📏 Living Area (sqft)", min_value=200, max_value=10000, value=2000)
sqft_lot = st.number_input("🌳 Lot Area (sqft)", min_value=500, max_value=50000, value=5000)
floors = st.slider("🏢 Number of Floors", 1, 4, 1)
waterfront = st.selectbox("🌊 Waterfront View (0 = No, 1 = Yes)", [0, 1])
view = st.slider("👁️ View Rating", 0, 4, 0)
condition = st.slider("🧱 Condition Rating", 1, 5, 3)
sqft_above = st.number_input("⬆️ Sqft Above Ground", min_value=200, max_value=10000, value=1800)
sqft_basement = st.number_input("⬇️ Sqft Basement", min_value=0, max_value=5000, value=200)
yr_built = st.number_input("🏗️ Year Built", min_value=1900, max_value=2025, value=1990)
yr_renovated = st.number_input("🔧 Year Renovated (0 = Never)", min_value=0, max_value=2025, value=0)

# Prediction
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                                waterfront, view, condition, sqft_above, sqft_basement,
                                yr_built, yr_renovated]],
                              columns=X_train.columns)
    
    prediction = model.predict(input_data)[0]
    price_inr = prediction * 83.0  # Convert USD to INR (approx rate)

    st.success(f"💰 Estimated House Price in {city if city else 'your city'}: **₹ {price_inr:,.2f}**")
    
    st.markdown("---")
    st.markdown(f"📈 **Model R² Score:** {r2:.2f}")
    st.markdown(f"📉 **MSE (INR):** ₹ {mse * 83**2:,.2f}")
