import streamlit as st
import joblib
import numpy as np

# Load trained components
model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")
# Page config
st.set_page_config(page_title="Energy Predictor", layout="centered")

# Title
st.title("🔋 Energy Consumption Prediction System")
st.markdown("### Predict building energy usage using ML")

st.write("Enter building & environmental details:")

# Inputs
square_feet = st.number_input("🏢 Square Feet", value=1000)
air_temperature = st.number_input("🌡 Air Temperature (°C)", value=25)
cloud_coverage = st.number_input("☁ Cloud Coverage", value=2)

hour = st.slider("⏰ Hour", 0, 23, 12)
day = st.slider("📅 Day", 1, 31, 15)
month = st.slider("📆 Month", 1, 12, 6)

# Predict button
if st.button("⚡ Predict Energy Consumption"):

    # Create input with SAME structure as training
    data = np.array([[square_feet, 0, 0,
                      air_temperature, cloud_coverage,
                      0, 0, hour, day, month]])

    # Apply same preprocessing
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)

    # Predict
    prediction = model.predict(data_pca)[0]

    # Fix negative values
    prediction = abs(prediction)

    # Output
    st.success(f"🔋 Estimated Energy Consumption: {prediction:.2f} kWh")
    st.info("This prediction is based on PCA + Regression model.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>🚀 Built by Srikanth K | VTU - CSE (ISE) | ML Project</p>",
    unsafe_allow_html=True
)
