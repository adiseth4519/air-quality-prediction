import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="PM2.5 Predictor")

# Load model + feature schema
model = joblib.load("models/linear_regression.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

st.title("🌫️ Air Quality Prediction System")
st.write("Predict PM2.5 concentration using weather and time features.")

# User inputs
temp = st.number_input("Temperature (°C)", value=10.0)
pressure = st.number_input("Pressure (hPa)", value=1013.0)
humidity = st.number_input("Humidity (%)", value=50.0)
hour = st.slider("Hour of Day", 0, 23, 12)

if st.button("Predict PM2.5"):
    # --- Build full feature vector ---
    now = datetime.now()

    input_data = {
        "TEMP": temp,
        "PRES": pressure,
        "HUMI": humidity,
        "month_sin": np.sin(2 * np.pi * now.month / 12),
        "month_cos": np.cos(2 * np.pi * now.month / 12),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }

    # Create empty row with all features
    X = pd.DataFrame(columns=feature_cols)
    X.loc[0] = 0.0

    # Fill known features
    for k, v in input_data.items():
        if k in X.columns:
            X.loc[0, k] = v

    # Predict
    prediction = model.predict(X)[0]

    st.success(f"🌍 Predicted PM2.5: **{prediction:.2f} μg/m³**")
    st.caption("Missing features filled with baseline values (demo assumption).")
