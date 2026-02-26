import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="AQI Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #2c5364);
        color: white;
    }
    h1 {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Models
# -------------------------
lr_model = joblib.load("aqi_model.pkl")          # Linear Regression
rf_model = joblib.load("random_forest.pkl")     # Random Forest
cnn_model = load_model("cnn_aqi_model.keras")   # CNN (NEW FORMAT)
cnn_scaler = joblib.load("cnn_scaler.pkl")      # CNN Scaler (IMPORTANT)

# -------------------------
# Store Prediction History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Title
# -------------------------
st.title("🌍 Air Quality And Pollution Spikes Prediction System")

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1, 2])

# -------------------------
# Inputs (LEFT SIDE)
# -------------------------
with col1:

    st.subheader("Enter Pollution Values")

    pm25 = st.number_input("PM2.5", min_value=0.0, value=10.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=20.0)
    temp = st.number_input("Temperature", value=25.0)

    model_choice = st.selectbox(
        "Select Prediction Model",
        ["Linear Regression", "Random Forest", "CNN"]
    )

    if st.button("Predict AQI"):

        input_df = pd.DataFrame({
            "pm2.5": [pm25],
            "pm10": [pm10],
            "temperature": [temp]
        })

        # -------------------------
        # Model Selection
        # -------------------------
        if model_choice == "Linear Regression":
            prediction = lr_model.predict(input_df)[0]

        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)[0]

        elif model_choice == "CNN":

            # SCALE FIRST
            scaled_input = cnn_scaler.transform(input_df)

            # RESHAPE FOR CNN
            scaled_input = scaled_input.reshape(1, 3, 1)

            prediction = cnn_model.predict(scaled_input)[0][0]

        prediction = round(float(prediction), 2)

        st.session_state.current_prediction = prediction
        st.session_state.history.append(prediction)

# -------------------------
# Results (RIGHT SIDE)
# -------------------------
with col2:

    if "current_prediction" in st.session_state:

        aqi = st.session_state.current_prediction

        if aqi <= 50:
            status = "Good"
            color = "green"
            comment = "Air quality is safe, Enjoy."
        elif aqi <= 100:
            status = "Slightly Polluted"
            color = "yellow"
            comment = "Air quality is Fair, sensitive groups take care."
        else:
            status = "Unhealthy"
            color = "red"
            comment = "Air quality is harmful, Avoid outdoor activities"

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi,
            title={'text': "AQI Gauge"},
            gauge={
                'axis': {'range': [0, 300]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 300], 'color': "red"},
                ],
            }
        ))

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Status Box
        st.markdown(f"""
        <div style="
            text-align:center;
            padding:20px;
            border-radius:25px;
            background-color:{color};
            font-weight:bold;
            font-size:20px;
            color:black;">
            {status}<br><br>{comment}
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# History Chart
# -------------------------
if len(st.session_state.history) > 0:
    st.subheader("Last 7 Predictions")

    history = st.session_state.history[-7:]

    history_df = pd.DataFrame({
        "Prediction": history
    })

    st.line_chart(history_df)