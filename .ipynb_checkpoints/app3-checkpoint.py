import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

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
# Load Model
# -------------------------
model = joblib.load("aqi_model.pkl")

# -------------------------
# Store Prediction History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Title
# -------------------------
st.title("🌍 Air Quality Index Prediction System")

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

    if st.button("Predict AQI"):

        # IMPORTANT: Lowercase feature names (same as training)
        input_data = pd.DataFrame({
            "pm2.5": [pm25],
            "pm10": [pm10],
            "temperature": [temp]
        })

        prediction = model.predict(input_data)[0]
        prediction = round(float(prediction), 2)

        st.session_state.current_prediction = prediction
        st.session_state.history.append(prediction)

# -------------------------
# Results (RIGHT SIDE)
# -------------------------
with col2:

    if "current_prediction" in st.session_state:

        aqi = st.session_state.current_prediction

        # AQI Categories
        if aqi <= 50:
            status = "Good"
            color = "green"
            comment = "Air quality is satisfactory."
        elif aqi <= 100:
            status = "Moderate"
            color = "yellow"
            comment = "Air quality is acceptable."
        else:
            status = "Unhealthy"
            color = "red"
            comment = "Air quality is harmful."

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

    st.subheader("📊 Last 7 Predictions")

    history = st.session_state.history[-7:]

    history_df = pd.DataFrame({
        "Prediction": history
    })

    st.line_chart(history_df)