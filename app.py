import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Load trained model
model = joblib.load("aqi_model.pkl")

st.set_page_config(layout="wide")
st.title("🌍 Air Quality Prediction System")

# Layout with 3 columns
col1, col2, col3 = st.columns([1,2,1])

# -------------------------
# LEFT SIDE – USER INPUT
# -------------------------
with col1:
    st.subheader("🔬 Input Parameters")

    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 20.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 500.0, 30.0)
    temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)

    predict_btn = st.button("Predict AQI")


# -------------------------
# WHEN BUTTON IS CLICKED
# -------------------------
if predict_btn:

    # Create dataframe with correct feature names
    input_data = pd.DataFrame([[pm25, pm10, temp]],
                              columns=["pm2.5", "pm10", "temperature"])

    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    # Determine AQI Category + Color
    if prediction <= 50:
        category = "GOOD"
        color = "green"
        advice = "Air quality is good. Enjoy outdoor activities!"
    elif prediction <= 100:
        category = "MODERATE"
        color = "yellow"
        advice = "Air quality is moderate. Sensitive groups should be cautious."
    else:
        category = "UNHEALTHY"
        color = "red"
        advice = "Air quality is unhealthy. Avoid outdoor exposure."

    # -------------------------
    # CENTER – CIRCULAR GAUGE
    # -------------------------
    with col2:
        st.subheader("📊 AQI Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': category},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 200], 'color': "lightcoral"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # RIGHT – CIRCULAR COMMENT BOX
    # -------------------------
    with col3:
        st.subheader("💬 Advice")

        st.markdown(
            f"""
            <div style="
                border: 3px solid {color};
                border-radius: 50%;
                padding: 40px;
                text-align: center;
                height: 250px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: {color};
                font-size: 16px;">
                {advice}
            </div>
            """,
            unsafe_allow_html=True
        )