import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="AQI Dashboard", layout="wide")

# ------------------------
# GRADIENT BACKGROUND
# ------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# LOAD MODEL
# ------------------------
model = joblib.load("aqi_model.pkl")

st.title("🌍 Air Quality Prediction Dashboard")

# ------------------------
# INPUT SECTION
# ------------------------
col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0, value=20.0)

with col2:
    pm10 = st.number_input("PM10", min_value=0.0, value=40.0)

with col3:
    temp = st.number_input("Temperature (°C)", value=25.0)

# ------------------------
# PREDICTION
# ------------------------
if st.button("Predict AQI"):

    input_data = pd.DataFrame([[pm25, pm10, temp]],
                              columns=["pm2.5", "pm10", "temperature"])

    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    # AQI CATEGORY
    if prediction <= 50:
        category = "Good"
        color = "green"
        comment = "Air quality is satisfactory. Enjoy your day!"
    elif prediction <= 100:
        category = "Moderate"
        color = "yellow"
        comment = "Air quality is acceptable. Sensitive groups should be cautious."
    else:
        category = "Unhealthy"
        color = "red"
        comment = "Air quality is unhealthy. Avoid outdoor activities."

    # ------------------------
    # LAYOUT
    # ------------------------
    left, center, right = st.columns([1,2,1])

    # ------------------------
    # GAUGE (Animated)
    # ------------------------
    with center:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "AQI Gauge", 'font': {'size': 26}},
            number={'font': {'size': 45}},
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

        fig.update_layout(
            margin=dict(t=80, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------
    # COMMENT BOX (MODERN CARD)
    # ------------------------
    st.markdown(f"""
    <div style="
        background-color:white;
        padding:25px;
        border-radius:15px;
        text-align:center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        margin-top:20px;">
        <h2 style="color:{color};">{category}</h2>
        <p class="big-font">{comment}</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------
    # AQI HISTORY CHART
    # ------------------------
    st.subheader("📈 AQI History")

    # Fake history data (for demo)
    history_days = 7
    history_values = np.random.normal(prediction, 10, history_days)

    history_df = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, history_days+1)],
        "AQI": history_values
    })

    history_fig = px.line(
        history_df,
        x="Day",
        y="AQI",
        markers=True,
        title="Last 7 Days AQI Trend"
    )

    history_fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0.8)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(history_fig, use_container_width=True)