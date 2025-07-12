import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Smart Energy Forecast", layout="wide")

# Load Clean Data and Model ---
df = pd.read_csv("clean_energy_data.csv", parse_dates=["datetime"], index_col="datetime")
model = load_model("lstm_energy_model.h5")

# Normalize Data ---
min_val = df['Global_active_power'].min()
max_val = df['Global_active_power'].max()
last_window = df['Global_active_power'].values[-60:]
scaled_window = (last_window - min_val) / (max_val - min_val)

# Forecast Function ---
def forecast_next_steps(model, last_seq, n_steps=1440):
    predictions = []
    seq = last_seq.copy()

    for _ in range(n_steps):
        pred = model.predict(seq.reshape(1, -1, 1), verbose=0)[0, 0]
        predictions.append(pred)
        seq = np.append(seq[1:], pred)

    return predictions

# Generate Forecast ---
forecast_scaled = forecast_next_steps(model, scaled_window, n_steps=1440)
forecast = np.array(forecast_scaled) * (max_val - min_val) + min_val

# Generate Future Timestamps ---
future_times = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=1), periods=1440, freq="T")
forecast_series = pd.Series(forecast, index=future_times)

# Optimal Suggestions ---
suggestions = forecast_series.nsmallest(5).index.tolist()

# Streamlit UI ---
st.title("🔋 Smart Energy Usage Forecasting")
st.markdown("""
This app predicts upcoming energy consumption using an LSTM model.
It also suggests optimal usage windows and alerts unusual usage spikes.
""")

with st.sidebar:
    st.header("⚙️ Settings")
    n_forecast = st.slider("Forecast Minutes", min_value=60, max_value=1440, value=1440, step=60)
    show_anomalies = st.checkbox("Show Anomaly Alerts", value=True)

# Forecast Chart ---
st.subheader(f"📈 Forecast for Next {n_forecast} Minutes")
st.line_chart(forecast_series[:n_forecast])

# Suggested Times ---
st.subheader("📉 Suggested Optimal Usage Times")
for time in suggestions:
    st.markdown(f"- 🕒 {time.strftime('%Y-%m-%d %H:%M')}")

# Anomaly Alert ---
if show_anomalies:
    latest_val = df['Global_active_power'].iloc[-1]
    threshold = df['Global_active_power'].mean() + 3 * df['Global_active_power'].std()
    if latest_val > threshold:
        st.error("🚨 Alert: Unusual energy consumption detected!")
    else:
        st.success("✅ Energy usage is within normal range.")

# Tabs ---

tab1, tab2 = st.tabs(["📊 Forecast", "🔍 Historical Patterns"])

with tab1:
    st.line_chart(forecast_series[:n_forecast])

with tab2:
    st.bar_chart(df.groupby(df.index.hour)['Global_active_power'].mean())

# Footer ---

st.markdown("---", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 14px;'>
    Made with ❤️ by <strong>Ishika</strong>. Trained on UCI dataset. Aligned with Microsoft Smart City vision.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 10px;'>
            
    <a href='mailto:ish21112002@gmail.com' target='_blank'>
        <img src='https://img.icons8.com/ios-filled/30/000000/new-post.png' style='margin-right:10px' title='Email Me'/>
    </a>
            
    <a href='https://github.com/skyish21' target='_blank'>
        <img src='https://img.icons8.com/ios-glyphs/30/000000/github.png' style='margin-right:10px' title='GitHub'/>
    </a>
            
    <a href='https://www.linkedin.com/in/ishika-sharma-79a67a326' target='_blank'>
        <img src='https://img.icons8.com/ios-filled/30/0077B5/linkedin.png' title='LinkedIn'/>
    </a>
</div>
""", unsafe_allow_html=True)
