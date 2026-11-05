"""
PowerPulse | Household Electricity Consumption Predictor

This app uses the model selected in the notebook (lowest RMSE among
KNN, Linear Regression, Decision Tree and Random Forest).
It predicts Global_active_power from the same input columns used in X.
"""
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PowerPulse | Electricity Predictor", page_icon="⚡", layout="wide")
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "electricity_consumption_model.pkl"

FEATURES = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "Hour", "Day", "Month", "DayOfWeek"
]

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background: radial-gradient(1000px 500px at 10% 0%, #163B55 0%, #0B1B2B 48%, #09151F 100%); color:#F4F7FA;}
header[data-testid="stHeader"] {display:none !important;}
.block-container {max-width:1100px; padding-top:1.5rem !important;}
.brand {font-size:2.6rem;font-weight:800;letter-spacing:.16em;color:white;}
.brand span {color:#2DD4BF;}
.subtitle {color:#AFC2D3;margin-top:-.25rem;margin-bottom:1rem;}
.result {background:linear-gradient(120deg,#0D9488,#102A43);border:1px solid #2DD4BF;border-radius:18px;padding:1.5rem;margin:1rem 0;}
.result-label {color:#D6F5F0;font-size:.95rem;}
.result-value {color:white;font-size:3rem;font-weight:800;}
[data-testid="stMetric"] {background:rgba(16,42,67,.55);border:1px solid #28506B;border-radius:14px;padding:.8rem 1rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} not found. Run the notebook's model-saving cell "
            "and place the downloaded file in the same folder as app.py."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.markdown('<div class="brand">POWER<span>PULSE</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Household Electricity Consumption Prediction</div>', unsafe_allow_html=True)
st.write("Enter the household power measurements and time details below to estimate **Global Active Power (kW)**.")
st.info("This is a regression prediction from entered measurements")

try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Enter input values")
with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        reactive = st.number_input("Global Reactive Power (kW)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
        voltage = st.number_input("Voltage (V)", min_value=0.0, value=240.0, step=1.0, format="%.2f")
        intensity = st.number_input("Global Intensity (A)", min_value=0.0, value=5.0, step=0.1, format="%.3f")
        sub1 = st.number_input("Sub Metering 1 (Wh)", min_value=0.0, value=0.0, step=1.0)
        sub2 = st.number_input("Sub Metering 2 (Wh)", min_value=0.0, value=0.0, step=1.0)
    with c2:
        sub3 = st.number_input("Sub Metering 3 (Wh)", min_value=0.0, value=0.0, step=1.0)
        hour = st.selectbox("Hour", list(range(24)), index=12, format_func=lambda x: f"{x:02d}:00")
        day = st.number_input("Day of month", min_value=1, max_value=31, value=15, step=1)
        month = st.selectbox("Month", list(range(1, 13)), index=0, format_func=lambda x: pd.Timestamp(2000, x, 1).strftime("%B"))
        day_name = st.selectbox("Day of week", list(range(7)), index=0,
                                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    submitted = st.form_submit_button("⚡ Predict Electricity Consumption", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([[
        reactive, voltage, intensity, sub1, sub2, sub3,
        hour, day, month, day_name
    ]], columns=FEATURES)
    try:
        prediction = float(model.predict(input_data)[0])
        st.markdown(f"""
        <div class="result">
          <div class="result-label">Predicted Global Active Power</div>
          <div class="result-value">{prediction:.3f} kW</div>
          <div class="result-label">Approximately {prediction * 1000:,.0f} watts</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Estimated consumption", f"{prediction:.3f} kW")
        st.caption("Prediction is based on the model trained with the UCI household electricity dataset.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("PowerPulse · KNN | Linear Regression | Decision Tree | Random Forest (best model selected by RMSE)")
