"""
Streamlit Web Dashboard for Greenhouse Digital Twin & Crop Growth Prediction.
"""

import sys
from pathlib import Path

# Add project root directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import paths, sim_config
from digital_twin.sensor_simulator import SensorSimulator
from digital_twin.twin import GreenhouseDigitalTwin
from digital_twin.state import DigitalTwinState
from dashboard.components.metrics_cards import render_metrics_cards
from dashboard.components.twin_view import render_twin_telemetry_view
from dashboard.components.lsystem_view import render_lsystem_view
from dashboard.components.model_eval_view import render_model_eval_view
from dashboard.components.rl_view import render_rl_view

st.set_page_config(
    page_title="Greenhouse Digital Twin Prototype",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f1f8e9;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌿 Greenhouse Digital Twin Prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Crop Growth Prediction using Sensor Data & Stacked LSTM Networks | MSc Artificial Intelligence Capstone Project</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.title("🎛️ Dataset Controls")

real_data_sample = st.sidebar.number_input(
    "Sample size for Mendeley data",
    min_value=0,
    value=0,
    step=100,
    help="Set to 0 to use the full local Mendeley dataset; use a sample for faster loading and rendering."
)

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Stacked LSTM Network", "Linear Regression (Baseline)"]
)

# Cached Real Dataset Loading
@st.cache_data
def get_real_sensor_data(sample_size: int):
    simulator = SensorSimulator()
    source = "Mendeley tomato greenhouse dataset"
    df = simulator.load_from_mendeley(
        data_dir=paths.raw_data_dir,
        download=False,
        extract=False,
        sample=None,
    )

    if df is None:
        st.warning("Mendeley data is unavailable. Loading the last generated dataset instead.")
        source = "Last generated dataset (Mendeley unavailable)"
        if paths.dataset_csv.exists():
            df = simulator.load_from_csv(paths.dataset_csv)

    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        for column, default in (("co2", 400.0), ("light_intensity", 300.0)):
            if column not in df.columns:
                df[column] = default
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)
            if df[column].notna().any():
                df[column] = df[column].interpolate().ffill().bfill()
            else:
                df[column] = default

        if sample_size > 0 and len(df) > sample_size:
            sample_indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            df = df.iloc[sample_indices].reset_index(drop=True)

        df["step"] = df.index

    return df, source

df_dataset, data_source = get_real_sensor_data(real_data_sample)

if df_dataset is None:
    st.error(
        "Unable to load any greenhouse dataset. Please ensure the local dataset exists at data/greenhouse_dataset.csv."
    )
    st.stop()

st.caption(f"Data source: {data_source}")

# Timeline Step Slider
max_step = len(df_dataset) - 1
selected_step = st.sidebar.slider(
    "Timeline Slider (Data Record)",
    min_value=0,
    max_value=max_step,
    value=max_step
)

# Construct Instantaneous State
current_row = df_dataset.iloc[selected_step]
current_state = DigitalTwinState(
    timestamp=str(current_row['timestamp']),
    temperature=float(current_row['temperature']),
    humidity=float(current_row['humidity']),
    co2=float(current_row['co2']),
    light_intensity=float(current_row['light_intensity']),
    plant_height=float(current_row['plant_height']),
    step=int(current_row['step'])
)

# -------------------------------------------------------------
# Top Metric Cards
# -------------------------------------------------------------
render_metrics_cards(current_state)

st.markdown("---")

# -------------------------------------------------------------
# Navigation Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Digital Twin & Sensor Telemetry",
    "🪴 Procedural L-System Plant Growth",
    "📈 Model Predictions & MSc Evaluation",
    "🤖 RL Environmental Control"
])

with tab1:
    df_history_sub = df_dataset.iloc[:selected_step + 1]
    render_twin_telemetry_view(df_history_sub)

with tab2:
    render_lsystem_view(current_state.plant_height)

with tab3:
    render_model_eval_view(paths.results_dir, paths.figures_dir)

with tab4:
    render_rl_view(paths.results_dir, paths.models_dir, current_state)
