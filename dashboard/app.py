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
st.sidebar.title("🎛️ Simulation Controls")

sim_days = st.sidebar.select_slider(
    "Simulation Period (Days)",
    options=[30, 60, 90, 120],
    value=60
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=9999,
    value=42
)

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Stacked LSTM Network", "Linear Regression (Baseline)"]
)

# Cached Dataset Generation
@st.cache_data
def get_sensor_data(days: int, seed: int):
    simulator = SensorSimulator(seed=seed)
    return simulator.generate_dataset(days=days, interval_minutes=60)

df_dataset = get_sensor_data(sim_days, random_seed)

# Timeline Step Slider
max_step = len(df_dataset) - 1
selected_step = st.sidebar.slider(
    "Timeline Slider (Simulation Hour)",
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
tab1, tab2, tab3 = st.tabs([
    "📊 Digital Twin & Sensor Telemetry",
    "🪴 Procedural L-System Plant Growth",
    "📈 Model Predictions & MSc Evaluation"
])

with tab1:
    df_history_sub = df_dataset.iloc[:selected_step + 1]
    render_twin_telemetry_view(df_history_sub)

with tab2:
    render_lsystem_view(current_state.plant_height)

with tab3:
    render_model_eval_view(paths.results_dir, paths.figures_dir)
