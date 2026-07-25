"""
Streamlit KPI Metric Cards Component.
"""

import streamlit as st
from digital_twin.state import DigitalTwinState

def render_metrics_cards(state: DigitalTwinState) -> None:
    """Renders top-level KPI metric cards for Digital Twin instantaneous state."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{state.temperature:.1f} °C",
            delta=f"{(state.temperature - 22.0):+.1f} °C"
        )
    with col2:
        st.metric(
            label="💧 Humidity",
            value=f"{state.humidity:.1f} %",
            delta=f"{(state.humidity - 70.0):+.1f} %"
        )
    with col3:
        st.metric(
            label="🌿 CO₂ Conc.",
            value=f"{state.co2:.0f} ppm",
            delta=f"{(state.co2 - 800.0):+.0f} ppm"
        )
    with col4:
        st.metric(
            label="☀️ PAR Light",
            value=f"{state.light_intensity:.0f} μmol/m²/s"
        )
    with col5:
        st.metric(
            label="🌱 Plant Height",
            value=f"{state.plant_height:.2f} cm"
        )
