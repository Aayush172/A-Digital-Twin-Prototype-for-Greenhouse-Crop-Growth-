"""
Streamlit Digital Twin Telemetry View Component.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_twin_telemetry_view(df_history: pd.DataFrame) -> None:
    """Renders historical sensor time-series telemetry charts."""
    st.subheader("📊 Real-Time Digital Twin Sensor Telemetry")

    if df_history.empty:
        st.info("No telemetry history recorded yet. Please run the simulation step or timeline playback.")
        return

    # Multi-variable environmental time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['temperature'], name="Temperature (°C)", line=dict(color='#ff5722')))
    fig.add_trace(go.Scatter(x=df_history['timestamp'], y=df_history['humidity'], name="Humidity (%)", line=dict(color='#0288d1'), yaxis="y2"))

    fig.update_layout(
        title="Microclimate Sensor Streams (Temperature & Humidity)",
        xaxis=dict(title="Timestamp"),
        yaxis=dict(
            title_text="Temperature (°C)",
            title_font=dict(color='#ff5722'),
            tickfont=dict(color='#ff5722')
        ),
        yaxis2=dict(
            title_text="Relative Humidity (%)",
            title_font=dict(color='#0288d1'),
            tickfont=dict(color='#0288d1'),
            overlaying="y",
            side="right"
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, width='stretch')

    # CO2 and PAR Light Charts
    c1, c2 = st.columns(2)
    with c1:
        fig_co2 = px.line(df_history, x='timestamp', y='co2', title="CO₂ Concentration (ppm)", color_discrete_sequence=['#4caf50'])
        st.plotly_chart(fig_co2, width='stretch')

    with c2:
        fig_par = px.line(df_history, x='timestamp', y='light_intensity', title="PAR Light Intensity (μmol/m²/s)", color_discrete_sequence=['#ffb300'])
        st.plotly_chart(fig_par, width='stretch')
