"""
Streamlit L-System Plant Growth Visualisation View.
"""

import streamlit as st
import plotly.graph_objects as go
from plant.renderer import PlantRenderer

def render_lsystem_view(current_height_cm: float) -> None:
    """Renders procedural L-system plant growth structure."""
    st.subheader("🪴 Procedural Plant Architecture Visualisation")
    st.write(
        "This component generates an interactive L-System plant structure directly driven by "
        "the Digital Twin's instantaneous crop height and growth state."
    )

    renderer = PlantRenderer(branching_angle=25.0)
    fig_plant = renderer.render_plotly_2d(current_height_cm)

    st.plotly_chart(fig_plant, width='stretch')
