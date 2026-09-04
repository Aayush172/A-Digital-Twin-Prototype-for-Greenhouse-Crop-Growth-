"""
Streamlit Model Evaluation & Comparative Analytics View.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

def render_model_eval_view(results_dir: Path, figures_dir: Path) -> None:
    """Renders machine learning performance metrics and comparative evaluation plots."""
    st.subheader("📈 Machine Learning Model Evaluation & Research Results")

    summary_csv = results_dir / "metrics_summary.csv"
    if summary_csv.exists():
        df_comp = pd.read_csv(summary_csv)
        st.markdown("### Model Performance Metrics Benchmark")
        st.dataframe(df_comp.style.highlight_min(axis=0, subset=['MAE', 'RMSE'], color='#c8e6c9').highlight_max(axis=0, subset=['R2'], color='#c8e6c9'), width='stretch')
    else:
        st.info("Metrics summary CSV not found. Please run main.py or click 'Run Evaluation Pipeline' in sidebar.")

    st.markdown("---")
    st.markdown("### Comparative Performance Visualisations")

    fig_act_vs_pred = figures_dir / "actual_vs_predicted.png"
    fig_residuals = figures_dir / "residuals_comparison.png"
    fig_loss = figures_dir / "lstm_training_loss.png"

    col1, col2 = st.columns(2)
    with col1:
        if fig_act_vs_pred.exists():
            st.image(str(fig_act_vs_pred), caption="Actual vs Predicted Growth Trajectory (LR vs LSTM)", width='stretch')
    with col2:
        if fig_residuals.exists():
            st.image(str(fig_residuals), caption="Residual Distributions & Error Scatter", width='stretch')

    if fig_loss.exists():
        st.image(str(fig_loss), caption="Stacked LSTM Training and Validation Loss Convergence", width='stretch')
