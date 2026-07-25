"""
Unit tests for Evaluation Metrics and Model Comparison.
"""

import pytest
import numpy as np
from evaluation.metrics import compute_regression_metrics
from evaluation.comparison import ModelComparison

def test_compute_regression_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])

    metrics = compute_regression_metrics(y_true, y_pred, training_time=1.5, inference_time=0.1)

    assert metrics["MAE"] == 1.0
    assert metrics["RMSE"] == 1.0
    assert metrics["R2"] > 0.95
    assert metrics["Training Time (s)"] == 1.5

def test_model_comparison_dataframe():
    m_lr = {"MAE": 2.5, "RMSE": 3.0, "R2": 0.85, "Training Time (s)": 0.1, "Inference Time (s)": 0.01}
    m_lstm = {"MAE": 0.8, "RMSE": 1.0, "R2": 0.98, "Training Time (s)": 15.0, "Inference Time (s)": 0.2}

    df_comp = ModelComparison.create_comparison_dataframe(m_lr, m_lstm)
    assert len(df_comp) == 2
    assert "RMSE Improvement (%)" in df_comp.columns
    assert df_comp["RMSE Improvement (%)"].iloc[1] > 60.0
