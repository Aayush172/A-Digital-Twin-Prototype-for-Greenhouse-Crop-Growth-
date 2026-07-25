"""
Model Comparison Module.
Aggregates performance metrics across baseline and deep learning models into tabular summaries.
"""

import pandas as pd
from typing import Dict, Any

class ModelComparison:
    """Structures comparative model evaluation tables."""

    @staticmethod
    def create_comparison_dataframe(metrics_lr: Dict[str, Any], metrics_lstm: Dict[str, Any]) -> pd.DataFrame:
        """Combines LR baseline and LSTM model metrics into a comparison DataFrame."""
        df = pd.DataFrame([
            {"Model": "Linear Regression (Baseline)", **metrics_lr},
            {"Model": "Stacked LSTM Neural Network", **metrics_lstm}
        ])
        
        # Calculate percentage improvements
        if metrics_lr.get("RMSE", 0) > 0:
            rmse_imp = ((metrics_lr["RMSE"] - metrics_lstm["RMSE"]) / metrics_lr["RMSE"]) * 100.0
            mae_imp = ((metrics_lr["MAE"] - metrics_lstm["MAE"]) / metrics_lr["MAE"]) * 100.0
        else:
            rmse_imp, mae_imp = 0.0, 0.0

        df["RMSE Improvement (%)"] = [0.0, round(rmse_imp, 2)]
        df["MAE Improvement (%)"] = [0.0, round(mae_imp, 2)]
        return df
