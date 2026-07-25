"""
Evaluation Metrics Module.
Calculates MAE, RMSE, R-squared (R²), and performance latency benchmarks.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              training_time: float = 0.0,
                              inference_time: float = 0.0) -> Dict[str, Any]:
    """Computes comprehensive regression and computational latency metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Training Time (s)": round(training_time, 4),
        "Inference Time (s)": round(inference_time, 4)
    }
