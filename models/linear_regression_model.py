"""
Linear Regression Baseline Model for Crop Growth Prediction.
"""

import time
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger("LinearRegressionModel")

class BaselineLinearRegressionModel:
    """Baseline Scikit-learn Linear Regression model using lagged feature representations."""

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.training_time = 0.0
        self.inference_time = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Fits Linear Regression model and records execution time."""
        logger.info(f"Training Baseline Linear Regression model on X shape {X_train.shape}...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        logger.info(f"Linear Regression training completed in {self.training_time:.4f} seconds.")
        return {"training_time_sec": self.training_time}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions and measures inference duration."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
        start_time = time.time()
        preds = self.model.predict(X)
        self.inference_time = time.time() - start_time
        return preds

    def save(self, filepath: Path) -> None:
        """Saves model to disk via joblib."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Linear Regression model to {filepath}")

    def load(self, filepath: Path) -> None:
        """Loads model from disk via joblib."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Loaded Linear Regression model from {filepath}")
