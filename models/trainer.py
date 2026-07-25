"""
Model Trainer Orchestrator.
Handles end-to-end model training, validation, duration timing, and model artifact persistence.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

from models.linear_regression_model import BaselineLinearRegressionModel
from models.lstm_model import GreenhouseLSTMModel
from utils.logger import get_logger

logger = get_logger("ModelTrainer")

class ModelTrainer:
    """Orchestrates model execution, fitting, and model saving."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def train_baseline_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> BaselineLinearRegressionModel:
        """Trains baseline Linear Regression model."""
        lr_model = BaselineLinearRegressionModel()
        lr_model.fit(X_train, y_train)
        lr_model.save(self.models_dir / "linear_regression.pkl")
        return lr_model

    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         val_data: Tuple[np.ndarray, np.ndarray],
                         epochs: int = 35, batch_size: int = 32) -> GreenhouseLSTMModel:
        """Trains Stacked LSTM model with validation monitoring."""
        seq_length, n_features = X_train.shape[1], X_train.shape[2]
        lstm_model = GreenhouseLSTMModel(sequence_length=seq_length, n_features=n_features)
        checkpoint_file = self.models_dir / "best_lstm_model.pt"
        lstm_model.fit(X_train, y_train, val_data=val_data, epochs=epochs, batch_size=batch_size, checkpoint_path=checkpoint_file)
        lstm_model.save(self.models_dir / "lstm_model_final.pt")
        return lstm_model
