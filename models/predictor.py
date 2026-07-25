"""
Growth Predictor Interface.
Provides real-time single-step and multi-horizon crop growth predictions for the Digital Twin state update engine.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, List

from models.preprocess import DataPreprocessor
from models.linear_regression_model import BaselineLinearRegressionModel
from models.lstm_model import GreenhouseLSTMModel
from utils.logger import get_logger

logger = get_logger("GrowthPredictor")

class GrowthPredictor:
    """Provides inference services for plant height growth forecasting."""

    def __init__(self, preprocessor: DataPreprocessor,
                 lr_model: BaselineLinearRegressionModel,
                 lstm_model: GreenhouseLSTMModel):
        self.preprocessor = preprocessor
        self.lr_model = lr_model
        self.lstm_model = lstm_model

    def predict_next_height_cm(self, recent_history_df: pd.DataFrame, model_type: str = "lstm") -> float:
        """
        Predicts next step plant height (cm) given recent history window (DataFrame of length >= sequence_length).
        """
        if len(recent_history_df) < self.preprocessor.sequence_length:
            raise ValueError(f"History window must contain at least {self.preprocessor.sequence_length} timesteps.")

        window_df = recent_history_df.tail(self.preprocessor.sequence_length)
        df_scaled = self.preprocessor.fit_transform(window_df)

        if model_type.lower() == "lstm":
            X_seq, _ = self.preprocessor.create_sequences(df_scaled)
            if len(X_seq) == 0:
                # Fallback to last sequence shape
                feat_matrix = df_scaled[self.preprocessor.feature_cols].values
                X_seq = np.expand_dims(feat_matrix, axis=0)
            pred_scaled = self.lstm_model.predict(X_seq)[-1]
        elif model_type.lower() in ["lr", "linear_regression"]:
            X_tab, _ = self.preprocessor.create_tabular_lagged_features(df_scaled)
            if len(X_tab) == 0:
                feat_matrix = df_scaled[self.preprocessor.feature_cols].values
                X_tab = feat_matrix.flatten().reshape(1, -1)
            pred_scaled = self.lr_model.predict(X_tab)[-1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        pred_cm = float(self.preprocessor.inverse_transform_target(np.array([pred_scaled]))[0])
        return pred_cm
