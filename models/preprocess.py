"""
Time-Series Data Preprocessor for Crop Growth ML Models.
Handles missing value imputation, MinMax scaling, sliding window sequence generation,
and chronological train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
from utils.logger import get_logger

logger = get_logger("Preprocessor")

class DataPreprocessor:
    """Preprocesses environmental sensor time-series for sequence learning."""

    def __init__(self, sequence_length: int = 24, target_col: str = "plant_height",
                 feature_cols: Optional[List[str]] = None):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.feature_cols = feature_cols or ["temperature", "humidity", "co2", "light_intensity", "plant_height"]
        
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses dataframe (sorting, cleaning, scaling)."""
        df_clean = df.copy()
        
        # Sort chronologically if timestamp column exists
        if "timestamp" in df_clean.columns:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
            df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

        # Impute missing values (forward fill + backward fill)
        df_clean[self.feature_cols] = df_clean[self.feature_cols].ffill().bfill()

        # Fit scalers
        scaled_features = self.feature_scaler.fit_transform(df_clean[self.feature_cols])
        scaled_target = self.target_scaler.fit_transform(df_clean[[self.target_col]])

        df_scaled = pd.DataFrame(scaled_features, columns=self.feature_cols)
        df_scaled[f"{self.target_col}_scaled"] = scaled_target
        self.is_fitted = True

        return df_scaled

    def create_sequences(self, df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D sequences (X) and target vectors (y) for LSTM.
        X shape: (N_samples, sequence_length, N_features)
        y shape: (N_samples,)
        """
        feature_matrix = df_scaled[self.feature_cols].values
        target_vector = df_scaled[f"{self.target_col}_scaled"].values

        X, y = [], []
        for i in range(len(feature_matrix) - self.sequence_length):
            X.append(feature_matrix[i : i + self.sequence_length])
            y.append(target_vector[i + self.sequence_length])

        return np.array(X), np.array(y)

    def create_tabular_lagged_features(self, df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flattens sliding window into 2D tabular matrix for Linear Regression baseline.
        X shape: (N_samples, sequence_length * N_features)
        y shape: (N_samples,)
        """
        X_seq, y = self.create_sequences(df_scaled)
        n_samples, seq_len, n_features = X_seq.shape
        X_tab = X_seq.reshape(n_samples, seq_len * n_features)
        return X_tab, y

    def split_train_test(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Chronologically splits arrays into training and testing sets."""
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Restores scaled target predictions back to original cm scale."""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()
