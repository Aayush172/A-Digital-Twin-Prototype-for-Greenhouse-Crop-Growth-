"""
Unit tests for DataPreprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
from models.preprocess import DataPreprocessor

@pytest.fixture
def sample_dataframe():
    dates = pd.date_range("2026-05-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature": np.random.uniform(18, 28, 100),
        "humidity": np.random.uniform(50, 80, 100),
        "co2": np.random.uniform(500, 900, 100),
        "light_intensity": np.random.uniform(0, 800, 100),
        "plant_height": np.linspace(5, 25, 100)
    })
    return df

def test_preprocessor_fit_transform(sample_dataframe):
    prep = DataPreprocessor(sequence_length=12)
    df_scaled = prep.fit_transform(sample_dataframe)

    assert "plant_height_scaled" in df_scaled.columns
    assert df_scaled["temperature"].max() <= 1.0
    assert df_scaled["temperature"].min() >= 0.0

def test_create_sequences(sample_dataframe):
    prep = DataPreprocessor(sequence_length=12)
    df_scaled = prep.fit_transform(sample_dataframe)
    X, y = prep.create_sequences(df_scaled)

    assert X.shape == (100 - 12, 12, 5)
    assert y.shape == (100 - 12,)

