"""
Unit tests for SensorSimulator and GreenhouseEnvironment.
"""

import pytest
import pandas as pd
from pathlib import Path
from digital_twin.sensor_simulator import SensorSimulator
from utils.mendeley_loader import MendeleyTomatoDataset

def test_sensor_simulator_dataset_generation():
    simulator = SensorSimulator(seed=42)
    df = simulator.generate_dataset(days=5, interval_minutes=60)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5 * 24
    assert set(["step", "timestamp", "temperature", "humidity", "co2", "light_intensity", "plant_height"]).issubset(df.columns)
    assert df["plant_height"].iloc[-1] > df["plant_height"].iloc[0]

def test_reproducibility_with_seed():
    sim1 = SensorSimulator(seed=123)
    df1 = sim1.generate_dataset(days=2, interval_minutes=60)

    sim2 = SensorSimulator(seed=123)
    df2 = sim2.generate_dataset(days=2, interval_minutes=60)

    pd.testing.assert_frame_equal(df1, df2)


def test_kaggle_loader_coerces_sensor_columns_to_numeric():
    simulator = SensorSimulator(seed=42)
    df = simulator.load_from_csv("data/greenhouse_dataset.csv")

    assert df is not None
    assert pd.api.types.is_numeric_dtype(df["temperature"])
    assert pd.api.types.is_numeric_dtype(df["humidity"])
    assert pd.api.types.is_numeric_dtype(df["co2"])
    assert pd.api.types.is_numeric_dtype(df["light_intensity"])
    assert pd.api.types.is_numeric_dtype(df["plant_height"])


def test_mendeley_loader_uses_local_excel_files_when_download_is_unavailable():
    loader = MendeleyTomatoDataset(data_dir=Path("data/raw"))
    df = loader.load_environment_data()

    assert df is not None
    assert len(df) > 0
    assert {"timestamp", "temperature", "humidity", "light_intensity"}.issubset(df.columns)
    assert pd.api.types.is_numeric_dtype(df["temperature"])
    assert pd.api.types.is_numeric_dtype(df["humidity"])
    assert pd.api.types.is_numeric_dtype(df["light_intensity"])
