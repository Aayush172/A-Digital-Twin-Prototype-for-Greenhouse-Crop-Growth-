"""
Unit tests for SensorSimulator and GreenhouseEnvironment.
"""

import pytest
import pandas as pd
from digital_twin.sensor_simulator import SensorSimulator

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
