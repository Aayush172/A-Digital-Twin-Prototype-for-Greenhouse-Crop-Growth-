"""
Unit tests for Digital Twin State Engine and SQLite Database.
"""

import pytest
import tempfile
from pathlib import Path
from digital_twin.state import DigitalTwinState
from digital_twin.twin import GreenhouseDigitalTwin

def test_digital_twin_state_serialization():
    state = DigitalTwinState(
        timestamp="2026-05-01 12:00:00",
        temperature=24.5,
        humidity=65.0,
        co2=800.0,
        light_intensity=650.0,
        plant_height=15.2,
        step=12
    )

    d = state.to_dict()
    assert d["temperature"] == 24.5
    
    state_restored = DigitalTwinState.from_dict(d)
    assert state_restored.plant_height == 15.2

def test_twin_sqlite_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_twin.db"
        twin = GreenhouseDigitalTwin(db_path=db_path)

        state = DigitalTwinState(
            timestamp="2026-05-01 12:00:00",
            temperature=24.5,
            humidity=65.0,
            co2=800.0,
            light_intensity=650.0,
            plant_height=15.2,
            step=1
        )
        twin.update_state(state, persist=True)

        df_db = twin.load_history_from_db()
        assert len(df_db) == 1
        assert df_db["temperature"].iloc[0] == 24.5
