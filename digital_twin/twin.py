"""
Greenhouse Digital Twin Core Engine.
Manages physical-to-digital state synchronization, SQLite telemetry database persistence,
and real-time crop prediction integration.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from digital_twin.state import DigitalTwinState
from utils.logger import get_logger

logger = get_logger("DigitalTwin")

class GreenhouseDigitalTwin:
    """Digital Twin instance managing state synchronization and database logging."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self.current_state: Optional[DigitalTwinState] = None
        self.history: List[DigitalTwinState] = []
        if self.db_path:
            self._init_database()

    def _init_database(self) -> None:
        """Initializes SQLite database schema for Digital Twin state storage."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_telemetry (
                    step INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    humidity REAL NOT NULL,
                    co2 REAL NOT NULL,
                    light_intensity REAL NOT NULL,
                    plant_height REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info(f"Initialized SQLite database at {self.db_path}")

    def update_state(self, new_state: DigitalTwinState, persist: bool = True) -> None:
        """Updates internal state and optionally persists to SQLite database."""
        self.current_state = new_state
        self.history.append(new_state)

        if persist and self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO twin_telemetry 
                    (step, timestamp, temperature, humidity, co2, light_intensity, plant_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_state.step,
                    new_state.timestamp,
                    new_state.temperature,
                    new_state.humidity,
                    new_state.co2,
                    new_state.light_intensity,
                    new_state.plant_height
                ))
                conn.commit()

    def load_history_from_db(self) -> pd.DataFrame:
        """Loads historical twin telemetry dataframe from SQLite."""
        if not self.db_path or not self.db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM twin_telemetry ORDER BY step ASC", conn)
        return df

    def get_history_dataframe(self) -> pd.DataFrame:
        """Returns history as a Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        records = [state.to_dict() for state in self.history]
        return pd.DataFrame(records)

    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Returns summary of current Digital Twin state."""
        if not self.current_state:
            return {}
        return {
            "Step": self.current_state.step,
            "Timestamp": self.current_state.timestamp,
            "Temperature (°C)": self.current_state.temperature,
            "Relative Humidity (%)": self.current_state.humidity,
            "CO2 (ppm)": self.current_state.co2,
            "Light Intensity PAR (μmol/m²/s)": self.current_state.light_intensity,
            "Plant Height (cm)": self.current_state.plant_height
        }
