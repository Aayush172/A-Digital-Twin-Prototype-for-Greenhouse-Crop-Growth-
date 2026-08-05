"""
Sensor Simulator Module.
Generates multi-day greenhouse sensor datasets with configurable periods (30, 60, 90, 120 days)
and supports loading real-world CSV sensor datasets and Mendeley tomato microclimate datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from digital_twin.environment import GreenhouseEnvironment
from plant.plant import PlantGrowthModel
from utils.logger import get_logger
from utils.mendeley_loader import MendeleyTomatoDataset
from utils.kaggle_loader import KaggleGreenhouseDataset

logger = get_logger("SensorSimulator")

class SensorSimulator:
    """Generates realistic greenhouse environmental time-series datasets."""

    def __init__(self, seed: int = 42, noise_level: float = 0.05):
        self.seed = seed
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        self.env = GreenhouseEnvironment()
        self.plant_model = PlantGrowthModel()

    def generate_dataset(self, days: int = 60, interval_minutes: int = 60, start_date: str = "2026-05-01 00:00:00") -> pd.DataFrame:
        """
        Generates time-series sensor data and plant growth telemetry.
        Supported days: 30, 60, 90, 120.
        """
        logger.info(f"Generating sensor dataset: {days} days at {interval_minutes}-minute intervals (Seed: {self.seed})...")
        self.rng = np.random.default_rng(self.seed)
        self.plant_model.reset()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        total_steps = int((days * 24 * 60) / interval_minutes)

        records = []
        current_height = 5.0  # Initial seedling height in cm

        for step in range(total_steps):
            current_dt = start_dt + timedelta(minutes=step * interval_minutes)
            env_data = self.env.compute_environmental_reading(current_dt, noise_level=self.noise_level, seed_rng=self.rng)

            # Update plant growth model based on environmental parameters over interval
            delta_days = interval_minutes / (24.0 * 60.0)
            growth_increment = self.plant_model.compute_step_growth(
                temp=env_data["temperature"],
                humidity=env_data["humidity"],
                co2=env_data["co2"],
                par_light=env_data["light_intensity"],
                delta_days=delta_days,
                current_height=current_height
            )
            current_height += growth_increment

            record = {
                "step": step,
                "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": env_data["temperature"],
                "humidity": env_data["humidity"],
                "co2": env_data["co2"],
                "light_intensity": env_data["light_intensity"],
                "plant_height": round(current_height, 3)
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info(f"Successfully generated {len(df)} records. Final plant height: {df['plant_height'].iloc[-1]:.2f} cm")
        return df

    @staticmethod
    def load_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and standardises sensor dataset from CSV file."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

        logger.info(f"Loading dataset from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Standardise column names
        rename_map = {
            "Temp": "temperature", "Temperature": "temperature", "temp": "temperature",
            "RH": "humidity", "Humidity": "humidity", "humidity": "humidity",
            "CO2": "co2", "co2": "co2",
            "PAR": "light_intensity", "Light": "light_intensity", "light": "light_intensity",
            "Height": "plant_height", "plant_height": "plant_height", "Height_cm": "plant_height",
            "Time": "timestamp", "Timestamp": "timestamp", "date": "timestamp"
        }
        df = df.rename(columns=rename_map)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["step"] = df.index

        return df

    @staticmethod
    def load_from_mendeley(data_dir: Optional[Path] = None, download: bool = True, 
                           extract: bool = True, sample: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Loads real tomato microclimate dataset from Mendeley Data.
        
        Args:
            data_dir: Directory to store downloaded/extracted dataset
            download: Whether to download dataset if not present
            extract: Whether to extract ZIP file
            sample: Optional number of records to sample (for quick testing)
            
        Returns:
            Processed DataFrame with standardized columns, or None if loading failed
        """
        logger.info("Loading Mendeley tomato microclimate dataset...")
        
        loader = MendeleyTomatoDataset(data_dir=data_dir)
        df = loader.load_full_dataset(download=download, extract=extract)
        
        if df is None:
            logger.error("Failed to load Mendeley dataset")
            return None
        
        logger.info(f"Loaded {len(df)} records from Mendeley dataset")
        
        # Optionally sample for faster testing
        if sample is not None and len(df) > sample:
            df = df.sample(n=sample, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {sample} records for testing")
        
        return df

    @staticmethod
    def load_from_kaggle_agc(data_dir: Optional[Path] = None, download: bool = True,
                             sample: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Loads real greenhouse dataset from Kaggle AGC 2nd challenge.

        Args:
            data_dir: Directory for cached processed data
            download: Whether to download via kagglehub if needed
            sample: Optional sample size for quick dashboard rendering

        Returns:
            Standardized DataFrame or None on failure
        """
        logger.info("Loading Kaggle AGC greenhouse dataset...")
        loader = KaggleGreenhouseDataset(data_dir=data_dir)
        df = loader.load_full_dataset(download=download, sample=sample)

        if df is None:
            logger.error("Failed to load Kaggle AGC dataset")
            return None

        logger.info(f"Loaded {len(df)} records from Kaggle AGC dataset")
        return df

