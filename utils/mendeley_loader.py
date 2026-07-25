"""
Mendeley Tomato Dataset Loader.
Downloads and processes the microclimate monitoring dataset for commercial tomato greenhouse production
from https://data.mendeley.com/datasets/tkbkzdt5nr/2

Dataset contains:
- Environment measurements: temperature, relative humidity, light (PAR)
- Tomato measurements: truss weight, stem length, stem diameter, firmness, color, soluble solids, ethylene, etc.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
import requests

from utils.logger import get_logger

logger = get_logger("MendeleyLoader")

MENDELEY_DATASET_URL = "https://data.mendeley.com/public-api/zip/tkbkzdt5nr/download/2"
MENDELEY_DATASET_ID = "tkbkzdt5nr"


class MendeleyTomatoDataset:
    """Loads and processes Mendeley tomato greenhouse microclimate dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize loader with optional data directory."""
        self.data_dir = Path(data_dir) if data_dir else Path("data/mendeley")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.data_dir / f"{MENDELEY_DATASET_ID}.zip"
        self.extract_dir = self.data_dir / "extracted"

    def download_dataset(self, force: bool = False) -> bool:
        """
        Downloads Mendeley dataset if not already present.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            True if download successful or file already exists
        """
        if self.zip_path.exists() and not force:
            logger.info(f"Dataset already downloaded at {self.zip_path}")
            return True

        try:
            logger.info(f"Downloading Mendeley dataset from {MENDELEY_DATASET_URL}...")
            response = requests.get(MENDELEY_DATASET_URL, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(self.zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"Downloaded {percent:.1f}%")

            logger.info(f"Successfully downloaded dataset ({downloaded / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False

    def extract_dataset(self, force: bool = False) -> bool:
        """Extracts dataset ZIP file."""
        if not self.zip_path.exists():
            logger.error(f"ZIP file not found at {self.zip_path}. Download first.")
            return False

        if self.extract_dir.exists() and not force:
            logger.info(f"Dataset already extracted at {self.extract_dir}")
            return True

        try:
            logger.info(f"Extracting dataset to {self.extract_dir}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            logger.info("Dataset extraction complete")
            return True

        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            return False

    def load_environment_data(self) -> Optional[pd.DataFrame]:
        """Loads environment measurements (temperature, humidity, light)."""
        env_dir = self.extract_dir / "Environment"
        
        if not env_dir.exists():
            logger.error(f"Environment folder not found at {env_dir}")
            return None

        env_files = list(env_dir.glob("*.csv"))
        if not env_files:
            logger.error(f"No CSV files found in {env_dir}")
            return None

        logger.info(f"Loading {len(env_files)} environment CSV files...")
        env_dfs = []

        for csv_file in env_files:
            try:
                df = pd.read_csv(csv_file)
                logger.debug(f"Loaded {csv_file.name}: {len(df)} rows, columns: {list(df.columns)}")
                env_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file.name}: {e}")

        if not env_dfs:
            return None

        # Concatenate and standardize
        env_df = pd.concat(env_dfs, ignore_index=True)
        env_df = self._standardize_env_columns(env_df)

        logger.info(f"Loaded {len(env_df)} environment records")
        return env_df

    def load_plant_data(self) -> Optional[pd.DataFrame]:
        """Loads plant measurements (stem length, stem diameter, truss weight, etc.)."""
        plant_dir = self.extract_dir / "PlantData"
        
        if not plant_dir.exists():
            logger.error(f"PlantData folder not found at {plant_dir}")
            return None

        plant_files = list(plant_dir.glob("*.csv"))
        if not plant_files:
            logger.error(f"No CSV files found in {plant_dir}")
            return None

        logger.info(f"Loading {len(plant_files)} plant data CSV files...")
        plant_dfs = []

        for csv_file in plant_files:
            try:
                df = pd.read_csv(csv_file)
                logger.debug(f"Loaded {csv_file.name}: {len(df)} rows")
                plant_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file.name}: {e}")

        if not plant_dfs:
            return None

        plant_df = pd.concat(plant_dfs, ignore_index=True)
        logger.info(f"Loaded {len(plant_df)} plant records")
        return plant_df

    def _standardize_env_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes environment data column names."""
        rename_map = {
            "Temp": "temperature",
            "Temperature": "temperature",
            "T": "temperature",
            "RH": "humidity",
            "Humidity": "humidity",
            "H": "humidity",
            "PAR": "light_intensity",
            "Light": "light_intensity",
            "par": "light_intensity",
            "Date": "timestamp",
            "date": "timestamp",
            "Time": "timestamp",
            "time": "timestamp",
            "Timestamp": "timestamp",
        }
        
        df = df.rename(columns=rename_map)
        
        # Parse timestamp if exists
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except:
                logger.warning("Could not parse timestamp column")

        # Fill missing essential columns with defaults
        if "temperature" not in df.columns:
            df["temperature"] = 22.0
            logger.warning("temperature column not found, using default 22.0°C")
        
        if "humidity" not in df.columns:
            df["humidity"] = 65.0
            logger.warning("humidity column not found, using default 65%")
        
        if "light_intensity" not in df.columns:
            df["light_intensity"] = 300.0
            logger.warning("light_intensity column not found, using default 300 µmol/m²/s")

        return df

    def create_unified_dataset(self, env_df: Optional[pd.DataFrame] = None,
                              plant_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Creates unified dataset from environment and plant measurements.
        
        If environment and plant data use different time intervals, we merge them
        and interpolate plant growth metrics.
        """
        if env_df is None:
            env_df = self.load_environment_data()
        if env_df is None:
            logger.error("Failed to load environment data")
            return None

        # If we have plant data, try to merge with environment data
        if plant_df is not None and "timestamp" in plant_df.columns:
            logger.info("Merging plant and environment data...")
            # Perform nearest-neighbor merge if timestamps don't align exactly
            if "timestamp" in plant_df.columns:
                plant_df["timestamp"] = pd.to_datetime(plant_df["timestamp"])
                env_df = env_df.sort_values("timestamp")
                plant_df = plant_df.sort_values("timestamp")
                
                # Merge on nearest timestamp
                merged_df = pd.merge_asof(env_df, plant_df, on="timestamp", direction="nearest")
            else:
                merged_df = env_df
        else:
            merged_df = env_df

        # Add missing expected columns with reasonable defaults
        if "plant_height" not in merged_df.columns:
            # Estimate plant height growth over time if we have stem_length
            if "stem_length" in merged_df.columns:
                merged_df["plant_height"] = merged_df["stem_length"]
            else:
                # Linear growth from 5cm to 200cm over the time series
                n_records = len(merged_df)
                merged_df["plant_height"] = np.linspace(5.0, 200.0, n_records)

        if "co2" not in merged_df.columns:
            merged_df["co2"] = 400.0  # Default CO2 concentration in ppm

        # Add step counter
        merged_df["step"] = range(len(merged_df))

        # Select essential columns
        essential_cols = ["step", "timestamp", "temperature", "humidity", "co2", "light_intensity", "plant_height"]
        available_cols = [col for col in essential_cols if col in merged_df.columns]
        result_df = merged_df[available_cols].copy()

        logger.info(f"Created unified dataset with {len(result_df)} records")
        return result_df

    def load_full_dataset(self, download: bool = True, extract: bool = True) -> Optional[pd.DataFrame]:
        """
        Complete pipeline: download → extract → load → merge.
        
        Returns:
            Unified DataFrame with environment and plant data, or None if failed
        """
        if download:
            if not self.download_dataset():
                logger.error("Download failed")
                return None

        if extract:
            if not self.extract_dataset():
                logger.error("Extraction failed")
                return None

        env_df = self.load_environment_data()
        plant_df = self.load_plant_data()

        unified_df = self.create_unified_dataset(env_df, plant_df)
        return unified_df

    def save_to_csv(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> bool:
        """Saves processed dataset to CSV."""
        if output_path is None:
            output_path = self.data_dir / "mendeley_dataset_processed.csv"

        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed dataset to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False
