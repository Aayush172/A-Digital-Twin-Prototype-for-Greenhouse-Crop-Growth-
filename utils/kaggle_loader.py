"""
Kaggle AGC Dataset Loader.
Downloads and standardizes the Autonomous Greenhouse Challenge dataset
from https://www.kaggle.com/datasets/piantic/autonomous-greenhouse-challengeagc-2nd-2019.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("KaggleLoader")

KAGGLE_DATASET_SLUG = "piantic/autonomous-greenhouse-challengeagc-2nd-2019"


class KaggleGreenhouseDataset:
    """Loads and standardizes AGC greenhouse telemetry data from Kaggle."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.data_dir / "kaggle_agc"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    @property
    def processed_csv_path(self) -> Path:
        return self.dataset_dir / "agc_dataset_processed.csv"

    def download_dataset(self) -> Optional[Path]:
        """Downloads latest AGC dataset via kagglehub into local cache path."""
        try:
            import kagglehub
        except Exception as e:
            logger.error(f"kagglehub is not installed: {e}")
            return None

        try:
            logger.info(f"Downloading Kaggle dataset: {KAGGLE_DATASET_SLUG}")
            path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
            logger.info(f"Dataset downloaded to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            return None

    def _score_columns(self, columns: list[str]) -> int:
        cols = [c.lower() for c in columns]
        score = 0
        if any("temp" in c for c in cols):
            score += 2
        if any("humid" in c or c == "rh" for c in cols):
            score += 2
        if any("co2" in c for c in cols):
            score += 1
        if any("light" in c or "par" in c or "radiation" in c for c in cols):
            score += 1
        if any("time" in c or "date" in c or "timestamp" in c for c in cols):
            score += 2
        return score

    def _find_best_csv(self, root: Path) -> Optional[Path]:
        csv_files = list(root.rglob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found under {root}")
            return None

        best: Optional[Tuple[int, Path]] = None
        for csv_file in csv_files:
            try:
                head = pd.read_csv(csv_file, nrows=30)
                score = self._score_columns(list(head.columns))
                if best is None or score > best[0]:
                    best = (score, csv_file)
            except Exception:
                continue

        if best is None:
            return None

        logger.info(f"Selected CSV: {best[1].name} (score={best[0]})")
        return best[1]

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        def _coalesce_duplicates(frame: pd.DataFrame, target: str) -> pd.DataFrame:
            cols = [c for c in frame.columns if c == target]
            if len(cols) <= 1:
                return frame

            merged = frame[cols].bfill(axis=1).iloc[:, 0]
            frame = frame.drop(columns=cols)
            frame[target] = merged
            return frame

        rename_map = {}
        for col in df.columns:
            key = col.strip().lower()
            if key in {"timestamp", "time", "date", "datetime", "date_time"}:
                rename_map[col] = "timestamp"
            elif "temp" in key or key in {"t", "air_temperature"}:
                rename_map[col] = "temperature"
            elif "humid" in key or key in {"rh", "relative_humidity"}:
                rename_map[col] = "humidity"
            elif "co2" in key:
                rename_map[col] = "co2"
            elif "light" in key or "par" in key or "radiation" in key:
                rename_map[col] = "light_intensity"
            elif "height" in key or "stem_length" in key:
                rename_map[col] = "plant_height"

        df = df.rename(columns=rename_map)
        df = _coalesce_duplicates(df, "co2")
        df = _coalesce_duplicates(df, "light_intensity")
        df = _coalesce_duplicates(df, "temperature")
        df = _coalesce_duplicates(df, "humidity")
        df = _coalesce_duplicates(df, "plant_height")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="h")

        if "temperature" not in df.columns:
            df["temperature"] = 22.0
        if "humidity" not in df.columns:
            df["humidity"] = 65.0
        if "co2" not in df.columns:
            df["co2"] = 400.0
        if "light_intensity" not in df.columns:
            df["light_intensity"] = 300.0

        if "plant_height" not in df.columns:
            df["plant_height"] = np.linspace(5.0, 200.0, len(df))

        numeric_columns = ["temperature", "humidity", "co2", "light_intensity", "plant_height"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["step"] = np.arange(len(df))

        required = ["step", "timestamp", "temperature", "humidity", "co2", "light_intensity", "plant_height"]
        return df[required].copy()

    def load_processed_csv(self) -> Optional[pd.DataFrame]:
        if not self.processed_csv_path.exists():
            return None
        try:
            df = pd.read_csv(self.processed_csv_path)
            # Re-standardize old cache files that may contain duplicate-suffixed columns
            # or string-based sensor values, then overwrite the cache with normalized data.
            if any("." in c for c in df.columns):
                df = self._standardize(df)
            elif set(["step", "timestamp", "temperature", "humidity", "co2", "light_intensity", "plant_height"]).issubset(df.columns):
                df = self._standardize(df)
            else:
                df = self._standardize(df)
            df.to_csv(self.processed_csv_path, index=False)
            return df
        except Exception as e:
            logger.error(f"Failed to load cached AGC processed CSV: {e}")
            return None

    def load_full_dataset(self, download: bool = True, sample: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Downloads (if requested), parses and standardizes AGC data."""
        cached = self.load_processed_csv()
        if cached is not None:
            df = cached
        else:
            source_root: Optional[Path] = None
            if download:
                source_root = self.download_dataset()
            if source_root is None:
                # Try existing local folder as a fallback.
                source_root = self.dataset_dir

            best_csv = self._find_best_csv(source_root)
            if best_csv is None:
                logger.error("Could not identify a suitable AGC CSV file.")
                return None

            try:
                raw = pd.read_csv(best_csv, low_memory=False)
                df = self._standardize(raw)
                df.to_csv(self.processed_csv_path, index=False)
                logger.info(f"Saved processed AGC CSV to {self.processed_csv_path}")
            except Exception as e:
                logger.error(f"Failed processing AGC dataset: {e}")
                return None

        if sample is not None and sample > 0 and len(df) > sample:
            df = df.sample(n=sample, random_state=42).sort_values("timestamp").reset_index(drop=True)
            df["step"] = np.arange(len(df))

        logger.info(f"Loaded AGC dataset records: {len(df)}")
        return df
