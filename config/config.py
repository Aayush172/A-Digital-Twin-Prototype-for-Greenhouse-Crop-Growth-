"""
Centralized Configuration for Digital Twin, Simulation, and Machine Learning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from utils.constants import (
    DEFAULT_SIMULATION_DAYS,
    DEFAULT_TIME_STEP_MINUTES,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
)

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class SimulationConfig:
    simulation_days: int = DEFAULT_SIMULATION_DAYS
    time_step_minutes: int = DEFAULT_TIME_STEP_MINUTES
    random_seed: int = RANDOM_SEED
    noise_level: float = 0.05

@dataclass
class ModelConfig:
    sequence_length: int = SEQUENCE_LENGTH
    train_split: float = 0.8
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    lstm_units: list = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2

@dataclass
class PathConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    dataset_csv: Path = BASE_DIR / "data" / "greenhouse_dataset.csv"
    db_path: Path = BASE_DIR / "data" / "digital_twin.db"
    results_dir: Path = BASE_DIR / "results"
    models_dir: Path = BASE_DIR / "results" / "models"
    figures_dir: Path = BASE_DIR / "results" / "figures"
