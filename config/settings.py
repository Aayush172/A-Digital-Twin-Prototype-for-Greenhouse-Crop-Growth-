"""
Global runtime settings and instance configurations.
"""

from config.config import SimulationConfig, ModelConfig, PathConfig
from utils.helpers import ensure_directories_exist

paths = PathConfig()
ensure_directories_exist(
    paths.data_dir,
    paths.raw_data_dir,
    paths.processed_data_dir,
    paths.results_dir,
    paths.models_dir,
    paths.figures_dir
)

sim_config = SimulationConfig()
model_config = ModelConfig()
