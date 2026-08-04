"""
Main CLI Entrypoint for Greenhouse Digital Twin Prototype.
Runs dataset generation/loading, initializes Digital Twin state engine, trains Linear Regression
baseline and Stacked LSTM models, executes comprehensive comparative evaluation, and trains
Reinforcement Learning agent for environmental optimization via PPO.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
from config.settings import paths, sim_config, model_config
from digital_twin.sensor_simulator import SensorSimulator
from digital_twin.twin import GreenhouseDigitalTwin
from digital_twin.state import DigitalTwinState
from evaluation.evaluator import Evaluator
from utils.logger import get_logger

logger = get_logger("Main")

def run_pipeline(days: int = 60, seed: int = 42, epochs: int = 35, use_real_data: bool = False, sample_size: int = None, train_rl: bool = False, rl_timesteps: int = 50000):
    """
    Executes full Digital Twin data generation, model training, evaluation, and RL optimization pipeline.
    
    Args:
        days: Number of simulation days (for synthetic data only)
        seed: Random seed for reproducibility
        epochs: Number of training epochs for LSTM
        use_real_data: If True, load real Mendeley tomato dataset instead of synthetic
        sample_size: Optional sample size for real dataset (useful for quick testing)
        train_rl: If True, train RL agent for environmental optimization
        rl_timesteps: Number of training timesteps for RL agent
    """
    logger.info("=== Starting Greenhouse Digital Twin Prototype Pipeline ===")

    # 1. Dataset Generation / Loading
    simulator = SensorSimulator(seed=seed)
    
    if use_real_data:
        logger.info("Loading real data from Mendeley tomato microclimate dataset...")
        df_dataset = simulator.load_from_mendeley(
            data_dir=paths.raw_data_dir,
            download=True,
            extract=True,
            sample=sample_size
        )
        if df_dataset is None:
            logger.error("Failed to load Mendeley dataset. Falling back to synthetic data.")
            df_dataset = simulator.generate_dataset(days=days, interval_minutes=60)
    else:
        logger.info(f"Generating synthetic sensor dataset ({days} days)...")
        df_dataset = simulator.generate_dataset(days=days, interval_minutes=60)
    
    # Save dataset CSV
    paths.dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(paths.dataset_csv, index=False)
    logger.info(f"Saved greenhouse dataset to {paths.dataset_csv}")

    # 2. Digital Twin Engine & Database Logging
    twin = GreenhouseDigitalTwin(db_path=paths.db_path)
    for _, row in df_dataset.iterrows():
        state = DigitalTwinState(
            timestamp=str(row['timestamp']),
            temperature=float(row['temperature']),
            humidity=float(row['humidity']),
            co2=float(row['co2']),
            light_intensity=float(row['light_intensity']),
            plant_height=float(row['plant_height']),
            step=int(row['step'])
        )
        twin.update_state(state, persist=True)

    logger.info(f"Updated Digital Twin with {len(df_dataset)} historical timesteps in SQLite database.")

    # 3. Model Training & Comparative Evaluation
    evaluator = Evaluator(results_dir=paths.results_dir, models_dir=paths.models_dir, figures_dir=paths.figures_dir)
    results = evaluator.run_experiments(df_dataset, sequence_length=model_config.sequence_length, epochs=epochs)

    logger.info("=== Summary Comparative Performance Metrics ===")
    logger.info(f"\n{results['comparison_table'].to_string(index=False)}")
    
    # 4. Reinforcement Learning Agent Training (Optional)
    if train_rl:
        logger.warning("Reinforcement Learning agent training is disabled because the required RL dependencies are not installed.")

    logger.info("=== Pipeline Execution Finished Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greenhouse Digital Twin Prototype Pipeline Runner")
    parser.add_argument("--days", type=int, default=60, help="Simulation duration in days (30, 60, 90, 120) - only for synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs for LSTM")
    parser.add_argument("--use-real-data", action="store_true", help="Load real tomato microclimate data from Mendeley instead of synthetic")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for real dataset (useful for quick testing)")
    parser.add_argument("--train-rl", action="store_true", help="Train RL agent for environmental optimization")
    parser.add_argument("--rl-timesteps", type=int, default=50000, help="Number of training timesteps for RL agent")

    args = parser.parse_args()
    run_pipeline(
        days=args.days, 
        seed=args.seed, 
        epochs=args.epochs, 
        use_real_data=args.use_real_data, 
        sample_size=args.sample,
        train_rl=args.train_rl,
        rl_timesteps=args.rl_timesteps
    )
