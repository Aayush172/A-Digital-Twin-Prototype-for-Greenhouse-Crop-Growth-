"""
Automated Evaluation Pipeline.
Runs Experiment 1 (LR Baseline), Experiment 2 (LSTM Network), and Experiment 3 (Comparison),
generating evaluation summaries and publication figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from models.preprocess import DataPreprocessor
from models.trainer import ModelTrainer
from evaluation.metrics import compute_regression_metrics
from evaluation.comparison import ModelComparison
from evaluation.plots import plot_actual_vs_predicted, plot_residuals, plot_training_loss
from utils.logger import get_logger

logger = get_logger("Evaluator")

class Evaluator:
    """Orchestrates comparative empirical experiments."""

    def __init__(self, results_dir: Path, models_dir: Path, figures_dir: Path):
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.figures_dir = figures_dir
        self.trainer = ModelTrainer(models_dir)

    def run_experiments(self, raw_df: pd.DataFrame, sequence_length: int = 24,
                        epochs: int = 35, batch_size: int = 32) -> Dict[str, Any]:
        """Runs Experiments 1, 2, and 3 on input sensor dataset."""
        logger.info("--- Starting Empirical Evaluation Pipeline ---")
        
        # Preprocessing
        preprocessor = DataPreprocessor(sequence_length=sequence_length)
        df_scaled = preprocessor.fit_transform(raw_df)

        # 3D Sequence formulation for LSTM
        X_seq, y_seq = preprocessor.create_sequences(df_scaled)
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = preprocessor.split_train_test(X_seq, y_seq)

        # 2D Tabular formulation for Linear Regression
        X_tab, y_tab = preprocessor.create_tabular_lagged_features(df_scaled)
        X_train_tab, X_test_tab, y_train_tab, y_test_tab = preprocessor.split_train_test(X_tab, y_tab)

        # -------------------------------------------------------------
        # Experiment 1: Linear Regression Baseline
        # -------------------------------------------------------------
        logger.info("Executing Experiment 1: Baseline Linear Regression...")
        lr_model = self.trainer.train_baseline_linear_regression(X_train_tab, y_train_tab)
        lr_preds_scaled = lr_model.predict(X_test_tab)
        
        # De-scale predictions and ground truth to cm scale
        y_test_cm = preprocessor.inverse_transform_target(y_test_seq)
        lr_preds_cm = preprocessor.inverse_transform_target(lr_preds_scaled)
        metrics_lr = compute_regression_metrics(y_test_cm, lr_preds_cm, lr_model.training_time, lr_model.inference_time)
        logger.info(f"Exp 1 Baseline LR Metrics: {metrics_lr}")

        # -------------------------------------------------------------
        # Experiment 2: Stacked LSTM Neural Network
        # -------------------------------------------------------------
        logger.info("Executing Experiment 2: Stacked LSTM Model...")
        val_split_idx = int(len(X_train_seq) * 0.85)
        X_tr, X_val = X_train_seq[:val_split_idx], X_train_seq[val_split_idx:]
        y_tr, y_val = y_train_seq[:val_split_idx], y_train_seq[val_split_idx:]

        lstm_model = self.trainer.train_lstm_model(
            X_tr, y_tr, val_data=(X_val, y_val), epochs=epochs, batch_size=batch_size
        )
        lstm_preds_scaled = lstm_model.predict(X_test_seq)
        lstm_preds_cm = preprocessor.inverse_transform_target(lstm_preds_scaled)
        metrics_lstm = compute_regression_metrics(y_test_cm, lstm_preds_cm, lstm_model.training_time, lstm_model.inference_time)
        logger.info(f"Exp 2 LSTM Metrics: {metrics_lstm}")

        # -------------------------------------------------------------
        # Experiment 3: Model Comparison & Results Artifact Export
        # -------------------------------------------------------------
        logger.info("Executing Experiment 3: Comparative Analysis & Plot Generation...")
        df_comparison = ModelComparison.create_comparison_dataframe(metrics_lr, metrics_lstm)
        
        csv_export_path = self.results_dir / "metrics_summary.csv"
        df_comparison.to_csv(csv_export_path, index=False)
        logger.info(f"Exported metrics summary table to {csv_export_path}")

        # Generate plots
        plot_actual_vs_predicted(y_test_cm, lr_preds_cm, lstm_preds_cm, save_path=self.figures_dir / "actual_vs_predicted.png")
        plot_residuals(y_test_cm, lr_preds_cm, lstm_preds_cm, save_path=self.figures_dir / "residuals_comparison.png")
        if lstm_model.history:
            plot_training_loss(lstm_model.history, save_path=self.figures_dir / "lstm_training_loss.png")

        return {
            "preprocessor": preprocessor,
            "lr_model": lr_model,
            "lstm_model": lstm_model,
            "comparison_table": df_comparison,
            "metrics_lr": metrics_lr,
            "metrics_lstm": metrics_lstm,
            "y_test_cm": y_test_cm,
            "lr_preds_cm": lr_preds_cm,
            "lstm_preds_cm": lstm_preds_cm
        }
