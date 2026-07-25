"""
Evaluation Plot Generator.
Generates publication-grade figures: Prediction Trajectories, Residual Distributions,
Training Loss Curves, and Environmental Feature Heatmaps.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

def save_and_show_figure(fig, filepath: Path) -> None:
    """Saves figure to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred_lr: np.ndarray,
                             y_pred_lstm: np.ndarray, save_path: Optional[Path] = None) -> plt.Figure:
    """Plots actual vs predicted plant growth trajectories for LR and LSTM."""
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = np.arange(len(y_true))

    ax.plot(time_steps, y_true, label='Ground Truth Height (cm)', color='#1f77b4', linewidth=2.5)
    ax.plot(time_steps, y_pred_lr, label='Linear Regression Baseline', color='#ff7f0e', linestyle='--', linewidth=1.8)
    ax.plot(time_steps, y_pred_lstm, label='Stacked LSTM Model', color='#2ca02c', linestyle='-', linewidth=2.0)

    ax.set_title("Greenhouse Crop Growth Prediction: Model Trajectory Comparison", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Test Time Step (Hours)", fontsize=12)
    ax.set_ylabel("Plant Height (cm)", fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

    if save_path:
        save_and_show_figure(fig, save_path)
    return fig

def plot_residuals(y_true: np.ndarray, y_pred_lr: np.ndarray,
                   y_pred_lstm: np.ndarray, save_path: Optional[Path] = None) -> plt.Figure:
    """Plots residual distributions for Linear Regression and LSTM models."""
    res_lr = y_true - y_pred_lr
    res_lstm = y_true - y_pred_lstm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot of residuals vs fitted
    ax1.scatter(y_pred_lr, res_lr, alpha=0.6, color='#ff7f0e', label='LR Baseline Residuals', s=20)
    ax1.scatter(y_pred_lstm, res_lstm, alpha=0.6, color='#2ca02c', label='LSTM Residuals', s=20)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title("Residuals vs Predicted Values", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predicted Plant Height (cm)")
    ax1.set_ylabel("Residual Error (cm)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Residual Histogram / KDE
    ax2.hist(res_lr, bins=25, alpha=0.5, color='#ff7f0e', label='Linear Regression', density=True)
    ax2.hist(res_lstm, bins=25, alpha=0.5, color='#2ca02c', label='LSTM Network', density=True)
    ax2.set_title("Residual Error Probability Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Residual Error (cm)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_and_show_figure(fig, save_path)
    return fig

def plot_training_loss(history_dict: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    """Plots training and validation loss curves across training epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history_dict['loss']) + 1)
    ax.plot(epochs, history_dict['loss'], label='Training Loss (MSE)', color='#1f77b4', linewidth=2.0)
    if 'val_loss' in history_dict:
        ax.plot(epochs, history_dict['val_loss'], label='Validation Loss (MSE)', color='#d62728', linestyle='--', linewidth=2.0)

    ax.set_title("LSTM Model Training & Validation Loss Convergence", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

    if save_path:
        save_and_show_figure(fig, save_path)
    return fig
