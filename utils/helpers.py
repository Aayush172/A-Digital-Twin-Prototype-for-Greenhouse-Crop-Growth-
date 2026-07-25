"""
Helper utilities for data manipulation, formatting, and file management.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def ensure_directories_exist(*paths: Path) -> None:
    """Ensures directories exist on disk."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def format_metrics_dict(metrics: dict) -> str:
    """Formats metrics dictionary into clean tabular string."""
    lines = [f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}" for k, v in metrics.items()]
    return "\n".join(lines)
