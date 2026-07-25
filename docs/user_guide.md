# Greenhouse Digital Twin Prototype - User Guide

## Overview
The Greenhouse Digital Twin software application simulates, monitors, visualises, and predicts greenhouse crop growth in real time using environmental sensor telemetry and machine learning models.

---

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (`venv` or `conda`)

### Installation Steps
```bash
cd greenhouse-digital-twin
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## How to Run

### 1. Execute End-to-End Pipeline via CLI
Run model training, dataset generation, and automated comparative evaluation:
```bash
python main.py --days 60 --seed 42 --epochs 35
```
Parameters:
- `--days`: Duration of simulation dataset (Options: 30, 60, 90, 120). Default: 60.
- `--seed`: Integer seed for pseudo-random data generation reproducibility. Default: 42.
- `--epochs`: Number of training epochs for the Stacked LSTM network. Default: 35.

### 2. Launch Interactive Streamlit Web Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

#### Features in Web Dashboard:
- **Sidebar Controls**: Adjust simulation length, random seed, active model selection, and time step playback.
- **Top KPI Cards**: Real-time snapshot of current Temperature, Relative Humidity, CO₂, PAR Light Intensity, and Plant Height.
- **Tab 1 - Digital Twin Telemetry**: Interactive multi-series charts of historical microclimate sensor streams.
- **Tab 2 - Procedural L-System Plant Growth**: Real-time 2D Plotly graphic of procedural branching structures scaling dynamically with plant height.
- **Tab 3 - Model Predictions & MSc Evaluation**: Comparative model performance metrics (MAE, RMSE, $R^2$, latency), residual plots, loss curves, and actual vs predicted growth trajectories.

### 3. Run Automated Unit Tests
```bash
pytest tests/
```
