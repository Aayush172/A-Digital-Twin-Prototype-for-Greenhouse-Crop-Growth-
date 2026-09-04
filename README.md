# Greenhouse Digital Twin Prototype for Crop Growth Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: NumPy](https://img.shields.io/badge/Framework-NumPy-orange.svg)](https://numpy.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A research prototype developed for an **MSc Artificial Intelligence Capstone Project**, demonstrating greenhouse simulation, telemetry persistence, exploratory prediction and optional reinforcement-learning experimentation.

The repository implements a **Digital Twin Prototype** representing greenhouse environmental microclimates (Temperature, Relative Humidity, $CO_2$, PAR Light) coupled with a **Stacked LSTM Neural Network** to predict crop growth dynamics (plant height), evaluated against a **Linear Regression baseline** and rendered interactively via **Procedural L-Systems** in a **Streamlit Web Dashboard**.

---

## 🌟 Key Features
- **Simplified Digital Twin Architecture**: Real-time state synchronization, physical environment simulation engine, and SQLite historical state log.
- **Physics-Informed Crop Growth Model**: Integrates Growing Degree-Days (GDD) with stress multipliers for microclimate variables.
- **Procedural Plant Visualisation**: Lindenmayer system (L-System) string expansion rendered into interactive 2D Plotly structures driven by Digital Twin state.
- **Sequence Model vs Baseline ML Pipeline**: Custom NumPy recurrent sequence model benchmarked against a Scikit-Learn Linear Regression baseline on identical sliding-window inputs.
- **Interactive Streamlit Web Dashboard**: Timeline playback, metric KPI cards, live telemetry charts, procedural growth renderer, and comparative evaluation plots.
- **Comprehensive Unit Testing**: Automated PyTest test suite verifying simulator, Digital Twin engine, preprocessor, ML models, and metrics.

---

## 📂 Project Structure
```
greenhouse-digital-twin/
├── main.py                     # Root CLI entrypoint for running complete pipeline
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
├── README.md                   # Repository documentation
│
├── config/                     # Configuration dataclasses & settings
│   ├── config.py
│   └── settings.py
│
├── digital_twin/               # Digital Twin engine & environment dynamics
│   ├── state.py                # Digital Twin state representation
│   ├── environment.py          # Physics-informed microclimate simulation
│   ├── sensor_simulator.py     # Multi-day telemetry generator & CSV loader
│   └── twin.py                 # Core Digital Twin engine & SQLite persistence
│
├── plant/                      # Crop biology & procedural visualization
│   ├── plant.py                # Physiological crop growth model
│   ├── lsystem.py              # Lindenmayer production rules string builder
│   └── renderer.py             # Plotly geometric plant structure renderer
│
├── models/                     # Machine learning models & preprocessor
│   ├── preprocess.py           # Time-series sliding window preprocessor
│   ├── linear_regression_model.py # Baseline Linear Regression model wrapper
│   ├── lstm_model.py           # Custom NumPy recurrent sequence model wrapper
│   ├── trainer.py              # Model fitting orchestrator
│   └── predictor.py            # Real-time single-step and horizon predictor
│
├── evaluation/                 # Metrics & publication plot generation
│   ├── metrics.py              # MAE, RMSE, R², training/inference latency
│   ├── comparison.py           # Tabular side-by-side comparative summary
│   ├── plots.py                # Publication-quality plot generator
│   └── evaluator.py            # Automated experiment orchestrator
│
├── dashboard/                  # Streamlit Web Application
│   ├── app.py                  # Main Streamlit web application
│   └── components/             # Modular dashboard views
│
├── utils/                      # Logging and helper utilities
├── tests/                      # PyTest automated unit test suite
├── docs/                       # User guide, developer guide, API docs
└── results/                    # Exported models, CSV summaries, and figures
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/your-username/greenhouse-digital-twin.git
cd greenhouse-digital-twin
python -m venv venv

# Activate Virtual Environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Pipeline & Experiments via CLI
Execute the end-to-end simulation, model training, and comparative evaluation:

**Using Generated Telemetry (Default, for simulation and development)**:
```bash
python main.py --days 60 --seed 42 --epochs 35
```

**Using the Mendeley Tomato Microclimate Dataset for the reported experiments**:
```bash
# Load the local processed Mendeley dataset
python main.py --use-real-data --epochs 50

# Quick test with sampled data (1000 records)
python main.py --use-real-data --sample 1000

# Full pipeline with real data + custom parameters
python main.py --use-real-data --seed 123 --epochs 40
```

For detailed instructions on using real data, see [REAL_DATA_USAGE.md](docs/REAL_DATA_USAGE.md).

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Run Unit Test Suite
```bash
pytest tests/
```

---

## 📜 Research & Dissertation
This repository accompanies the MSc Artificial Intelligence Capstone dissertation:
> **"Digital Twin for Greenhouse Crop Growth Prediction Using L-System Modelling, Sensor Integration, and Reinforcement Learning"**

The full ~11,062-word dissertation document with complete literature review, mathematical derivations, architecture diagrams, IEEE citations, and critical discussion is provided in the repository workspace.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
