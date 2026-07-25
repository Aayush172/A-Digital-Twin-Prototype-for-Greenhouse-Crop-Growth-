# Greenhouse Digital Twin Prototype - Developer Guide

## Architecture Overview
The system follows a clean object-oriented design adhering to SOLID software engineering principles:

```
greenhouse-digital-twin/
├── config/             # Dataclass configurations & path resolution
├── digital_twin/       # State representation, physics simulator, SQLite persistence
├── plant/              # Biological growth functions, L-System string builder, Plotly renderer
├── models/             # Time-series sliding window preprocessor, LR baseline, Stacked LSTM
├── evaluation/         # Metrics (MAE, RMSE, R2, latency), comparison engine, plot generator
├── dashboard/          # Streamlit UI & modular component renderers
├── tests/              # Comprehensive PyTest unit test suite
├── results/            # Saved model artifacts, exported CSV summaries, figure files
└── main.py             # CLI runner script
```

---

## Modifying Modules

### Adding New Environmental Sensors
1. Edit `digital_twin/state.py` to add new fields to `DigitalTwinState`.
2. Update physical equations in `digital_twin/environment.py`.
3. Add column mapping to `digital_twin/sensor_simulator.py`.
4. Update `models/preprocess.py` `feature_cols` list.

### Customising the LSTM Architecture
Edit `models/lstm_model.py`:
- Modify layer counts, hidden unit dimensions, or activation functions inside `_build_model()`.
- Adjust callbacks (e.g. LearningRateScheduler, TensorBoard) inside `fit()`.

### Customising Procedural L-System Rules
Edit `plant/lsystem.py`:
- Modify replacement rules dictionary (e.g. `{"X": "F[+X][-X]FX"}`).
- Update branching angles inside `plant/renderer.py`.

---

## Database Schema (SQLite)
Table `twin_telemetry`:
- `step` INTEGER PRIMARY KEY
- `timestamp` TEXT
- `temperature` REAL
- `humidity` REAL
- `co2` REAL
- `light_intensity` REAL
- `plant_height` REAL
- `created_at` TIMESTAMP
