# Greenhouse Digital Twin Prototype - API Documentation

## Module: `digital_twin`

### Class: `DigitalTwinState`
- `timestamp`: string (ISO-8601)
- `temperature`: float (°C)
- `humidity`: float (%)
- `co2`: float (ppm)
- `light_intensity`: float (μmol/m²/s)
- `plant_height`: float (cm)
- `step`: int
- `to_dict() -> Dict`: Serialises state to dictionary.
- `to_json() -> str`: Serialises state to JSON.

### Class: `GreenhouseDigitalTwin`
- `__init__(db_path: Optional[Path])`: Instantiates Digital Twin engine and SQLite database connection.
- `update_state(new_state: DigitalTwinState, persist: bool = True)`: Synchronises new state observation.
- `load_history_from_db() -> pd.DataFrame`: Fetches historical telemetry log from database.

---

## Module: `plant`

### Class: `LSystemGenerator`
- `generate_string(iterations: int) -> str`: Expands Lindenmayer axiom across specified depth.
- `get_iteration_depth_from_height(plant_height_cm: float) -> int`: Returns iteration depth based on plant height.

### Class: `PlantRenderer`
- `render_plotly_2d(plant_height_cm: float) -> go.Figure`: Generates interactive 2D structure plot.

---

## Module: `models`

### Class: `DataPreprocessor`
- `fit_transform(df: pd.DataFrame) -> pd.DataFrame`: Cleans, imputes, and scales feature variables.
- `create_sequences(df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`: Generates 3D sliding-window tensor for LSTM.
- `create_tabular_lagged_features(df_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`: Flattens sequences for Linear Regression baseline.

### Class: `GreenhouseLSTMModel`
- `fit(X_train, y_train, val_data, epochs, batch_size) -> Dict`: Trains Stacked Keras LSTM network.
- `predict(X) -> np.ndarray`: Performs time-series forecasting.
