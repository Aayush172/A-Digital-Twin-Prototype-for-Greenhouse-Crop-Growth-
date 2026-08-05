# Using Real Mendeley Tomato Microclimate Dataset

Your greenhouse digital twin project now supports loading **real microclimate data** from the Mendeley tomato dataset instead of using only generated telemetry.

## Dataset Information

**Dataset**: Microclimate monitoring in commercial tomato (Solanum Lycopersicum L.) greenhouse production

- **Source**: Mendeley Data - https://data.mendeley.com/datasets/tkbkzdt5nr/2
- **Collection Period**: 2018-2020
- **Collection Site**: Proefcentrum Hoogstraaten (Belgium)
- **Project**: GROW! (Interreg VI-NL)

### Data Contents

**Environment Measurements** (hourly):
- Temperature (°C)
- Relative Humidity (%)
- Light intensity / PAR (µmol/m²/s)

**Plant Measurements** (weekly/as-needed):
- Truss weight
- Truss number
- Firmness
- Color
- Soluble solids content
- Ethylene emission rate
- Stem length (cm)
- Stem diameter (mm)

## How to Use Real Data

### Option 1: Command Line (Recommended)

Run the pipeline with real data:

```bash
# Load real data with full pipeline
python main.py --use-real-data

# Load real data and sample 1000 records (for quick testing)
python main.py --use-real-data --sample 1000

# Combine with other parameters
python main.py --use-real-data --epochs 50 --seed 123
```

### Option 2: Python Script

```python
from digital_twin.sensor_simulator import SensorSimulator
from config.settings import paths
import pandas as pd

# Create simulator
simulator = SensorSimulator(seed=42)

# Load real data from Mendeley
df_dataset = simulator.load_from_mendeley(
    data_dir=paths.raw_data_dir,
    download=True,      # Download if not present
    extract=True,       # Extract ZIP file
    sample=None         # Set to integer for sampling
)

# Use like any other dataset
print(f"Loaded {len(df_dataset)} records")
print(df_dataset.head())
```

### Option 3: Manual Data Loader

```python
from utils.mendeley_loader import MendeleyTomatoDataset
from pathlib import Path

loader = MendeleyTomatoDataset(data_dir=Path("data/mendeley"))

# Download and extract
loader.download_dataset()
loader.extract_dataset()

# Load individual datasets
env_df = loader.load_environment_data()  # Temperature, humidity, light
plant_df = loader.load_plant_data()       # Plant measurements

# Create unified dataset
unified_df = loader.create_unified_dataset(env_df, plant_df)

# Save to CSV
loader.save_to_csv(unified_df)
```

## Data Processing Details

### Column Standardization

The loader automatically standardizes column names from various formats:

**Environment Data**:
- `Temp`, `Temperature`, `temp` → `temperature`
- `RH`, `Humidity` → `humidity`
- `PAR`, `Light` → `light_intensity`

**Timestamps**:
- `Date`, `Time`, `Timestamp` → `timestamp`

### Missing Data Handling

- **Missing timestamp**: Auto-generated sequence
- **Missing CO2**: Defaults to 400 ppm (atmospheric)
- **Missing plant_height**: Derived from stem_length or interpolated
- **Missing values**: Filled using forward/backward fill

### Data Merge Strategy

Environment data (hourly) and plant data (weekly) are merged using nearest-neighbor matching on timestamp. This creates a unified time-series where:
- Hourly environmental conditions are preserved
- Plant measurements are matched to their nearest timestamp
- Missing values are intelligently imputed

## File Structure

After first run with real data:

```
data/mendeley/
├── tkbkzdt5nr.zip                    # Downloaded dataset (46 MB)
├── extracted/
│   ├── Environment/
│   │   ├── info.txt
│   │   ├── measurements_1.csv
│   │   └── measurements_2.csv
│   ├── PlantData/
│   │   ├── info.txt
│   │   ├── truss_data.csv
│   │   └── stem_measurements.csv
│   └── README.md
└── mendeley_dataset_processed.csv    # Final standardized dataset
```

## Advantages of Real Data

✅ **Authentic greenhouse conditions** from commercial tomato production  
✅ **Multi-year data** capturing seasonal variations  
✅ **Complete measurement sets** for validation and benchmarking  
✅ **Professional data collection** with standardized protocols  
✅ **Published & citable** for academic work  

## Performance Expectations

Real data characteristics:

- **Dataset size**: ~30,000+ records per year
- **Timestamp coverage**: 2018-2020 (3+ years of continuous data)
- **Environmental ranges**:
  - Temperature: 15-30°C
  - Humidity: 40-95%
  - Light: 0-800 µmol/m²/s
- **Plant growth**: 5-200+ cm stem length

Models trained on real data may show:
- Different error patterns vs. generated telemetry
- Seasonal trends and anomalies
- More realistic environmental fluctuations

## Troubleshooting

### Download Issues

If download fails (firewall, large file):

```python
# Try with smaller file or alternative
# Check Mendeley website directly:
# https://data.mendeley.com/datasets/tkbkzdt5nr/2
```

### Memory Issues

For very large datasets:

```python
# Sample data for testing
df = simulator.load_from_mendeley(sample=5000)

# Or load specific time periods manually
loader = MendeleyTomatoDataset()
env_df = loader.load_environment_data()
# Process and filter in chunks
```

### Data Quality Issues

Check loaded data:

```python
df = simulator.load_from_mendeley()
print(f"Records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Missing values:\n{df.isnull().sum()}")
```

## Citation

If using this dataset for research:

```bibtex
@data{Salagovic2024,
  author = {Salagovic, Jakub and Vanhees, Dorien and Verboven, Pieter and Holsteens, Kristof and Verlinden, Bert and Huysmans, Marlies and Van de Poel, Bram and Nicolai, Bart},
  year = {2024},
  title = {Microclimate monitoring in commercial tomato (Solanum Lycopersicum L.) greenhouse production and its effect on plant growth, yield and fruit quality dataset},
  publisher = {Mendeley Data},
  version = {V2},
  doi = {10.17632/tkbkzdt5nr.2}
}
```

## References

- Dataset: https://data.mendeley.com/datasets/tkbkzdt5nr/2
- GROW! Project: https://www.grensregio.eu/projecten/grow
- Proefcentrum Hoogstraaten: https://www.proefcentrumhoogstraaten.be/
