from pathlib import Path
from utils.mendeley_loader import MendeleyTomatoDataset
loader = MendeleyTomatoDataset(data_dir=Path('data/raw'))
df = loader.load_environment_data()
print('loaded', df is not None)
print('rows', len(df) if df is not None else None)
print('cols', list(df.columns) if df is not None else None)
print(df.head(2).to_string(index=False) if df is not None else 'NONE')
