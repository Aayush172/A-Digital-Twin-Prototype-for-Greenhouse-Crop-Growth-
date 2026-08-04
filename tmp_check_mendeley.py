from utils.mendeley_loader import MendeleyTomatoDataset
from pathlib import Path
import sys

print('python', sys.executable)
loader = MendeleyTomatoDataset(data_dir=Path('data/raw'))
env_df = loader.load_environment_data()
print('env_df is None:', env_df is None)
if env_df is not None:
    print(env_df.head(5).to_string(index=False))
    print('rows', len(env_df))
    print('cols', list(env_df.columns))
