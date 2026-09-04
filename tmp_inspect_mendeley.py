from pathlib import Path
import pandas as pd
root = Path('data/raw/Microclimate monitoring in commercial tomato (Solanum Lycopersicum L.) greenhouse production and its effect on plant growth, yield and fruit quality dataset/T&RH')
files = []
for p in sorted(root.glob('*')):
    if p.is_dir():
        files.extend(sorted(p.glob('*.xlsx')))
for path in files:
    try:
        df = pd.read_excel(path)
        print(path, '->', list(df.columns))
    except Exception as e:
        print('ERR', path, e)
