import pandas as pd

FILENAME = 'data/Historical Weather Plainview TX CLEANED.csv'
df = pd.read_csv(FILENAME, parse_dates=['dt_iso'])

cols = ['rain_1h', 'rain_3h', 'snow_1h', 'snow_3h']

for col in cols:
    print(f"\n{col}:")
    print("  % Null:", df[col].isnull().mean()*100)
    print("  Nonzero count:", (df[col].fillna(0) > 0).sum())
    print("  Stats:\n", df[col].describe())
    print("  Top 5 events:\n", df.nlargest(5, col)[['dt_iso', col]])
