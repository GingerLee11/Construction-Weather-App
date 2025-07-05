import pandas as pd

FILENAME = 'Historical Weather Plainview TX.csv'
df = pd.read_csv(FILENAME, low_memory=False)

# Robust datetime parsing (handles trailing ' UTC')
if 'dt_iso' in df.columns:
    df['dt_iso'] = df['dt_iso'].str.replace(' UTC', '', regex=False)
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

print('\n--- Columns & Types ---')
print(df.dtypes)
print('\n--- First 3 rows ---')
print(df.head(3))
print('\n--- Last 3 rows ---')
print(df.tail(3))

# Time span
if 'dt_iso' in df.columns:
    print('\n--- Date Range ---')
    print(f"First date: {df['dt_iso'].min()}, Last date: {df['dt_iso'].max()}")
    print(f"Total records: {len(df)}")
    print(f"Unique dates: {df['dt_iso'].dt.date.nunique()}")
    print(f"Unique hours: {df['dt_iso'].nunique()}")

# Missing data
print('\n--- Missing Values (%) ---')
print(df.isnull().mean().sort_values(ascending=False) * 100)

# Numeric summary
num_cols = ['temp','wind_speed','wind_gust','rain_1h','rain_3h','snow_1h','snow_3h']
num_cols = [col for col in num_cols if col in df.columns]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if num_cols:
    print('\n--- Numeric Column Summary ---')
    print(df[num_cols].describe())
    for col in num_cols:
        print(f'\n{col} - Nonzero count: {df[col].gt(0).sum()}')
        print(df[col].describe())
else:
    print('No numeric weather columns found!')

# Weather main/desc value counts
if 'weather_main' in df.columns:
    print('\n--- Weather Main (top 10) ---')
    print(df['weather_main'].value_counts().head(10))
if 'weather_description' in df.columns:
    print('\n--- Weather Description (top 10) ---')
    print(df['weather_description'].value_counts().head(10))

# Duplicates in dt_iso
if 'dt_iso' in df.columns:
    dup_dt = df['dt_iso'][df['dt_iso'].duplicated()]
    if not dup_dt.empty:
        print('\n--- Duplicate dt_iso found! ---')
        print(dup_dt)

    # Gaps in data
    df_sorted = df.sort_values('dt_iso')
    dt_diff = df_sorted['dt_iso'].diff().dt.total_seconds().dropna()
    gaps = dt_diff[dt_diff > 3600]
    print('\n--- Gaps larger than 1hr between records ---')
    print(gaps)

# Print a few rows with high wind/rain/snow
if 'wind_speed' in df.columns:
    print('\n--- High wind_speed (>25 mph) ---')
    print(df[df['wind_speed'].fillna(0) > 25][['dt_iso','wind_speed','wind_gust']].head())
if 'rain_1h' in df.columns:
    print('\n--- Rain 1h > 0.5 in ---')
    print(df[df['rain_1h'].fillna(0) > 0.5][['dt_iso','rain_1h','rain_3h']].head())
if 'snow_1h' in df.columns:
    print('\n--- Snow 1h > 0.1 in ---')
    print(df[df['snow_1h'].fillna(0) > 0.1][['dt_iso','snow_1h','snow_3h']].head())

# Sample row with high hazard
if set(['wind_speed', 'rain_1h', 'snow_1h']).issubset(df.columns):
    hazard = (df['wind_speed'].fillna(0) > 25) | (df['rain_1h'].fillna(0) > 0.5) | (df['snow_1h'].fillna(0) > 0.1)
    print('\n--- Sample complete row with high hazard ---')
    print(df[hazard].head(1).T)
