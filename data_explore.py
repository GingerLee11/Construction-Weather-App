import pandas as pd
import logging
import os

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_explore.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

FILENAME = 'Historical Weather Plainview TX.csv'
log.info("Reading CSV...")
df = pd.read_csv(FILENAME, low_memory=False)

# Robust datetime parsing (handles trailing ' UTC')
if 'dt_iso' in df.columns:
    df['dt_iso'] = df['dt_iso'].str.replace(' UTC', '', regex=False)
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

log.info(f"Columns & Types:\n{df.dtypes}")
log.info(f"First 3 rows:\n{df.head(3)}")
log.info(f"Last 3 rows:\n{df.tail(3)}")

# Time span
if 'dt_iso' in df.columns:
    log.info(f"Date Range: First date: {df['dt_iso'].min()}, Last date: {df['dt_iso'].max()}")
    log.info(f"Total records: {len(df)}")
    log.info(f"Unique dates: {df['dt_iso'].dt.date.nunique()}")
    log.info(f"Unique hours: {df['dt_iso'].nunique()}")

# Missing data
missing = df.isnull().mean().sort_values(ascending=False) * 100
log.info(f"Missing Values (%):\n{missing}")

# Numeric summary
num_cols = ['temp','wind_speed','wind_gust','rain_1h','rain_3h','snow_1h','snow_3h']
num_cols = [col for col in num_cols if col in df.columns]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if num_cols:
    log.info(f"Numeric Column Summary:\n{df[num_cols].describe()}")
    for col in num_cols:
        nonzero = df[col].gt(0).sum()
        log.info(f"{col} - Nonzero count: {nonzero}")
        log.info(f"{col} stats:\n{df[col].describe()}")
else:
    log.warning('No numeric weather columns found!')

# Weather main/desc value counts
if 'weather_main' in df.columns:
    log.info(f"Weather Main (top 10):\n{df['weather_main'].value_counts().head(10)}")
if 'weather_description' in df.columns:
    log.info(f"Weather Description (top 10):\n{df['weather_description'].value_counts().head(10)}")

# Duplicates in dt_iso
if 'dt_iso' in df.columns:
    dup_dt = df['dt_iso'][df['dt_iso'].duplicated()]
    if not dup_dt.empty:
        log.warning(f"Duplicate dt_iso found! Count: {dup_dt.count()}\n{dup_dt}")

    # Gaps in data
    df_sorted = df.sort_values('dt_iso')
    dt_diff = df_sorted['dt_iso'].diff().dt.total_seconds().dropna()
    gaps = dt_diff[dt_diff > 3600]
    if not gaps.empty:
        log.warning(f"Gaps larger than 1hr between records:\n{gaps}")
    else:
        log.info("No hour gaps detected in dt_iso.")

# Print a few rows with high wind/rain/snow
if 'wind_speed' in df.columns:
    wind_rows = df[df['wind_speed'].fillna(0) > 25][['dt_iso','wind_speed','wind_gust']].head()
    log.info(f"High wind_speed (>25 mph):\n{wind_rows}")
if 'rain_1h' in df.columns:
    rain_rows = df[df['rain_1h'].fillna(0) > 0.5][['dt_iso','rain_1h','rain_3h']].head()
    log.info(f"Rain 1h > 0.5 in:\n{rain_rows}")
if 'snow_1h' in df.columns:
    snow_rows = df[df['snow_1h'].fillna(0) > 0.1][['dt_iso','snow_1h','snow_3h']].head()
    log.info(f"Snow 1h > 0.1 in:\n{snow_rows}")

# Sample row with high hazard
if set(['wind_speed', 'rain_1h', 'snow_1h']).issubset(df.columns):
    hazard = (df['wind_speed'].fillna(0) > 25) | (df['rain_1h'].fillna(0) > 0.5) | (df['snow_1h'].fillna(0) > 0.1)
    hazard_row = df[hazard].head(1).T
    log.info(f"Sample complete row with high hazard:\n{hazard_row}")

log.info("Data exploration complete.")
