# data_loader.py

import pandas as pd
import logging
import os

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/data_loader.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

DATA_PATH = "data/Historical Weather Plainview TX CLEANED.csv"
OUT_PATH = "data/daily_aggregated.csv"


def load_hourly_csv(path=DATA_PATH):
    log.info(f"Loading hourly CSV from {path}...")
    df = pd.read_csv(path, low_memory=False, parse_dates=['dt_iso'])
    log.info(f"Loaded data shape: {df.shape}")
    return df


def add_thunderstorm_flag(df):
    df['is_thunderstorm'] = (
        df['weather_main'].str.contains('thunderstorm', case=False, na=False) |
        df['weather_description'].str.contains('thunderstorm', case=False, na=False)
    )
    log.info(f"Total thunderstorm hours: {df['is_thunderstorm'].sum()}")
    return df


def aggregate_daily(df):
    log.info("Aggregating to daily...")
    df['date'] = df['dt_iso'].dt.date
    agg_funcs = {
        'temp': 'mean',
        'temp_min': 'min',
        'temp_max': 'max',
        'wind_speed': 'max',
        'wind_gust': 'max',
        'rain_1h': ['sum', 'max'],
        'rain_3h': ['sum', 'max'],
        'snow_1h': ['sum', 'max'],
        'snow_3h': ['sum', 'max'],
        'is_thunderstorm': 'any',
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    daily = df.groupby('date').agg(agg_funcs)
    daily = daily.reset_index()

    # Rename columns for clarity
    rename_map = {
        'temp_mean': 'temp_mean',
        'temp_min_min': 'temp_min',
        'temp_max_max': 'temp_max',
        'wind_speed_max': 'wind_speed_max',
        'wind_gust_max': 'wind_gust_max',
        'rain_1h_sum': 'rain_1h_total',
        'rain_1h_max': 'rain_1h_max',
        'rain_3h_sum': 'rain_3h_total',
        'rain_3h_max': 'rain_3h_max',
        'snow_1h_sum': 'snow_1h_total',
        'snow_1h_max': 'snow_1h_max',
        'snow_3h_sum': 'snow_3h_total',
        'snow_3h_max': 'snow_3h_max',
        'is_thunderstorm_any': 'thunderstorm_day'
    }
    # Fix MultiIndex columns
    daily.columns = [
        col[0] if isinstance(col, str) else '_'.join(col)
        if isinstance(col, tuple) else str(col)
        for col in daily.columns.values
    ]
    daily = daily.rename(columns=rename_map)
    log.info(f"Aggregated daily data shape: {daily.shape}")
    log.info(f"Sample daily rows:\n{daily.head()}")
    return daily

def main():
    df = load_hourly_csv()
    df = add_thunderstorm_flag(df)
    daily = aggregate_daily(df)
    daily.to_csv(OUT_PATH, index=False)
    log.info(f"Saved daily aggregated data to {OUT_PATH}")
    return daily


if __name__ == "__main__":
    main()
