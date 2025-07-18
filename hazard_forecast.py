# hazard_forecast.py

import pandas as pd
import numpy as np
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/hazard_forecast.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()


def filter_working_hours(df, work_start=7, work_end=17):
    """Keep only rows within working hours (e.g., 7:00–16:59)."""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['dt_iso']).dt.hour
    filtered = df[(df['hour'] >= work_start) & (df['hour'] < work_end)]
    log.info("Filtered to working hours %02d:00–%02d:00; %d rows remain.", work_start, work_end, len(filtered))
    return filtered

def flag_hourly_hazards(df, thresholds):
    """
    Flags hourly hazards based on thresholds.
    Uses feels_like for cold/heat.
    """
    df = df.copy()
    df['is_wind_hazard'] = df['wind_speed'] >= thresholds.get('wind_speed', 28)
    df['is_heat_hazard'] = df['feels_like'] >= thresholds.get('temp_heat', 80)
    df['is_cold_hazard'] = df['feels_like'] <= thresholds.get('temp_cold', 32)
    df['is_rain_1h_hazard'] = df['rain_1h'].fillna(0) >= thresholds.get('rain_1h', 0.25)
    df['is_rain_3h_hazard'] = df['rain_3h'].fillna(0) >= thresholds.get('rain_3h', 1.0)
    df['is_snow_1h_hazard'] = df['snow_1h'].fillna(0) >= thresholds.get('snow_1h', 0.5)
    df['is_snow_3h_hazard'] = df['snow_3h'].fillna(0) >= thresholds.get('snow_3h', 1.5)
    df['is_hazard'] = df[
        [
            'is_wind_hazard', 'is_heat_hazard', 'is_cold_hazard',
            'is_rain_1h_hazard', 'is_rain_3h_hazard',
            'is_snow_1h_hazard', 'is_snow_3h_hazard'
        ]
    ].any(axis=1)
    log.info("Flagged hazards for %d hourly rows.", len(df))
    return df

def flag_daily_hazards(df_hourly):
    """
    For each day, count hazard hours by type.
    Returns a daily DataFrame with columns: date, year, month, day, [hazard_hr...], total_hazard_hr.
    """
    df_hourly = df_hourly.copy()
    df_hourly['date'] = pd.to_datetime(df_hourly['dt_iso']).dt.date

    hazard_types = [
        'is_wind_hazard', 'is_heat_hazard', 'is_cold_hazard',
        'is_rain_1h_hazard', 'is_rain_3h_hazard',
        'is_snow_1h_hazard', 'is_snow_3h_hazard'
    ]

    agg = {
        h: 'sum' for h in hazard_types
    }
    agg['is_hazard'] = 'sum'  # Total hours any hazard

    daily = df_hourly.groupby('date').agg(agg).reset_index()
    daily['year'] = pd.to_datetime(daily['date']).dt.year
    daily['month'] = pd.to_datetime(daily['date']).dt.month
    daily['day'] = pd.to_datetime(daily['date']).dt.day
    log.info("Aggregated daily hazard hours for %d days.", len(daily))
    return daily

def forecast_hazards(daily, start_date, end_date, min_year=1979, max_year=2024):
    """
    For each day in forecast window, compute mean hazard hours by type across all years.
    """
    hazard_cols = [
        'is_wind_hazard', 'is_heat_hazard', 'is_cold_hazard',
        'is_rain_1h_hazard', 'is_rain_3h_hazard',
        'is_snow_1h_hazard', 'is_snow_3h_hazard', 'is_hazard'
    ]
    window_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    out = []
    for dt in window_dates:
        m, d = dt.month, dt.day
        subset = daily[(daily['month'] == m) & (daily['day'] == d) &
                       (daily['year'] >= min_year) & (daily['year'] <= max_year)]
        n_years = subset['year'].nunique()
        stats = {col.replace("is_", "").replace("_hazard", "_hr"): subset[col].mean() if n_years > 0 else np.nan
                 for col in hazard_cols}
        row = {
            'date': dt.date(),
            'n_years': n_years,
            **stats
        }
        out.append(row)
        log.info("Forecast for %s: %s", dt.date(), row)
    return pd.DataFrame(out)

if __name__ == "__main__":
    # --- Usage Example ---
    thresholds = {
        'wind_speed': 28,
        'temp_heat': 80,
        'temp_cold': 32,
        'rain_1h': 0.25,
        'rain_3h': 1.0,
        'snow_1h': 0.5,
        'snow_3h': 1.5,
    }
    # User can change these as needed
    work_start = 7
    work_end = 17

    df = pd.read_csv("data/Historical Weather Plainview TX CLEANED.csv", parse_dates=['dt_iso'])
    log.info("Loaded %d rows from hourly data.", len(df))

    df = filter_working_hours(df, work_start=work_start, work_end=work_end)
    df = flag_hourly_hazards(df, thresholds)
    daily = flag_daily_hazards(df)
    forecast = forecast_hazards(daily, start_date="2025-01-01", end_date="2025-01-10")
    print(forecast)
    log.info("Forecast window complete.")
