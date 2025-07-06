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

def flag_hourly_hazards(df, thresholds):
    """
    Flags hourly hazards based on thresholds.
    Thunderstorms are NOT included.
    """
    df = df.copy()
    df['is_wind_hazard'] = df['wind_speed'] >= thresholds.get('wind_speed', 28)
    df['is_heat_hazard'] = df['temp'] >= thresholds.get('temp_heat', 80)
    df['is_cold_hazard'] = df['temp'] <= thresholds.get('temp_cold', 32)
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
    Aggregates to daily hazard flags: any_hour and four_hour.
    """
    df_hourly = df_hourly.copy()
    df_hourly['date'] = pd.to_datetime(df_hourly['dt_iso']).dt.date
    group = df_hourly.groupby('date')['is_hazard']
    daily = pd.DataFrame({
        'any_hour_flag': group.any(),
        'four_hour_flag': group.sum() >= 4
    }).reset_index()
    daily['year'] = pd.to_datetime(daily['date']).dt.year
    daily['month'] = pd.to_datetime(daily['date']).dt.month
    daily['day'] = pd.to_datetime(daily['date']).dt.day
    log.info("Aggregated daily hazard flags for %d days.", len(daily))
    return daily

def forecast_hazards(daily, start_date, end_date, min_year=1979, max_year=2024):
    """
    For each day in the given (future) window, computes historical probability of hazard.
    For Jan 2, 2025, uses Jan 2 of all past years 1979–2024.
    Returns DataFrame [date, p_any, p_four, n_years].
    """
    window_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    out = []
    for dt in window_dates:
        m, d = dt.month, dt.day
        mask = (daily['month'] == m) & (daily['day'] == d) & \
               (daily['year'] >= min_year) & (daily['year'] <= max_year)
        subset = daily[mask]
        n_years = subset['year'].nunique()
        p_any = subset['any_hour_flag'].mean() if n_years > 0 else np.nan
        p_four = subset['four_hour_flag'].mean() if n_years > 0 else np.nan
        out.append({
            'date': dt.date(),
            'p_any': p_any,
            'p_four': p_four,
            'n_years': n_years
        })
        log.info(
            "Forecast for %s: n_years=%d, p_any=%.3f, p_four=%.3f",
            dt.date(), n_years, p_any if n_years else np.nan, p_four if n_years else np.nan
        )
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
        # No thunderstorm!
    }
    df = pd.read_csv("data/Historical Weather Plainview TX CLEANED.csv", parse_dates=['dt_iso'])
    log.info("Loaded %d rows from hourly data.", len(df))

    df = flag_hourly_hazards(df, thresholds)
    daily = flag_daily_hazards(df)

    # Example: Forecast for Jan 1–Jan 10, 2025
    forecast = forecast_hazards(
        daily, start_date="2025-01-01", end_date="2025-01-10", min_year=1979, max_year=2024
    )
    print(forecast)
    log.info("Forecast window complete.")
