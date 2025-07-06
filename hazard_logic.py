# hazard_logic.py

import pandas as pd
import numpy as np
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/hazard_logic.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

# -------------------------
# 1. Hazard Flagging
# -------------------------

def flag_hourly_hazards(df, thresholds, mode="any"):
    log.info("Flagging hourly hazards with thresholds: %s", thresholds)
    df = df.copy()
    df['is_wind_hazard'] = df['wind_speed'] >= thresholds.get('wind_speed', 28)
    df['is_heat_hazard'] = df['temp'] >= thresholds.get('temp_heat', 80)
    df['is_cold_hazard'] = df['temp'] <= thresholds.get('temp_cold', 32)
    df['is_rain_1h_hazard'] = df['rain_1h'].fillna(0) >= thresholds.get('rain_1h', 0.25)
    df['is_rain_3h_hazard'] = df['rain_3h'].fillna(0) >= thresholds.get('rain_3h', 1.0)
    df['is_snow_1h_hazard'] = df['snow_1h'].fillna(0) >= thresholds.get('snow_1h', 0.5)
    df['is_snow_3h_hazard'] = df['snow_3h'].fillna(0) >= thresholds.get('snow_3h', 1.5)
    df['is_thunderstorm_hazard'] = df['is_thunderstorm'] if thresholds.get('thunderstorm', True) else False

    hazard_cols = [
        'is_wind_hazard', 'is_heat_hazard', 'is_cold_hazard',
        'is_rain_1h_hazard', 'is_rain_3h_hazard',
        'is_snow_1h_hazard', 'is_snow_3h_hazard',
        'is_thunderstorm_hazard'
    ]

    selected_cols = [col for col, key in zip(hazard_cols, thresholds.keys()) if thresholds.get(key, None) is not None]
    if not selected_cols:
        selected_cols = hazard_cols

    if mode == "all":
        df['is_hazard'] = df[selected_cols].all(axis=1)
        log.info("Flagging with ALL hazards (AND).")
    else:
        df['is_hazard'] = df[selected_cols].any(axis=1)
        log.info("Flagging with ANY hazard (OR).")
    log.info("Flagged %d of %d hours as hazard.", df['is_hazard'].sum(), len(df))
    return df

# -------------------------
# 2. Daily Hazard Flags
# -------------------------

def flag_daily_hazards(df_hourly):
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
    log.info("Daily hazard flags calculated for %d days.", len(daily))
    log.info("Sample flagged days:\n%s", daily.head().to_string())
    return daily

# -------------------------
# 3. Historical Probability Calculation
# -------------------------

def compute_hazard_probability(daily, start_date, end_date, hazard_type="any"):
    df = daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Restrict to years with data (e.g., thunderstorms: 2004+)
    if hazard_type == "thunderstorm":
        before = len(df)
        df = df[df['year'] >= 2004]
        log.info("Restricted to years >=2004 for thunderstorm (was %d rows, now %d rows).", before, len(df))

    window = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    df_window = df[window]
    log.info("Probability window: %s to %s (%d days)", start_date, end_date, len(df_window))

    # Compute per calendar day (month, day)
    result = []
    for (month, day), grp in df_window.groupby(['month', 'day']):
        n_years = grp['year'].nunique()
        any_years = grp[grp['any_hour_flag']]['year'].nunique()
        four_years = grp[grp['four_hour_flag']]['year'].nunique()
        p_any = any_years / n_years if n_years else np.nan
        p_four = four_years / n_years if n_years else np.nan
        result.append({'month': month, 'day': day, 'p_any': p_any, 'p_four': p_four})
        log.info("Month %02d, Day %02d: %d years, p_any=%.3f, p_four=%.3f", month, day, n_years, p_any, p_four)
    log.info("Probability calculation complete for %d days.", len(result))
    return pd.DataFrame(result)

# -------------------------
# 4. Summary (optional, for reporting)
# -------------------------

def summarize_probabilities(prob_df):
    days = len(prob_df)
    mean_any = prob_df['p_any'].mean() * days
    mean_four = prob_df['p_four'].mean() * days
    pct80_any = np.percentile(prob_df['p_any'].dropna(), 80) * days
    pct90_any = np.percentile(prob_df['p_any'].dropna(), 90) * days
    log.info("Summary: mean_any=%.2f, mean_four=%.2f, 80th_any=%.2f, 90th_any=%.2f",
             mean_any, mean_four, pct80_any, pct90_any)
    return {
        'mean_any': mean_any,
        'mean_four': mean_four,
        '80th_any': pct80_any,
        '90th_any': pct90_any
    }

# -------------------------
# Usage Example
# -------------------------

if __name__ == "__main__":
    # Example thresholds
    thresholds = {
        'wind_speed': 28,
        'temp_heat': 80,
        'temp_cold': 32,
        'rain_1h': 0.25,
        'rain_3h': 1.0,
        'snow_1h': 0.5,
        'snow_3h': 1.5,
        'thunderstorm': True,
    }
    # # Sample loading (uncomment and supply your file)
    # df = pd.read_csv("data/Historical Weather Plainview TX CLEANED.csv", parse_dates=['dt_iso'])
    # df = flag_hourly_hazards(df, thresholds)
    # daily = flag_daily_hazards(df)
    # prob = compute_hazard_probability(daily, "2023-06-01", "2023-06-10", hazard_type="thunderstorm")
    # print(prob)
    pass
