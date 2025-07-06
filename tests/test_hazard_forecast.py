# tests/test_hazard_forecast.py

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

# Import your hazard_forecast functions
def import_hazard_forecast():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import hazard_forecast
    return hazard_forecast

hazard_forecast = import_hazard_forecast()

# ---- Fixtures ----

@pytest.fixture
def sample_hourly_df():
    data = {
        'dt_iso': [
            datetime(2003, 6, 1, 0), datetime(2003, 6, 1, 1), datetime(2003, 6, 1, 2),  # Pre-2004
            datetime(2004, 6, 1, 0), datetime(2004, 6, 1, 1), datetime(2004, 6, 1, 2),  # 2004
            datetime(2004, 6, 2, 0), datetime(2004, 6, 2, 1),                           # 2004, another day
        ],
        'temp': [70, 71, 74, 68, 85, 81, 90, 30],
        'wind_speed': [10, 28, 29, 12, 40, 27, 15, 16],
        'rain_1h': [0, 0.5, 0, 0, 0.3, 0, 0, 1.0],
        'snow_1h': [0, 0, 0, 0, 0, 0, 0.6, 0],
        'is_thunderstorm': [False, False, False, False, True, True, True, False]
    }
    return pd.DataFrame(data)

# ---- Tests ----

def test_flag_hourly_hazards(sample_hourly_df):
    thresholds = {
        'wind_speed': 28,
        'temp_heat': 80,
        'temp_cold': 32,
        'rain_1h': 0.25,
        'snow_1h': 0.5,
        'thunderstorm': True,
    }
    df = hazard_forecast.flag_hourly_hazards(sample_hourly_df.copy(), thresholds)
    # Wind hazard on rows 1,2,4 (>=28 or 40), temp heat on 4,5,6, cold on row 7
    assert 'is_hazard' in df.columns
    assert df['is_wind_hazard'].sum() == 3
    assert df['is_heat_hazard'].sum() == 3
    assert df['is_cold_hazard'].sum() == 1
    assert df['is_rain_1h_hazard'].sum() == 3
    assert df['is_snow_1h_hazard'].sum() == 1
    assert df['is_thunderstorm_hazard'].sum() == 3

def test_flag_daily_hazards(sample_hourly_df):
    thresholds = {'wind_speed': 28, 'temp_heat': 80, 'rain_1h': 0.25, 'thunderstorm': True}
    df = hazard_forecast.flag_hourly_hazards(sample_hourly_df.copy(), thresholds)
    daily = hazard_forecast.flag_daily_hazards(df)
    # Should be 3 unique days
    assert daily.shape[0] == 3
    # Day flags
    # 2003-06-01: Any-hour (True, since wind at hour 1 and 2), Four-hour (False, only 2/3 flagged)
    d0 = daily[daily['date'] == pd.to_datetime('2003-06-01').date()].iloc[0]
    assert d0['any_hour_flag'] == True
    assert d0['four_hour_flag'] == False

def test_compute_hazard_probability_thunderstorm(sample_hourly_df):
    thresholds = {'wind_speed': 28, 'temp_heat': 80, 'rain_1h': 0.25, 'thunderstorm': True}
    df = hazard_forecast.flag_hourly_hazards(sample_hourly_df.copy(), thresholds)
    daily = hazard_forecast.flag_daily_hazards(df)
    # Probability window covers both years
    prob = hazard_forecast.compute_hazard_probability(
        daily,
        start_date="2003-06-01",
        end_date="2004-06-02",
        hazard_type="thunderstorm"
    )
    # Since only 2004 is counted for thunderstorm, there are 2 unique days in 2004
    assert set(prob['month']) == {6}
    assert prob.shape[0] == 2  # Should be 2 days (June 1 and June 2)
    assert all(prob['p_any'] <= 1.0)
    # June 1 2004: True (since 2/3 is_thunderstorm flagged)
    # p_any for June 1 should be 1.0 (flag is set for only year, 2004)
    assert prob.loc[prob['day'] == 1, 'p_any'].iloc[0] == 1.0

def test_summarize_probabilities():
    # Simple fake prob_df
    prob_df = pd.DataFrame({
        'p_any': [0.2, 0.6, 0.9],
        'p_four': [0.1, 0.4, 0.8]
    })
    result = hazard_forecast.summarize_probabilities(prob_df)
    assert isinstance(result, dict)
    assert 'mean_any' in result
    assert np.isclose(result['mean_any'], np.mean([0.2, 0.6, 0.9]) * 3)
    assert '80th_any' in result
    assert '90th_any' in result

if __name__ == "__main__":
    pytest.main([__file__])
