# tests/test_data_loader.py
import os
import pandas as pd
import pytest
from datetime import datetime

# Import data_loader functions
def import_data_loader():
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import data_loader
    return data_loader

data_loader = import_data_loader()

# --- Fixtures ---
@pytest.fixture
def sample_hourly_df():
    # Small hourly dataframe, 2 days, covers all columns of interest
    data = {
        'dt_iso': [
            datetime(2023, 6, 1, 0), datetime(2023, 6, 1, 1), datetime(2023, 6, 1, 2),
            datetime(2023, 6, 2, 0), datetime(2023, 6, 2, 1),
        ],
        'temp': [70, 71, 74, 68, 69],
        'temp_min': [69, 68, 72, 67, 68],
        'temp_max': [72, 74, 76, 70, 71],
        'wind_speed': [10, 15, 5, 20, 18],
        'wind_gust': [12, 20, 7, 24, 22],
        'rain_1h': [0.1, 0.2, 0, 0, 0.3],
        'rain_3h': [0.1, None, 0.4, None, 0.6],
        'snow_1h': [0, 0, 0, 0, 0],
        'snow_3h': [None, None, None, None, None],
        'weather_main': ['Clear', 'Thunderstorm', 'Rain', 'Clear', 'Thunderstorm'],
        'weather_description': ['clear sky', 'thunderstorm', 'light rain', 'clear sky', 'thunderstorm'],
    }
    df = pd.DataFrame(data)
    return df

# --- Tests ---
def test_add_thunderstorm_flag(sample_hourly_df):
    df = data_loader.add_thunderstorm_flag(sample_hourly_df.copy())
    assert 'is_thunderstorm' in df.columns
    assert df['is_thunderstorm'].sum() == 2
    assert df.loc[1, 'is_thunderstorm'] == True
    assert df.loc[4, 'is_thunderstorm'] == True

def test_aggregate_daily(sample_hourly_df):
    df = data_loader.add_thunderstorm_flag(sample_hourly_df.copy())
    daily = data_loader.aggregate_daily(df)
    # Should have 2 days
    assert daily.shape[0] == 2
    # Check columns exist
    cols = daily.columns.tolist()
    assert 'temp_min' in cols
    assert 'temp_max' in cols
    assert 'wind_speed_max' in cols
    assert 'rain_1h_total' in cols
    assert 'rain_1h_max' in cols
    assert 'thunderstorm_day' in cols
    # Day 1 checks
    d1 = daily.iloc[0]
    assert d1['temp_mean'] == pytest.approx((70 + 71 + 74) / 3)
    assert d1['temp_min'] == min([69, 68, 72])
    assert d1['temp_max'] == max([72, 74, 76])
    assert d1['wind_speed_max'] == 15
    assert d1['rain_1h_total'] == pytest.approx(0.1 + 0.2 + 0)
    assert d1['rain_1h_max'] == 0.2
    assert d1['thunderstorm_day'] == True
    # Day 2 checks
    d2 = daily.iloc[1]
    assert d2['thunderstorm_day'] == True

if __name__ == "__main__":
    pytest.main([__file__])
