import pandas as pd

FILENAME = 'data/Historical Weather Plainview TX CLEANED.csv'
df = pd.read_csv(FILENAME, parse_dates=['dt_iso'])

# Check nulls
print("Null % wind_speed:", df['wind_speed'].isnull().mean()*100)
print("Null % wind_gust:", df['wind_gust'].isnull().mean()*100)

# Describe stats
print("\nWind_speed stats:\n", df['wind_speed'].describe())
print("\nWind_gust stats:\n", df['wind_gust'].describe())

# Gust < speed?
if 'wind_gust' in df.columns:
    bad = df[(df['wind_gust'].notnull()) & (df['wind_gust'] < df['wind_speed'])]
    print(f"\nRows where wind_gust < wind_speed: {len(bad)}")
    if len(bad):
        print(bad[['dt_iso', 'wind_speed', 'wind_gust']].head())

# Top wind events
print("\nTop 10 wind_speed:\n", df.nlargest(10, 'wind_speed')[['dt_iso','wind_speed','wind_gust']])
if 'wind_gust' in df.columns:
    print("\nTop 10 wind_gust:\n", df.nlargest(10, 'wind_gust')[['dt_iso','wind_speed','wind_gust']])
