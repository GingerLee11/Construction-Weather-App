import pandas as pd
import matplotlib.pyplot as plt

FILENAME = 'data/Historical Weather Plainview TX CLEANED.csv'
df = pd.read_csv(FILENAME, parse_dates=['dt_iso'])

# Flag thunderstorm hours
df['is_thunderstorm'] = (
    df['weather_main'].str.contains('thunderstorm', case=False, na=False) |
    df['weather_description'].str.contains('thunderstorm', case=False, na=False)
)

# Extract month/year
df['month'] = df['dt_iso'].dt.month
df['year'] = df['dt_iso'].dt.year
df['date'] = df['dt_iso'].dt.date

# 1. Thunderstorm hours per month (sum over all years)
th_hours_per_month = df.groupby('month')['is_thunderstorm'].sum()

# 2. Thunderstorm days per month (unique dates with any TS per month)
ts_day_flag = df[df['is_thunderstorm']].drop_duplicates('date')
ts_days_per_month = ts_day_flag.groupby(ts_day_flag['dt_iso'].dt.month).size()

# Plotting
plt.figure()
th_hours_per_month.plot(kind='bar')
plt.title("Thunderstorm Hours per Month (Total, All Years)")
plt.xlabel("Month")
plt.ylabel("Thunderstorm Hours")
plt.tight_layout()
plt.savefig("thunderstorm_hours_per_month.png")
plt.close()

plt.figure()
ts_days_per_month.plot(kind='bar')
plt.title("Thunderstorm Days per Month (At Least 1 Hour)")
plt.xlabel("Month")
plt.ylabel("Thunderstorm Days")
plt.tight_layout()
plt.savefig("thunderstorm_days_per_month.png")
plt.close()

print("Saved: thunderstorm_hours_per_month.png, thunderstorm_days_per_month.png")

# Earliest thunderstorm record
ts = df[df['is_thunderstorm']]
if not ts.empty:
    print("Earliest thunderstorm record:", ts['dt_iso'].min())
    print("Sample record:", ts.iloc[0])
else:
    print("No thunderstorm records found.")

# Year counts
print(ts['dt_iso'].dt.year.value_counts().sort_index())

# Thunderstorm by weather_id
df['is_thunderstorm_id'] = df['weather_id'].between(200, 299)
# Compare with is_thunderstorm
print("Thunderstorm by ID:", df['is_thunderstorm_id'].sum())
print("Earliest thunderstorm (ID):", df[df['is_thunderstorm_id']]['dt_iso'].min())

df['year'] = df['dt_iso'].dt.year
ts_by_year = df.groupby('year')['is_thunderstorm'].sum()
ts_by_year.plot(kind='bar', figsize=(10,4))
plt.title('Thunderstorm Hours by Year')
plt.show()

