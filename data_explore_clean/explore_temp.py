import pandas as pd
import matplotlib.pyplot as plt

FILENAME = 'data\Historical Weather Plainview TX CLEANED.csv'

df = pd.read_csv(FILENAME, parse_dates=['dt_iso'])

# Select columns of interest
temp_cols = ['dt_iso', 'temp', 'dew_point', 'feels_like', 'temp_min', 'temp_max']
df_temp = df[temp_cols]

# How often does temp != temp_min or temp_max?
min_diff = (df_temp['temp'] != df_temp['temp_min']).sum()
max_diff = (df_temp['temp'] != df_temp['temp_max']).sum()

print(f"Rows where temp != temp_min: {min_diff}")
print(f"Rows where temp != temp_max: {max_diff}")

# Show samples where different
print("\nSample rows where temp != temp_min:")
print(df_temp[df_temp['temp'] != df_temp['temp_min']].head())

print("\nSample rows where temp != temp_max:")
print(df_temp[df_temp['temp'] != df_temp['temp_max']].head())

# Describe stats for all temp columns
print("\nColumn stats:")
print(df_temp.describe())


# --- 1. Histograms ---
for col in temp_cols:
    plt.figure()
    df[col].hist(bins=50)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.savefig(f'temp_hist_{col}.png')
    plt.close()

# --- 2. Scatter plots ---
plt.figure()
plt.scatter(df['temp'], df['temp_min'], alpha=0.2, s=2)
plt.xlabel('temp')
plt.ylabel('temp_min')
plt.title('temp vs temp_min')
plt.savefig('scatter_temp_vs_temp_min.png')
plt.close()

plt.figure()
plt.scatter(df['temp'], df['temp_max'], alpha=0.2, s=2)
plt.xlabel('temp')
plt.ylabel('temp_max')
plt.title('temp vs temp_max')
plt.savefig('scatter_temp_vs_temp_max.png')
plt.close()

plt.figure()
plt.scatter(df['temp'], df['feels_like'], alpha=0.2, s=2)
plt.xlabel('temp')
plt.ylabel('feels_like')
plt.title('temp vs feels_like')
plt.savefig('scatter_temp_vs_feels_like.png')
plt.close()