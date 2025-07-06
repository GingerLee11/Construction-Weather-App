import pandas as pd

FILENAME = 'data/Historical Weather Plainview TX CLEANED.csv'
df = pd.read_csv(FILENAME, usecols=['weather_id', 'weather_main', 'weather_description'])

# Drop duplicates: each unique weather_id/main/desc combo
code_dict = df.drop_duplicates().sort_values(['weather_id', 'weather_main', 'weather_description'])

# Count occurrences of each id
code_counts = df.groupby(['weather_id', 'weather_main', 'weather_description']).size().reset_index(name='count')
code_counts = code_counts.sort_values(['weather_id', 'count'], ascending=[True, False])

# Save to CSV for easy review
code_counts.to_csv('weather_code_lookup.csv', index=False)

# Also, print out all unique codes with mapping
print(code_counts.to_string(index=False, max_rows=50))
