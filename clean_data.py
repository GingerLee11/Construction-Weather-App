import pandas as pd
import logging
import os

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/clean_data.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

FILENAME = 'Historical Weather Plainview TX.csv'
OUTFILE = 'Historical Weather Plainview TX CLEANED.csv'

log.info("Reading data file...")
df = pd.read_csv(FILENAME, low_memory=False)

# 1. Drop 100% missing columns
missing_pct = df.isnull().mean()
cols_drop = missing_pct[missing_pct == 1.0].index.tolist()
df = df.drop(columns=cols_drop)
log.info(f"Dropped columns with 100% missing: {cols_drop}")

# 2. Parse datetime
df['dt_iso'] = df['dt_iso'].str.replace(' UTC', '', regex=False)
df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

# 3. Check duplicates by dt_iso
dups = df[df.duplicated(subset=['dt_iso'], keep=False)]
if not dups.empty:
    log.warning(f"Duplicate dt_iso timestamps: {dups['dt_iso'].nunique()} unique times, {len(dups)} rows total.")

# 4. Handle duplicates
log.info("Aggregating duplicates if needed...")
grouped = df.groupby('dt_iso', as_index=False)

def aggregate_group(grp):
    # If all rows identical, just keep first
    if (grp.nunique() == 1).all():
        return grp.iloc[0]
    # If not identical: Aggregate
    # Numeric columns: take max (hazard), otherwise first (categorical)
    num_cols = grp.select_dtypes(include='number').columns
    agg = {}
    for col in grp.columns:
        if col in num_cols:
            agg[col] = grp[col].max()
        else:
            agg[col] = grp[col].iloc[0]
    log.warning(f"Aggregated non-identical duplicate at {grp['dt_iso'].iloc[0]}")
    return pd.Series(agg)

cleaned = grouped.apply(aggregate_group)
cleaned = cleaned.reset_index(drop=True)
log.info(f"Cleaned data shape: {cleaned.shape}")

# 5. Save cleaned data
cleaned.to_csv(OUTFILE, index=False)
log.info(f"Saved cleaned data to: {OUTFILE}")

log.info("Clean_data.py finished successfully.")
