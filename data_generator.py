import os
import sys

# Add project root to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yfinance as yf
import numpy as np

from market_bm.utils import get_quantizer, TICKERS, START_DATE, END_DATE

# =============================================================================
# Data Loading
# =============================================================================

prices = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)["Adj Close"]

returns = prices.pct_change().dropna(how="all")
log_returns = np.log(prices).diff().dropna(how="all")

# =============================================================================
# Create Gray-coded Dataset with (today, tomorrow) pairs
# =============================================================================

import pandas as pd

quantizer = get_quantizer(returns)
binary_df, states_df = quantizer.encode_dataframe(returns[TICKERS])

# Create today/tomorrow paired dataset
# Rename columns: AAPL_g0 -> AAPL_today_g0
today_cols = {col: col.replace("_g", "_today_g") for col in binary_df.columns}
today_df = binary_df.rename(columns=today_cols)

# Shift by -1 to get tomorrow's values, rename: AAPL_g0 -> AAPL_tomorrow_g0
tomorrow_cols = {col: col.replace("_g", "_tomorrow_g") for col in binary_df.columns}
tomorrow_df = binary_df.shift(-1).rename(columns=tomorrow_cols)

# Concatenate today and tomorrow, drop last row (no tomorrow data)
paired_df = pd.concat([today_df, tomorrow_df], axis=1).iloc[:-1].astype(np.uint8)

print(f"Paired dataset shape: {paired_df.shape}")
print(f"  - {len(paired_df)} samples (today, tomorrow pairs)")
print(f"  - {len(TICKERS)} assets × {quantizer.nbits} bits × 2 (today+tomorrow) = {paired_df.shape[1]} columns")
print(f"Date range: {paired_df.index[0]} to {paired_df.index[-1]}")

# Split into train (up to 2025-12-31) and test (2026 onwards)
SPLIT_DATE = "2025-12-31"
train_df = paired_df[paired_df.index <= SPLIT_DATE]
test_df = paired_df[paired_df.index > SPLIT_DATE]

print(f"\nTrain: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
print(f"Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")

# Export to CSV (without date index)
data_dir = os.path.dirname(__file__)
train_df.to_csv(os.path.join(data_dir, "data.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
print(f"\nSaved to: {data_dir}/data.csv, {data_dir}/test.csv")
