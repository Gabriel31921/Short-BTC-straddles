import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd

import math
import datetime

from datetime import datetime

start = "2025-02-25"
start_time = "2025-02-25 23:00:00"
expiration_time = "2025-02-26 08:00:00"
start_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
end_ms = int(pd.Timestamp(expiration_time).timestamp() * 1000)

BTC = yf.Ticker("BTC-USD").history(start=start, auto_adjust=False, interval="5m")
BTC.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
BTC.reset_index(inplace=True)

BTC["Timestamp"] = [date.timestamp() * 1000 for date in BTC["Datetime"]]
filtered_BTC = BTC[(BTC['Timestamp'] >= start_ms) & (BTC['Timestamp'] <= end_ms)]
filtered_BTC = filtered_BTC.reset_index(drop=True).copy()

print("Filtered Bitcoin Data:")
print(filtered_BTC.head())

# Calculate Parkinson Volatility
def calculate_parkinson_volatility(data, window=None, annualization_factor=252):
    """
    Calculate Parkinson volatility using high and low prices.
    
    Parameters:
    - data: DataFrame with High and Low columns
    - window: Number of periods to use for calculation (None = all data)
    - annualization_factor: Factor to annualize volatility (252 for daily, 12 for monthly, etc.)
    
    Returns:
    - Parkinson volatility value
    """
    if window is None:
        window = len(data)
    
    # Calculate log(High/Low)^2
    log_hl_squared = (np.log(data['High'] / data['Low']) ** 2)
    
    # Parkinson's estimator uses a scaling factor of 1/(4 * ln(2))
    scaling_factor = 1 / (4 * np.log(2))
    
    # Calculate the sum of scaled log_hl_squared values
    sum_scaled = scaling_factor * log_hl_squared.sum()
    
    # Calculate daily variance
    daily_variance = sum_scaled / window
    
    # Calculate daily volatility
    daily_volatility = np.sqrt(daily_variance)
    
    # Annualize volatility
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor)
    
    return daily_volatility, annualized_volatility

# Calculate time-based annualization factor
def calculate_annualization_factor(data, interval_minutes=5):
    """
    Calculate appropriate annualization factor based on data interval.
    For 5-minute data: 
    - Minutes in a year: 525,600 (365 days * 24 hours * 60 minutes)
    - Number of 5-minute intervals in a year: 525,600 / 5 = 105,120
    """
    minutes_in_year = 525600  # 365 days * 24 hours * 60 minutes
    return minutes_in_year / interval_minutes

# Calculate the annualization factor for 5-minute data
annualization_factor = calculate_annualization_factor(filtered_BTC, interval_minutes=5)

# Calculate Parkinson volatility for the filtered data
daily_vol, annualized_vol = calculate_parkinson_volatility(
    filtered_BTC, 
    window=None, 
    annualization_factor=annualization_factor
)

# Print results
print("\nParkinson Volatility Results:")
print(f"Number of observations: {len(filtered_BTC)}")
print(f"Time period: {start_time} to {expiration_time}")
print(f"Daily volatility: {daily_vol:.6f}")
print(f"Annualized volatility: {annualized_vol:.6f}")
print(f"Annualized volatility (%): {annualized_vol * 100:.2f}%")

# Plot the data and volatility
plt.figure(figsize=(12, 8))

# Plot 1: Price Chart
plt.subplot(2, 1, 1)
plt.plot(filtered_BTC['Datetime'], filtered_BTC['Close'], label='Close Price')
plt.fill_between(filtered_BTC['Datetime'], filtered_BTC['Low'], filtered_BTC['High'], alpha=0.2, color='blue')
plt.title(f'BTC-USD Price ({start_time} to {expiration_time})')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()

# Plot 2: High-Low Range
plt.subplot(2, 1, 2)
high_low_range = filtered_BTC['High'] - filtered_BTC['Low']
plt.bar(filtered_BTC['Datetime'], high_low_range, color='red', alpha=0.6)
plt.title(f'High-Low Range (Used for Parkinson Volatility)')
plt.ylabel('Range (USD)')
plt.grid(True)

plt.figtext(0.5, 0.01, f"Parkinson Volatility: {annualized_vol * 100:.2f}%", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()