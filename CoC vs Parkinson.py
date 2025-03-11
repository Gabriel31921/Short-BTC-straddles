import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import math
import datetime
from datetime import datetime

# Define time range
start = "2025-03-10"
start_time = "2025-03-10 23:00:00"
expiration_time = "2025-03-11 08:00:00"
start_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
end_ms = int(pd.Timestamp(expiration_time).timestamp() * 1000)

# Get BTC data
BTC = yf.Ticker("BTC-USD").history(start=start, auto_adjust=False, interval="1m")
BTC.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
BTC.reset_index(inplace=True)

# Convert datetime to timestamp for filtering
BTC["Timestamp"] = [date.timestamp() * 1000 for date in BTC["Datetime"]]
filtered_BTC = BTC[(BTC['Timestamp'] >= start_ms) & (BTC['Timestamp'] <= end_ms)]
filtered_BTC = filtered_BTC.reset_index(drop=True).copy()

print("Filtered Bitcoin Data:")
print(filtered_BTC.head())

# Calculate time-based annualization factor
def calculate_annualization_factor(interval_minutes=5):
    """
    Calculate appropriate annualization factor based on data interval.
    For 5-minute data: 
    - Minutes in a year: 525,600 (365 days * 24 hours * 60 minutes)
    - Number of 5-minute intervals in a year: 525,600 / 5 = 105,120
    """
    minutes_in_year = 525600  # 365 days * 24 hours * 60 minutes
    return minutes_in_year / interval_minutes

# Calculate Parkinson Volatility
def calculate_parkinson_volatility(data, window=None, annualization_factor=252):
    """
    Calculate Parkinson volatility using high and low prices.
    
    Parameters:
    - data: DataFrame with High and Low columns
    - window: Number of periods to use for calculation (None = all data)
    - annualization_factor: Factor to annualize volatility
    
    Returns:
    - Parkinson volatility value (daily and annualized)
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

# Calculate Close-to-Close Volatility
def calculate_close_to_close_volatility(data, window=None, annualization_factor=252):
    """
    Calculate Close-to-Close volatility using log returns.
    
    Parameters:
    - data: DataFrame with Close prices
    - window: Number of periods to use for calculation (None = all data)
    - annualization_factor: Factor to annualize volatility
    
    Returns:
    - Close-to-Close volatility value (daily and annualized)
    """
    if window is None:
        window = len(data) - 1  # Need n-1 for returns
    
    # Calculate log returns
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    # Calculate variance of log returns
    variance = log_returns.var()
    
    # Calculate daily volatility
    daily_volatility = np.sqrt(variance)
    
    # Annualize volatility
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor)
    
    return daily_volatility, annualized_volatility

# Calculate the annualization factor for 5-minute data
annualization_factor = calculate_annualization_factor(interval_minutes=5)

# Calculate Parkinson volatility for the filtered data
parkinson_daily_vol, parkinson_annualized_vol = calculate_parkinson_volatility(
    filtered_BTC, 
    window=None, 
    annualization_factor=annualization_factor
)

# Calculate Close-to-Close volatility for the filtered data
close_daily_vol, close_annualized_vol = calculate_close_to_close_volatility(
    filtered_BTC, 
    window=None, 
    annualization_factor=annualization_factor
)

# Print results
print("\nVolatility Comparison Results:")
print(f"Number of observations: {len(filtered_BTC)}")
print(f"Time period: {start_time} to {expiration_time}")
print("\nParkinson Volatility:")
print(f"Daily: {parkinson_daily_vol:.6f}")
print(f"Annualized: {parkinson_annualized_vol:.6f}")
print(f"Annualized (%): {parkinson_annualized_vol * 100:.2f}%")
print("\nClose-to-Close Volatility:")
print(f"Daily: {close_daily_vol:.6f}")
print(f"Annualized: {close_annualized_vol:.6f}")
print(f"Annualized (%): {close_annualized_vol * 100:.2f}%")
print(f"\nDifference (Parkinson - Close): {(parkinson_annualized_vol - close_annualized_vol) * 100:.2f}%")
print(f"Ratio (Parkinson/Close): {(parkinson_annualized_vol / close_annualized_vol):.2f}x")

# Calculate log returns for plotting
filtered_BTC['log_returns'] = np.log(filtered_BTC['Close'] / filtered_BTC['Close'].shift(1)).fillna(0)
filtered_BTC['high_low_range'] = filtered_BTC['High'] - filtered_BTC['Low']

# Plot the data and volatilities
plt.figure(figsize=(14, 12))

# Plot 1: Price Chart
plt.subplot(3, 1, 1)
plt.plot(filtered_BTC['Datetime'], filtered_BTC['Close'], label='Close Price')
plt.fill_between(filtered_BTC['Datetime'], filtered_BTC['Low'], filtered_BTC['High'], alpha=0.2, color='blue')
plt.title(f'BTC-USD Price ({start_time} to {expiration_time})')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()

# Plot 2: High-Low Range (Parkinson)
plt.subplot(3, 1, 2)
plt.bar(filtered_BTC['Datetime'], filtered_BTC['high_low_range'], color='green', alpha=0.6, label='High-Low Range')
plt.title('High-Low Range (Used for Parkinson Volatility)')
plt.ylabel('Range (USD)')
plt.grid(True)
plt.legend()
plt.axhline(y=filtered_BTC['high_low_range'].mean(), color='red', linestyle='--', label='Mean Range')
plt.annotate(f'Parkinson Vol: {parkinson_annualized_vol * 100:.2f}%', 
             xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3))

# Plot 3: Log Returns (Close-to-Close)
plt.subplot(3, 1, 3)
plt.bar(filtered_BTC['Datetime'][1:], filtered_BTC['log_returns'][1:], color='purple', alpha=0.6, label='Log Returns')
plt.title('Log Returns (Used for Close-to-Close Volatility)')
plt.ylabel('Log Return')
plt.grid(True)
plt.legend()
plt.axhline(y=0, color='black', linestyle='-')
plt.axhline(y=filtered_BTC['log_returns'][1:].mean(), color='red', linestyle='--', label='Mean Return')
plt.annotate(f'Close-to-Close Vol: {close_annualized_vol * 100:.2f}%', 
             xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="purple", alpha=0.3))

# Add comparison text
plt.figtext(0.5, 0.01, 
            f"Volatility Comparison: Parkinson ({parkinson_annualized_vol * 100:.2f}%) vs Close-to-Close ({close_annualized_vol * 100:.2f}%)\n" +
            f"Difference: {(parkinson_annualized_vol - close_annualized_vol) * 100:.2f}% | Ratio: {(parkinson_annualized_vol / close_annualized_vol):.2f}x", 
            ha="center", fontsize=12, bbox={"facecolor":"yellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.show()