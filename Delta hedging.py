import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import math
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.optimize import brentq
from scipy.stats import norm

# Setup parameters
start = "2025-03-10"
start_time = "2025-03-10 23:00:00"
expiration_time = "2025-03-11 08:00:00"
start_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
end_ms = int(pd.Timestamp(expiration_time).timestamp() * 1000)

# Fetch BTC data
BTC = yf.Ticker("BTC-USD").history(start=start, interval="1m")
BTC.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
BTC.reset_index(inplace=True)
BTC["Timestamp"] = [date.timestamp() * 1000 for date in BTC["Datetime"]]
filtered_BTC = BTC[(BTC['Timestamp'] >= start_ms) & (BTC['Timestamp'] <= end_ms)]
filtered_BTC = filtered_BTC.reset_index(drop=True).copy()

# Option parameters
strike = 79500          # Assume a fixed strike (e.g. ATM based on initial price)
r = 0.375               # Risk-free rate (for simplicity)
sigma = 0.5670          # Fixed implied volatility (static IV)
expiration = pd.Timestamp(expiration_time, tz="UTC")  # Option expiration time
threshold = 0.15         # Delta threshold for hedging
reduction_multiplier = 1.1  # Multiplier for when delta is decreasing
increase_multiplier = 1.1

# Define Black–Scholes delta functions (for call and put)
def bs_call_delta(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def bs_put_delta(S, K, T, r, sigma):
    return bs_call_delta(S, K, T, r, sigma) - 1

# Initialize variables for the simulation:
cumulative_hedge_pnl = 0.0   # Will accumulate the PNL from hedge trades
hedge_shares = 0.0           # Current hedge position (starts at zero)
prev_price = None            # Underlying price at the start of the interval
prev_net_delta = None        # Previous net delta to compare with current
hedge_trades = []            # Track when and how much we're trading for the hedge

# Create lists to store data for plotting
prices = []
timestamps = []
net_deltas = []
pnl_history = []
hedge_positions = []
theoretical_positions = []   # What position would be if we just set it to -net_delta

# Loop over each time step in the filtered BTC data (each minute):
for i, row in filtered_BTC.iterrows():
    # Get the current time and underlying price:
    current_time = pd.to_datetime(row["Datetime"])
    S = row["Close"]
    
    # Store price and timestamp for plotting
    prices.append(S)
    timestamps.append(current_time)
    
    # Compute time to expiration in years:
    T_remaining = (expiration - current_time).total_seconds() / (365 * 24 * 3600)
    if T_remaining <= 0:
        break  # Stop the simulation if expiration has passed
    
    # Calculate option deltas (using static sigma) with updated T and S:
    call_delta = bs_call_delta(S, strike, T_remaining, r, sigma)
    put_delta = bs_put_delta(S, strike, T_remaining, r, sigma)
    
    # For a short straddle (short one call and one put), net delta is:
    net_delta = -(call_delta + put_delta)
    net_deltas.append(net_delta)
    
    # Calculate theoretical position (what it would be if we just set to -net_delta)
    theoretical_position = -net_delta
    theoretical_positions.append(theoretical_position)
    
    # Handle first iteration
    if prev_price is None:
        prev_price = S
        prev_net_delta = net_delta
        # Initialize hedge based on threshold - only hedge if above threshold
        if abs(net_delta) > threshold:
            hedge_shares = -net_delta  # Initial hedge matches delta exactly
            hedge_trades.append(hedge_shares)
        else:
            hedge_shares = 0  # No initial hedge if below threshold
            hedge_trades.append(0)
        
        hedge_positions.append(hedge_shares)
        pnl_history.append(0)
        continue
    
    # Calculate PnL from previous hedge before any adjustment
    pnl_trade = hedge_shares * (S - prev_price)
    cumulative_hedge_pnl += pnl_trade
    
    # Determine hedge adjustment based on delta change
    current_abs_delta = abs(net_delta)
    prev_abs_delta = abs(prev_net_delta)
    
    # Track hedge trade for this step (default to zero)
    hedge_trade = 0
    
    # Check if we need to completely close the hedge
    if current_abs_delta < threshold:
        # Close entire position if below threshold
        hedge_trade = -hedge_shares  # Trade to close is opposite of current position
        hedge_shares = 0  # Position is now zero
    else:
        # Calculate delta change
        delta_change = -net_delta - (-prev_net_delta)  # Change in theoretical hedge position
        
        if abs(net_delta) > abs(prev_net_delta):
            # Delta is increasing - add the exact delta change
            hedge_trade = delta_change * increase_multiplier
            hedge_shares += delta_change * increase_multiplier
        else:
            # Delta is decreasing - remove more than the delta change
            hedge_trade = delta_change * reduction_multiplier
            hedge_shares += delta_change * reduction_multiplier
            # Make sure we don't over-hedge in the opposite direction
            if np.sign(hedge_shares) != np.sign(-net_delta) and abs(hedge_shares) > 0.01:
                hedge_shares = 0  # Reset to zero if we cross over

    # Record the hedge trade
    hedge_trades.append(hedge_trade)
    
    # Store hedge position for plotting
    hedge_positions.append(hedge_shares)
    
    # Record PnL
    pnl_history.append(cumulative_hedge_pnl)
    
    # Update values for next iteration
    prev_price = S
    prev_net_delta = net_delta

# Output the cumulative hedging PnL after the simulation:
print("Cumulative Hedging PnL:", cumulative_hedge_pnl)

# Create a figure with subplots
plt.figure(figsize=(14, 18))

# Plot 1: BTC Price over time
plt.subplot(3, 1, 1)
plt.plot(timestamps, prices)
plt.title('BTC Price')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.grid(True)

# Plot 2: Cumulative PnL over time
plt.subplot(3, 1, 2)
plt.plot(timestamps[:len(pnl_history)], pnl_history)
plt.title('Cumulative Hedging PnL')
plt.xlabel('Time')
plt.ylabel('PnL (USD)')
plt.grid(True)

# Plot 3: Net Delta over time
plt.subplot(3, 1, 3)
plt.plot(timestamps[:len(net_deltas)], net_deltas)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.title('Net Delta over Time')
plt.xlabel('Time')
plt.ylabel('Net Delta')
plt.legend()
plt.grid(True)
plt.show()

# Plot 4: Hedge Position vs Theoretical
plt.subplot(2, 1, 1)
plt.plot(timestamps[:len(hedge_positions)], hedge_positions, label='Actual Hedge')
plt.plot(timestamps[:len(theoretical_positions)], theoretical_positions, 'k--', alpha=0.6, 
         label='Theoretical (-net_delta)')
plt.title('Actual Hedge vs Theoretical Position')
plt.xlabel('Time')
plt.ylabel('Position Size')
plt.legend()
plt.grid(True)

# Plot 5: Hedge Trades
plt.subplot(2, 1, 2)
plt.bar(timestamps[:len(hedge_trades)], hedge_trades, width=0.003, alpha=0.7)
plt.title('Hedge Trades at Each Step')
plt.xlabel('Time')
plt.ylabel('Trade Size (+ buy, - sell)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Additional plot: Relationship between price and positions
plt.figure(figsize=(12, 8))
plt.scatter(prices, hedge_positions, label='Actual Hedge', alpha=0.7, c='blue')
plt.scatter(prices, theoretical_positions, label='Theoretical (-net_delta)', alpha=0.3, c='red')
plt.title('Hedge Position vs Price')
plt.xlabel('Price (USD)')
plt.ylabel('Position Size')
plt.legend()
plt.grid(True)
plt.show()

# Create a plot comparing PnL performance
# Calculate what the PnL would be with a standard hedge strategy
standard_pnl = 0
standard_hedge = 0
standard_pnl_history = []

for i in range(1, len(filtered_BTC)):
    current_price = filtered_BTC.loc[i, 'Close']
    prev_price = filtered_BTC.loc[i-1, 'Close']
    
    # Calculate theoretical hedge at previous step
    current_time = pd.to_datetime(filtered_BTC.loc[i, 'Datetime'])
    T_remaining = (expiration - current_time).total_seconds() / (365 * 24 * 3600)
    if T_remaining <= 0:
        break
    
    call_delta = bs_call_delta(prev_price, strike, T_remaining, r, sigma)
    put_delta = bs_put_delta(prev_price, strike, T_remaining, r, sigma)
    net_delta = -(call_delta + put_delta)
    
    # Apply standard delta hedge (always -net_delta if above threshold)
    if abs(net_delta) > threshold:
        standard_hedge = -net_delta
    else:
        standard_hedge = 0
    
    # Calculate PnL
    std_pnl = standard_hedge * (current_price - prev_price)
    standard_pnl += std_pnl
    standard_pnl_history.append(standard_pnl)

# Add comparison plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps[1:len(pnl_history)], pnl_history[1:], label='Incremental Strategy with Multiplier')
plt.plot(timestamps[1:len(standard_pnl_history)+1], standard_pnl_history, label='Standard Delta Hedge')
plt.title('PnL Comparison: Incremental vs Standard Strategy')
plt.xlabel('Time')
plt.ylabel('PnL (USD)')
plt.legend()
plt.grid(True)
plt.show()