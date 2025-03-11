import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import datetime
import math

from datetime import timedelta, datetime
from scipy.optimize import brentq
from scipy.stats import norm

start = "2025-03-07"
start_time = "2025-03-07 23:00:00"
expiration_time = "2025-03-08 08:00:00"
start_ms = int(pd.Timestamp(start_time).timestamp() * 1000)
end_ms = int(pd.Timestamp(expiration_time).timestamp() * 1000)

strike = 86500          # Assume a fixed strike (e.g. ATM based on initial price)
r = 0.375               # Risk-free rate (for simplicity)
sigma = 0.5072          # Fixed implied volatility (static IV)
expiration = pd.Timestamp(expiration_time, tz="UTC")  # Option expiration time

# Define absolute delta threshold
delta_threshold = 0.10  # Only hedge when absolute net delta is above this value

BTC = yf.Ticker("BTC-USD").history(start=start, interval="1m")
BTC.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
BTC.reset_index(inplace=True)

BTC["Timestamp"] = [date.timestamp() * 1000 for date in BTC["Datetime"]]
filtered_BTC = BTC[(BTC['Timestamp'] >= start_ms) & (BTC['Timestamp'] <= end_ms)]
filtered_BTC = filtered_BTC.reset_index(drop=True).copy()

# Define Black–Scholes delta functions (for call and put)
def bs_call_delta(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def bs_put_delta(S, K, T, r, sigma):
    return bs_call_delta(S, K, T, r, sigma) - 1

# Initialize variables for the simulation:
cumulative_hedge_pnl = 0.0   # Will accumulate the PNL from hedge trades
prev_price = None            # Underlying price at the start of the interval
actual_hedge_shares = 0      # The actual hedge position after applying threshold
trade_count = 0              # Count of trades executed
last_hedge_price = None      # Price at which last hedge was executed

# Tracking variables
timestamps = []
prices = []
net_deltas = []
hedge_positions = []
actual_hedge_positions = []
trade_executed = []
pnl_values = []
delta_above_threshold = []   # Track if delta is above threshold
hedging_status = []          # Track if we're in hedging mode

# Loop over each time step in the filtered BTC data (each minute):
for i, row in filtered_BTC.iterrows():
    # Get the current time and underlying price:
    current_time = pd.to_datetime(row["Datetime"])
    S = row["Close"]
    
    # Compute time to expiration in years:
    T_remaining = (expiration - current_time).total_seconds() / (365 * 24 * 3600)
    if T_remaining <= 0:
        break  # Stop the simulation if expiration has passed
    
    # Calculate option deltas (using static sigma) with updated T and S:
    call_delta = bs_call_delta(S, strike, T_remaining, r, sigma)
    put_delta = bs_put_delta(S, strike, T_remaining, r, sigma)
    
    # For a short straddle (short one call and one put), net delta is:
    net_delta = - (call_delta + put_delta)
    # The theoretical hedge position is the negative of the net delta:
    hedge_shares = -net_delta
    
    # Check if absolute net delta is above threshold
    is_above_threshold = abs(net_delta) > delta_threshold
    
    # For the very first time step, initialize values:
    if prev_price is None:
        prev_price = S
        actual_hedge_shares = 0  # Start with no hedge
        trade_this_step = False
        in_hedging_mode = False
        
        # Save tracking data
        timestamps.append(current_time)
        prices.append(S)
        net_deltas.append(net_delta)
        hedge_positions.append(hedge_shares)
        actual_hedge_positions.append(actual_hedge_shares)
        trade_executed.append(trade_this_step)
        pnl_values.append(0)
        delta_above_threshold.append(is_above_threshold)
        hedging_status.append(in_hedging_mode)
        continue
    
    # Determine hedging mode and actions
    if not in_hedging_mode and is_above_threshold:
        # Start hedging mode
        in_hedging_mode = True
        actual_hedge_shares = hedge_shares
        trade_this_step = True
        trade_count += 1
        last_hedge_price = S
    elif in_hedging_mode and not is_above_threshold:
        # Exit hedging mode
        in_hedging_mode = False
        actual_hedge_shares = 0  # Unwind hedge
        trade_this_step = True
        trade_count += 1
        last_hedge_price = S
    elif in_hedging_mode:
        # Continue hedging mode with minute-by-minute rebalancing
        actual_hedge_shares = hedge_shares
        trade_this_step = True
        trade_count += 1
        last_hedge_price = S
    else:
        # Not in hedging mode and delta still below threshold
        actual_hedge_shares = 0
        trade_this_step = False
    
    # Calculate PNL for the interval (only if we had a position)
    if actual_hedge_shares != 0 or prev_price == last_hedge_price:
        pnl_trade = actual_hedge_shares * (S - prev_price)
        cumulative_hedge_pnl += pnl_trade
    else:
        pnl_trade = 0
    
    # Save tracking data
    timestamps.append(current_time)
    prices.append(S)
    net_deltas.append(net_delta)
    hedge_positions.append(hedge_shares)
    actual_hedge_positions.append(actual_hedge_shares)
    trade_executed.append(trade_this_step)
    pnl_values.append(pnl_trade)
    delta_above_threshold.append(is_above_threshold)
    hedging_status.append(in_hedging_mode)
    
    # Update for next interval
    prev_price = S

# Create tracking dataframe
tracking_df = pd.DataFrame({
    'Timestamp': timestamps,
    'Price': prices,
    'Net_Delta': net_deltas,
    'Theoretical_Hedge': hedge_positions,
    'Actual_Hedge': actual_hedge_positions,
    'Trade_Executed': trade_executed,
    'Delta_Above_Threshold': delta_above_threshold,
    'In_Hedging_Mode': hedging_status,
    'Interval_PnL': pnl_values
})

# Calculate cumulative PnL
tracking_df['Cumulative_PnL'] = tracking_df['Interval_PnL'].cumsum()

# Output the results:
print(f"Delta threshold: {delta_threshold}")
print(f"Cumulative Hedging PnL: {cumulative_hedge_pnl:.2f}")
print(f"Total number of trades: {trade_count}")
print(f"Original trades (without threshold): {len(tracking_df)}")
print(f"Trade reduction: {((len(tracking_df) - trade_count) / len(tracking_df) * 100):.2f}%")
print(f"Time spent in hedging mode: {tracking_df['In_Hedging_Mode'].sum()} / {len(tracking_df)} minutes")
print(f"Percentage of time hedging: {(tracking_df['In_Hedging_Mode'].sum() / len(tracking_df) * 100):.2f}%")

# Plot visualization
plt.figure(figsize=(12, 8))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Plot 1: Price over time
ax1.plot(tracking_df['Timestamp'], tracking_df['Price'], 'b-')
ax1.set_title('BTC Price')
ax1.set_ylabel('Price (USD)')
ax1.grid(True)

# Plot 2: Net Delta and Hedge Positions
ax2.plot(tracking_df['Timestamp'], tracking_df['Net_Delta'], 'g-', label='Net Delta')
ax2.plot(tracking_df['Timestamp'], tracking_df['Theoretical_Hedge'], 'r--', label='Theoretical Hedge')
ax2.plot(tracking_df['Timestamp'], tracking_df['Actual_Hedge'], 'b-', label='Actual Hedge')

# Add threshold lines
ax2.axhline(y=delta_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Upper Threshold (+{delta_threshold})')
ax2.axhline(y=-delta_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Lower Threshold (-{delta_threshold})')

ax2.set_title('Delta and Hedge Positions')
ax2.set_ylabel('Delta/Position')
ax2.grid(True)
ax2.legend()

# Plot 3: PnL
ax3.plot(tracking_df['Timestamp'], tracking_df['Cumulative_PnL'], 'purple')
ax3.set_title('Cumulative PnL')
ax3.set_xlabel('Time')
ax3.set_ylabel('PnL (USD)')
ax3.grid(True)

# Add shading for hedging periods
for i in range(len(tracking_df)-1):
    if tracking_df['In_Hedging_Mode'].iloc[i]:
        ax1.axvspan(tracking_df['Timestamp'].iloc[i], tracking_df['Timestamp'].iloc[i+1], 
                   alpha=0.2, color='green')
        ax2.axvspan(tracking_df['Timestamp'].iloc[i], tracking_df['Timestamp'].iloc[i+1], 
                   alpha=0.2, color='green')
        ax3.axvspan(tracking_df['Timestamp'].iloc[i], tracking_df['Timestamp'].iloc[i+1], 
                   alpha=0.2, color='green')

# Add markers for executed trades
for i, executed in enumerate(tracking_df['Trade_Executed']):
    if executed:
        ax2.plot(tracking_df['Timestamp'].iloc[i], tracking_df['Actual_Hedge'].iloc[i], 'ro', markersize=4)

plt.tight_layout()
plt.show()

# Create a scatter plot to visualize the relationship between price and net delta
plt.figure(figsize=(10, 6))
scatter = plt.scatter(tracking_df['Price'], tracking_df['Net_Delta'], 
                     c=tracking_df['In_Hedging_Mode'], cmap='coolwarm', 
                     alpha=0.7, edgecolors='k', linewidth=0.5)
plt.axhline(y=delta_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Upper Threshold (+{delta_threshold})')
plt.axhline(y=-delta_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Lower Threshold (-{delta_threshold})')
plt.legend(handles=[*scatter.legend_elements()[0]], labels=['No Hedging', 'Hedging'])
plt.title('Relationship between Price and Net Delta')
plt.xlabel('BTC Price (USD)')
plt.ylabel('Net Delta')
plt.grid(True)
plt.show()