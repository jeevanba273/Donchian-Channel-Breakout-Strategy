import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates  # For improved date formatting
import io
import contextlib

# =============================================================================
# Helper to capture printed output from functions
# =============================================================================
def capture_print_output(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()

# =============================================================================
# Analysis Functions
# =============================================================================
def calculate_donchian_channels(df, window=20):
    """Compute Donchian Channels."""
    df['upper_channel'] = df['HIGH_PRICE'].rolling(window=window, min_periods=1).max()
    df['lower_channel'] = df['LOW_PRICE'].rolling(window=window, min_periods=1).min()
    df['channel_width'] = df['upper_channel'] - df['lower_channel']
    df['channel_width_percentage'] = (df['channel_width'] / df['CLOSE_PRICE']) * 100
    return df

def calculate_additional_indicators(df, 
                                    ma_periods=[50, 200],
                                    macd_fast=12, macd_slow=26, macd_signal_period=9,
                                    stoch_period=14, stoch_smooth=3,
                                    adx_period=14):
    """
    Compute additional indicators:
      - RSI (14-day)
      - Moving Averages (for each period in ma_periods)
      - ATR (14-day)
      - Bollinger Bands (20-day)
      - MACD (using macd_fast, macd_slow, macd_signal_period)
      - Stochastic Oscillator (using stoch_period and stoch_smooth)
      - OBV (if VOLUME exists)
      - ADX (using adx_period)
    """
    # RSI
    delta = df['CLOSE_PRICE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    for period in ma_periods:
        df[f'MA{period}'] = df['CLOSE_PRICE'].rolling(window=period).mean()
    
    # ATR
    high_low = df['HIGH_PRICE'] - df['LOW_PRICE']
    high_close = np.abs(df['HIGH_PRICE'] - df['CLOSE_PRICE'].shift())
    low_close = np.abs(df['LOW_PRICE'] - df['CLOSE_PRICE'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Bollinger Bands (20-day)
    df['BB_middle'] = df['CLOSE_PRICE'].rolling(window=20).mean()
    df['BB_std'] = df['CLOSE_PRICE'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # MACD
    ema_fast = df['CLOSE_PRICE'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['CLOSE_PRICE'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal_period, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Stochastic Oscillator
    low_min = df['LOW_PRICE'].rolling(window=stoch_period).min()
    high_max = df['HIGH_PRICE'].rolling(window=stoch_period).max()
    df['Stoch_K'] = 100 * (df['CLOSE_PRICE'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=stoch_smooth).mean()
    
    # OBV (if VOLUME exists)
    if 'VOLUME' in df.columns:
        df['OBV'] = (np.sign(df['CLOSE_PRICE'].diff()) * df['VOLUME']).fillna(0).cumsum()
    else:
        df['OBV'] = np.nan
    
    # ADX Calculation
    df['H-L'] = df['HIGH_PRICE'] - df['LOW_PRICE']
    df['H-PC'] = abs(df['HIGH_PRICE'] - df['CLOSE_PRICE'].shift())
    df['L-PC'] = abs(df['LOW_PRICE'] - df['CLOSE_PRICE'].shift())
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['+DM'] = np.where((df['HIGH_PRICE'] - df['HIGH_PRICE'].shift(1)) > (df['LOW_PRICE'].shift(1) - df['LOW_PRICE']),
                         np.maximum(df['HIGH_PRICE'] - df['HIGH_PRICE'].shift(1), 0), 0)
    df['-DM'] = np.where((df['LOW_PRICE'].shift(1) - df['LOW_PRICE']) > (df['HIGH_PRICE'] - df['HIGH_PRICE'].shift(1)),
                         np.maximum(df['LOW_PRICE'].shift(1) - df['LOW_PRICE'], 0), 0)
    df['TR_sum'] = df['TR'].rolling(window=adx_period).sum()
    df['+DM_sum'] = df['+DM'].rolling(window=adx_period).sum()
    df['-DM_sum'] = df['-DM'].rolling(window=adx_period).sum()
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=adx_period).mean()
    
    # Drop temporary columns
    df.drop(columns=['BB_std', 'H-L', 'H-PC', 'L-PC', 'TR', '+DM', '-DM', 'TR_sum', '+DM_sum', '-DM_sum', '+DI', '-DI', 'DX'], inplace=True)
    
    return df

def generate_signals(df, 
                     use_rsi=False, rsi_threshold=50,
                     use_ma=False, ma_long=50, ma_short=200,
                     use_macd=False,
                     use_adx=False, adx_threshold=25,
                     use_stoch=False):
    """
    Generate trade signals based on Donchian Channel breakout.
    Extra filters cancel signals if conditions are not met.
    """
    df['signal'] = 0
    # Base signal: Donchian Channel Breakout
    df.loc[(df['CLOSE_PRICE'] > df['upper_channel'].shift(1)) &
           (df['CLOSE_PRICE'].shift(1) <= df['upper_channel'].shift(1)), 'signal'] = 1
    df.loc[(df['CLOSE_PRICE'] < df['lower_channel'].shift(1)) &
           (df['CLOSE_PRICE'].shift(1) >= df['lower_channel'].shift(1)), 'signal'] = -1

    if use_rsi:
        df.loc[(df['signal'] == 1) & (df['RSI'].shift(1) <= rsi_threshold), 'signal'] = 0
        df.loc[(df['signal'] == -1) & (df['RSI'].shift(1) >= rsi_threshold), 'signal'] = 0

    if use_ma:
        df.loc[(df['signal'] == 1) & (df['CLOSE_PRICE'].shift(1) <= df[f'MA{ma_long}'].shift(1)), 'signal'] = 0
        df.loc[(df['signal'] == -1) & (df['CLOSE_PRICE'].shift(1) >= df[f'MA{ma_short}'].shift(1)), 'signal'] = 0

    if use_macd:
        df.loc[(df['signal'] == 1) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 0
        df.loc[(df['signal'] == -1) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = 0

    if use_adx:
        df.loc[(df['signal'] != 0) & (df['ADX'].shift(1) < adx_threshold), 'signal'] = 0

    if use_stoch:
        df.loc[(df['signal'] == 1) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)), 'signal'] = 0
        df.loc[(df['signal'] == -1) & (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)), 'signal'] = 0

    return df

def backtest_strategy(df):
    """
    Calculate strategy returns and track trades.
    """
    df['returns'] = df['CLOSE_PRICE'].pct_change()
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    trades = []
    current_trade = None
    for i in range(len(df)):
        if current_trade is None and df['signal'].iloc[i] != 0:
            current_trade = {
                'entry_date': df.index[i],
                'entry_price': df['CLOSE_PRICE'].iloc[i],
                'trade_type': 'Long' if df['signal'].iloc[i] == 1 else 'Short',
                'entry_signal': df['signal'].iloc[i]
            }
        elif current_trade is not None:
            if (current_trade['entry_signal'] == 1 and df['signal'].iloc[i] == -1) or \
               (current_trade['entry_signal'] == -1 and df['signal'].iloc[i] == 1) or \
               (df['signal'].iloc[i] == 0):
                current_trade['exit_date'] = df.index[i]
                current_trade['exit_price'] = df['CLOSE_PRICE'].iloc[i]
                current_trade['trade_return'] = ((current_trade['exit_price'] - current_trade['entry_price'])
                                                 / current_trade['entry_price'] * current_trade['entry_signal'])
                current_trade['trade_duration'] = (current_trade['exit_date'] - current_trade['entry_date']).days
                trades.append(current_trade)
                current_trade = None
    df['trade_details'] = None
    return df, trades

def calculate_performance_metrics(cumulative_returns, strategy_data, trades):
    """
    Compute key performance metrics.
    """
    start_date = cumulative_returns.index[0].date()
    end_date = cumulative_returns.index[-1].date()
    total_return = cumulative_returns.iloc[-1]
    years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    annualized_return = (total_return ** (1/years) - 1) * 100
    risk_free_rate = 0.03
    daily_returns = cumulative_returns.pct_change()
    excess_returns = daily_returns - (risk_free_rate/252)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['trade_return'] > 0]
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_trade_duration = trades_df['trade_duration'].mean() if len(trades_df) > 0 else 0
    trade_returns = strategy_data['strategy_returns']
    trade_series = trade_returns > 0
    max_consecutive_winners = trade_series.groupby((trade_series != trade_series.shift()).cumsum()).sum().max()
    max_consecutive_losers = (~trade_series).groupby((trade_series != trade_series.shift()).cumsum()).sum().max()
    
    return {
        'Analysis Period': f"From {start_date} to {end_date}",
        'Total Analysis Duration': f"{(end_date - start_date).days} days",
        'Total Return (%)': (total_return - 1) * 100,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate (%)': win_rate,
        'Max Consecutive Winners': max_consecutive_winners,
        'Max Consecutive Losers': max_consecutive_losers,
        'Total Number of Trades': len(trades_df),
        'Average Trade Duration': avg_trade_duration
    }

def analyze_drawdowns(cumulative_returns):
    """
    Compute drawdown metrics.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown_info = []
    current_peak_idx = 0
    current_peak_value = cumulative_returns.iloc[0]
    current_trough_idx = 0
    current_trough_value = cumulative_returns.iloc[0]
    in_drawdown = False
    for i in range(len(drawdown)):
        if cumulative_returns.iloc[i] > current_peak_value and not in_drawdown:
            current_peak_idx = i
            current_peak_value = cumulative_returns.iloc[i]
        if cumulative_returns.iloc[i] < current_peak_value:
            in_drawdown = True
            if cumulative_returns.iloc[i] < current_trough_value or not in_drawdown:
                current_trough_idx = i
                current_trough_value = cumulative_returns.iloc[i]
        if in_drawdown and i > current_trough_idx and cumulative_returns.iloc[i] >= current_peak_value:
            dd_percent = (current_trough_value - current_peak_value) / current_peak_value
            drawdown_info.append({
                'peak_date': cumulative_returns.index[current_peak_idx],
                'peak_value': current_peak_value,
                'trough_date': cumulative_returns.index[current_trough_idx],
                'trough_value': current_trough_value,
                'recovery_date': cumulative_returns.index[i],
                'drawdown_percentage': dd_percent,
                'drawdown_duration': (cumulative_returns.index[i] - cumulative_returns.index[current_peak_idx]).days,
                'recovery_duration': (cumulative_returns.index[i] - cumulative_returns.index[current_trough_idx]).days
            })
            in_drawdown = False
            current_peak_idx = i
            current_peak_value = cumulative_returns.iloc[i]
            current_trough_value = cumulative_returns.iloc[i]
    if in_drawdown:
        dd_percent = (current_trough_value - current_peak_value) / current_peak_value
        drawdown_info.append({
            'peak_date': cumulative_returns.index[current_peak_idx],
            'peak_value': current_peak_value,
            'trough_date': cumulative_returns.index[current_trough_idx],
            'trough_value': current_trough_value,
            'recovery_date': None,
            'drawdown_percentage': dd_percent,
            'drawdown_duration': (cumulative_returns.index[-1] - cumulative_returns.index[current_peak_idx]).days,
            'recovery_duration': None
        })
    return drawdown_info

def analyze_returns(cumulative_returns, strategy_returns):
    """
    Compute returns analysis and return both a figure and a dictionary of key metrics.
    """
    print("\n--- Detailed Returns Analysis ---")
    total_return = cumulative_returns.iloc[-1]
    start_date = cumulative_returns.index[0].date()
    end_date = cumulative_returns.index[-1].date()
    analysis_duration = (end_date - start_date).days
    years = analysis_duration / 365.25
    annualized_return = (total_return ** (1/years) - 1) * 100
    daily_returns = strategy_returns.dropna()
    print(f"Analysis Period: {start_date} to {end_date} ({analysis_duration} days)")
    print(f"Total Cumulative Return: {(total_return - 1) * 100:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print("\nReturn Distribution:")
    print(f"Mean Daily Return: {daily_returns.mean() * 100:.4f}%")
    print(f"Median Daily Return: {daily_returns.median() * 100:.4f}%")
    print(f"Standard Deviation of Daily Returns: {daily_returns.std() * 100:.4f}%")
    print(f"Return Skewness: {daily_returns.skew():.4f}")
    print(f"Return Kurtosis: {daily_returns.kurtosis():.4f}")
    t_statistic, p_value = stats.ttest_1samp(daily_returns, 0)
    print(f"\nt-test for Returns (H0: Mean Return = 0)")
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    monthly_returns = daily_returns.resample('M').sum()
    if isinstance(monthly_returns.index, pd.PeriodIndex):
        monthly_returns.index = monthly_returns.index.to_timestamp()
    yearly_returns = daily_returns.resample('Y').sum()
    
    print("\nMonthly Returns:")
    print(monthly_returns.describe())
    print("\nYearly Returns:")
    print(yearly_returns.describe())
    
    fig = plt.figure(figsize=(15, 10))
    # Daily Returns Distribution
    plt.subplot(2, 2, 1)
    daily_returns.hist(bins=50, edgecolor='black')
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    
    # Monthly Returns (display selected tick labels)
    ax2 = plt.subplot(2, 2, 2)
    monthly_returns.plot(kind='bar', ax=ax2)
    ax2.set_title('Monthly Returns')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Monthly Return')
    visible_ticks = []
    visible_labels = []
    for i, date in enumerate(monthly_returns.index):
        if date.month == 1 and date.year % 2 == 0:
            visible_ticks.append(i)
            visible_labels.append(str(date.year))
    ax2.set_xticks(visible_ticks)
    ax2.set_xticklabels(visible_labels)
    plt.xticks(rotation=45)
    
    # Cumulative Returns
    plt.subplot(2, 2, 3)
    cumulative_returns.plot()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    
    # 30-Day Rolling Returns
    plt.subplot(2, 2, 4)
    rolling_returns = daily_returns.rolling(window=30).mean()
    rolling_returns.plot()
    plt.title('30-Day Rolling Returns')
    plt.xlabel('Date')
    plt.ylabel('30-Day Rolling Return')
    
    plt.tight_layout()
    
    returns_metrics = {
        "Analysis Period": f"{start_date} to {end_date} ({analysis_duration} days)",
        "Total Cumulative Return (%)": f"{(total_return - 1) * 100:.2f}",
        "Annualized Return (%)": f"{annualized_return:.2f}",
        "Mean Daily Return (%)": f"{daily_returns.mean() * 100:.4f}",
        "Median Daily Return (%)": f"{daily_returns.median() * 100:.4f}",
        "Std Dev of Daily Returns (%)": f"{daily_returns.std() * 100:.4f}",
        "Return Skewness": f"{daily_returns.skew():.4f}",
        "Return Kurtosis": f"{daily_returns.kurtosis():.4f}",
        "t-statistic": f"{t_statistic:.4f}",
        "p-value": f"{p_value:.4f}"
    }
    
    return fig, returns_metrics

def print_detailed_trade_analysis(trades):
    """
    Print detailed trade analysis.
    """
    print("\n--- Detailed Trade Analysis ---")
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['trade_return'] > 0]
    losing_trades = [t for t in trades if t['trade_return'] <= 0]
    print(f"Total Number of Trades: {total_trades}")
    print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.2f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.2f}%)")
    trade_returns = [t['trade_return'] for t in trades]
    print(f"\nBest Trade Return: {max(trade_returns)*100:.2f}%")
    print(f"Worst Trade Return: {min(trade_returns)*100:.2f}%")
    print(f"Average Trade Return: {np.mean(trade_returns)*100:.2f}%")
    trade_durations = [t['trade_duration'] for t in trades]
    print(f"\nShortest Trade Duration: {min(trade_durations)} days")
    print(f"Longest Trade Duration: {max(trade_durations)} days")
    print(f"Average Trade Duration: {np.mean(trade_durations):.2f} days")
    print("\nTop 10 Trades:")
    sorted_trades = sorted(trades, key=lambda x: abs(x['trade_return']), reverse=True)[:10]
    for trade in sorted_trades:
        print(f"Type: {trade['trade_type']}, Entry: {trade['entry_date'].date()}, Exit: {trade['exit_date'].date()}, Return: {trade['trade_return']*100:.2f}%, Duration: {trade['trade_duration']} days")

def print_detailed_drawdown_analysis(drawdown_info):
    """
    Print detailed drawdown analysis.
    """
    print("\n--- Detailed Drawdown Analysis ---")
    total_drawdowns = len(drawdown_info)
    print(f"Total Number of Drawdowns: {total_drawdowns}")
    sorted_drawdowns = sorted(drawdown_info, key=lambda x: x['drawdown_percentage'])
    if len(sorted_drawdowns) > 0:
        worst_drawdown = sorted_drawdowns[0]
        print("\nWorst Drawdown:")
        print(f"Peak Date: {worst_drawdown['peak_date'].date()}")
        print(f"Trough Date: {worst_drawdown['trough_date'].date()}")
        if worst_drawdown['recovery_date'] is not None:
            print(f"Recovery Date: {worst_drawdown['recovery_date'].date()}")
        else:
            print("Recovery Date: Not yet recovered")
        print(f"Drawdown Percentage: {worst_drawdown['drawdown_percentage']*100:.2f}%")
        print(f"Drawdown Duration: {worst_drawdown['drawdown_duration']} days")
        drawdown_durations = [dd['drawdown_duration'] for dd in drawdown_info]
        print(f"\nShortest Drawdown Duration: {min(drawdown_durations)} days")
        print(f"Longest Drawdown Duration: {max(drawdown_durations)} days")
        print(f"Average Drawdown Duration: {np.mean(drawdown_durations):.2f} days")
        print("\nTop 5 Worst Drawdowns:")
        for dd in sorted_drawdowns[:5]:
            recovery_date = dd['recovery_date'].date() if dd['recovery_date'] is not None else "Not yet recovered"
            print(f"Peak: {dd['peak_date'].date()}, Trough: {dd['trough_date'].date()}, Recovery: {recovery_date}, Drawdown: {dd['drawdown_percentage']*100:.2f}%, Duration: {dd['drawdown_duration']} days")
    else:
        print("No drawdowns identified")

def visualize_drawdowns(cumulative_returns, drawdown_info):
    """
    Create drawdown visualizations.
    """
    running_max = cumulative_returns.cummax()
    drawdown_series = (cumulative_returns - running_max) / running_max
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(drawdown_series, color='red')
    plt.title('Drawdowns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    if len(drawdown_info) > 0:
        worst_drawdowns = sorted(drawdown_info, key=lambda x: x['drawdown_percentage'])[:3]
        for dd in worst_drawdowns:
            plt.axvspan(dd['peak_date'], dd['trough_date'], 
                        alpha=0.3, color='gray', 
                        label=f"{dd['peak_date'].date()} to {dd['trough_date'].date()}")
        plt.legend(title='Major Drawdowns')
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_returns, color='blue')
    plt.title('Cumulative Returns with Major Drawdown Periods')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    if len(drawdown_info) > 0:
        for dd in worst_drawdowns:
            plt.axvspan(dd['peak_date'], dd['trough_date'], alpha=0.3, color='red')
            mid_point = dd['peak_date'] + (dd['trough_date'] - dd['peak_date'])/2
            before_points = cumulative_returns[cumulative_returns.index <= mid_point]
            after_points = cumulative_returns[cumulative_returns.index >= mid_point]
            if not before_points.empty and not after_points.empty:
                before_val = before_points.iloc[-1]
                after_val = after_points.iloc[0]
                if before_points.index[-1] != after_points.index[0]:
                    total_time_diff = (after_points.index[0] - before_points.index[-1]).total_seconds()
                    mid_time_diff = (mid_point - before_points.index[-1]).total_seconds()
                    ratio = mid_time_diff / total_time_diff if total_time_diff > 0 else 0
                    y_val = before_val + (after_val - before_val) * ratio
                else:
                    y_val = before_val
                plt.annotate(f"{dd['drawdown_percentage']*100:.1f}%", 
                             xy=(mid_point, y_val),
                             xytext=(0, 30), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.tight_layout()
    return fig

def comprehensive_stock_analysis(ticker_symbol, FILE_URL,
                                 use_rsi=False, rsi_threshold=50,
                                 use_ma=False, ma_long=50, ma_short=200,
                                 use_macd=False,
                                 use_adx=False, adx_threshold=25,
                                 use_stoch=False,
                                 macd_fast=12, macd_slow=26, macd_signal_period=9,
                                 stoch_period=14, stoch_smooth=3,
                                 adx_period=14):
    """
    Perform comprehensive analysis for the given ticker.
    Extra filters are applied if enabled.
    """
    data = pd.read_csv(FILE_URL, parse_dates=['DATE'], low_memory=False)
    ticker_data = data[data['SYMBOL'] == ticker_symbol].set_index('DATE')
    ticker_data.sort_index(ascending=True, inplace=True)
    
    # Compute Donchian channels
    ticker_data = calculate_donchian_channels(ticker_data, window=20)
    # Compute additional indicators with user-specified parameters
    ma_periods = [ma_long, ma_short] if use_ma else [50, 200]
    ticker_data = calculate_additional_indicators(ticker_data, 
                                                  ma_periods=ma_periods,
                                                  macd_fast=macd_fast, macd_slow=macd_slow, macd_signal_period=macd_signal_period,
                                                  stoch_period=stoch_period, stoch_smooth=stoch_smooth,
                                                  adx_period=adx_period)
    # Generate signals with extra filters as specified
    ticker_data = generate_signals(ticker_data, 
                                   use_rsi=use_rsi, rsi_threshold=rsi_threshold,
                                   use_ma=use_ma, ma_long=ma_long, ma_short=ma_short,
                                   use_macd=use_macd,
                                   use_adx=use_adx, adx_threshold=adx_threshold,
                                   use_stoch=use_stoch)
    ticker_data, trades = backtest_strategy(ticker_data)
    
    cumulative_returns = (1 + ticker_data['strategy_returns'].dropna()).cumprod()
    
    # Main Visualizations (2x2 subplot)
    fig_main = plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    cumulative_returns.plot()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    
    plt.subplot(2, 2, 2)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    drawdown.plot()
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    
    plt.subplot(2, 2, 3)
    trade_returns = [t['trade_return'] for t in trades]
    plt.hist(trade_returns, bins=20, edgecolor='black')
    plt.title('Trade Returns Distribution')
    plt.xlabel('Trade Return')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    ticker_data['channel_width_percentage'].plot()
    plt.title('Donchian Channel Width')
    plt.xlabel('Date')
    plt.ylabel('Channel Width (%)')
    
    plt.tight_layout()
    fig_main_out = fig_main  # Capture main figure
    
    # Returns Analysis: capture figure and metrics dictionary
    returns_fig, returns_metrics = analyze_returns(cumulative_returns, ticker_data['strategy_returns'])
    returns_analysis_text = capture_print_output(analyze_returns, cumulative_returns, ticker_data['strategy_returns'])
    
    trade_analysis_text = capture_print_output(print_detailed_trade_analysis, trades)
    drawdown_info = analyze_drawdowns(cumulative_returns)
    drawdown_analysis_text = capture_print_output(print_detailed_drawdown_analysis, drawdown_info)
    fig_drawdowns = visualize_drawdowns(cumulative_returns, drawdown_info)
    
    performance_metrics = calculate_performance_metrics(cumulative_returns, ticker_data, trades)
    perf_metrics_text = "\n".join([f"{k}: {v}" for k, v in performance_metrics.items()])
    
    return {
        'ticker_data': ticker_data,
        'trades': trades,
        'cumulative_returns': cumulative_returns,
        'drawdown_info': drawdown_info,
        'performance_metrics': performance_metrics,
        'fig_main': fig_main_out,
        'fig_returns': returns_fig,
        'returns_metrics': returns_metrics,
        'fig_drawdowns': fig_drawdowns,
        'returns_analysis_text': returns_analysis_text,
        'trade_analysis_text': trade_analysis_text,
        'drawdown_analysis_text': drawdown_analysis_text,
        'perf_metrics_text': perf_metrics_text
    }

# =============================================================================
# Main Streamlit App Code
# =============================================================================
st.set_page_config(page_title="Donchian Channel Breakout: Detailed Drawdown & Advanced Indicator Suite", layout="wide")
st.title("Donchian Channel Breakout: Detailed Drawdown & Advanced Indicator Suite")

# Sidebar: Ticker and Indicator Filter Inputs

# Add creator information at the top of the sidebar
st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/jeevanba273/"
st.sidebar.markdown(
    f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: white;">'
    f'<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">'
    f'`JEEVAN B A`</a>',
    unsafe_allow_html=True
)

# Instead of using a local file path, use your Google Drive link.
file_id = "1FHFthKW-L1hIY0AnrvZnWro-yuf0tUnW"  # Replace with your actual file ID
FILE_URL = f"https://drive.google.com/uc?export=download&id={file_id}"

ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", value="NIFTY 50")

st.sidebar.markdown("### RSI Filter")
use_rsi_filter = st.sidebar.checkbox("Apply RSI Filter", value=False)
rsi_threshold = st.sidebar.number_input("RSI Threshold", value=50, step=1) if use_rsi_filter else 50

st.sidebar.markdown("### Moving Average Filter")
use_ma_filter = st.sidebar.checkbox("Apply MA Filter", value=False)
if use_ma_filter:
    ma_long = st.sidebar.number_input("Long MA Period", value=50, step=1)
    ma_short = st.sidebar.number_input("Short MA Period", value=200, step=1)
else:
    ma_long, ma_short = 50, 200

st.sidebar.markdown("### MACD Filter")
use_macd_filter = st.sidebar.checkbox("Apply MACD Filter", value=False)
if use_macd_filter:
    macd_fast = st.sidebar.number_input("MACD Fast Period", value=12, step=1)
    macd_slow = st.sidebar.number_input("MACD Slow Period", value=26, step=1)
    macd_signal_period = st.sidebar.number_input("MACD Signal Period", value=9, step=1)
else:
    macd_fast, macd_slow, macd_signal_period = 12, 26, 9

st.sidebar.markdown("### ADX Filter")
use_adx_filter = st.sidebar.checkbox("Apply ADX Filter", value=False)
adx_threshold = st.sidebar.number_input("ADX Threshold", value=25, step=1) if use_adx_filter else 25
adx_period = st.sidebar.number_input("ADX Period", value=14, step=1) if use_adx_filter else 14

st.sidebar.markdown("### Stochastic Filter")
use_stoch_filter = st.sidebar.checkbox("Apply Stochastic Filter", value=False)
if use_stoch_filter:
    stoch_period = st.sidebar.number_input("Stochastic %K Period", value=14, step=1)
    stoch_smooth = st.sidebar.number_input("Stochastic %D Period", value=3, step=1)
else:
    stoch_period, stoch_smooth = 14, 3

if st.sidebar.button("Run Analysis"):
    with st.spinner("Performing analysis..."):
        results = comprehensive_stock_analysis(
            ticker_symbol, 
            FILE_URL, 
            use_rsi=use_rsi_filter, 
            rsi_threshold=rsi_threshold, 
            use_ma=use_ma_filter, 
            ma_long=ma_long, 
            ma_short=ma_short,
            use_macd=use_macd_filter,
            use_adx=use_adx_filter, 
            adx_threshold=adx_threshold,
            use_stoch=use_stoch_filter,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal_period=macd_signal_period,
            stoch_period=stoch_period, stoch_smooth=stoch_smooth,
            adx_period=adx_period
        )
    
    # Create tabs for output (Overview, Ticker Data, Trades Data, Returns Analysis, Drawdown Analysis, Performance Metrics)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Ticker Data", 
        "Trades Data", 
        "Returns Analysis", 
        "Drawdown Analysis", 
        "Performance Metrics"
    ])
    
    with tab1:
        st.header("Overview")
        st.subheader("Main Visualizations")
        st.pyplot(results['fig_main'])
    
    with tab2:
        st.header("Ticker Data")
        st.subheader("Data Preview")
        st.dataframe(results['ticker_data'].head())
    
    with tab3:
        st.header("Trades Data")
        trades_df = pd.DataFrame(results['trades'])
        if not trades_df.empty:
            # Multiply trade_return column by 100 for percentage display
            if 'trade_return' in trades_df.columns:
                trades_df['trade_return'] = trades_df['trade_return'] * 100
            st.dataframe(trades_df)
        else:
            st.info("No trades recorded.")
    
    with tab4:
        st.header("Returns Analysis")
        st.subheader("Returns Analysis Figure")
        st.pyplot(results['fig_returns'])
        st.subheader("Returns Metrics Table")
        returns_df = pd.DataFrame.from_dict(results['returns_metrics'], orient="index", columns=["Value"])
        # Convert values to string to avoid Arrow serialization errors
        returns_df["Value"] = returns_df["Value"].astype(str)
        st.table(returns_df)
    
    with tab5:
        st.header("Drawdown Analysis")
        st.subheader("Drawdown Analysis Figure")
        st.pyplot(results['fig_drawdowns'])
        st.subheader("Drawdown Analysis Text")
        st.text(results['drawdown_analysis_text'])
        dd_df = pd.DataFrame(results['drawdown_info'])
        if not dd_df.empty:
            # Multiply drawdown_percentage by 100 for display and convert to string
            dd_df['drawdown_percentage'] = (dd_df['drawdown_percentage'] * 100).astype(str)
            st.subheader("Drawdown Info Table")
            st.dataframe(dd_df)
    
    with tab6:
        st.header("Performance Metrics")
        perf_df = pd.DataFrame.from_dict(results['performance_metrics'], orient="index", columns=["Value"])
        perf_df["Value"] = perf_df["Value"].astype(str)
        st.table(perf_df)
        st.subheader("Raw Performance Metrics Text")
        st.text(results['perf_metrics_text'])
else:
    st.info("Enter a ticker symbol and adjust the indicator conditions, then click 'Run Analysis' in the sidebar.")
