import numpy as np
import pandas as pd

def bollinger_bands(data, window=20):
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (std_dev * 2)
    lower_band = middle_band - (std_dev * 2)
    return upper_band, middle_band, lower_band

def exponential_moving_average(data, period=10):
    ema = data['Close'].ewm(span=period, adjust=False).mean()
    return ema

def average_true_range(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr

def relative_strength_index(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.iloc[:period] = rsi.iloc[period]

    return rsi

def on_balance_volume(data):
    obv = np.zeros(len(data))
    obv[0] = data['Volume'][0]

    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i-1]:
            obv[i] = obv[i-1] + data['Volume'][i]
        elif data['Close'][i] < data['Close'][i-1]:
            obv[i] = obv[i-1] - data['Volume'][i]
        else:
            obv[i] = obv[i-1]

    return obv

def stochastic_oscillator(data, period=14):
    high = data['High'].rolling(window=period).max()
    low = data['Low'].rolling(window=period).min()

    k = (data['Close'] - low) / (high - low) * 100
    k.iloc[:period] = k.iloc[period]

    return k

def moving_average_convergence_divergence(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()

    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()

    return macd, signal

def moving_average(data, period=10):
    ma = data['Close'].rolling(window=period).mean()
    ma.iloc[:period] = ma.iloc[period]

    return ma

def signal_variance(data, period=10):
    variance = data['Close'].rolling(window=period).var()
    variance.iloc[:period] = variance.iloc[period]

    return variance


def simple_moving_average(data, period=10):
    sma = data['Close'].rolling(window=period).mean()
    sma.iloc[:period] = sma.iloc[period]

    return sma

def weighted_moving_average(data, period=10):
    weights = np.arange(1, period + 1)
    wma = data['Close'].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    wma.iloc[:period] = wma.iloc[period]

    return wma

def rate_of_change(data, period=10):
    roc = data['Close'].diff(periods=period) / data['Close'].shift(periods=period) * 100
    roc.iloc[:period] = roc.iloc[period]

    return roc

def bullish_engulfing(data):
    condition = (
        (data['Close'].shift(1) < data['Open'].shift(1)) &
        (data['Close'] > data['Open']) &
        (data['Open'] < data['Close'].shift(1)) &
        (data['Close'] > data['Open'].shift(1))
    )

    # Calculate the ratio of the current candle body to the previous candle body
    ratio = ((data['Close'] - data['Open']) / (data['Open'].shift(1) - data['Close'].shift(1))).where(condition, 0)

    return ratio

def bearish_engulfing(data):
    condition = (
        (data['Close'].shift(1) > data['Open'].shift(1)) &
        (data['Close'] < data['Open']) &
        (data['Open'] > data['Close'].shift(1)) &
        (data['Close'] < data['Open'].shift(1))
    )

    # Calculate the ratio of the current candle body to the previous candle body
    ratio = ((data['Open'] - data['Close']) / (data['Close'].shift(1) - data['Open'].shift(1))).where(condition, 0)

    return ratio




def apply_indicators(data):
    data['RSI'] = relative_strength_index(data)
    data['OBV'] = on_balance_volume(data)
    data['MACD'], data['MACD_Signal'] = moving_average_convergence_divergence(data)
    data['%K'] = stochastic_oscillator(data)
    data['MA'] = moving_average(data)
    data['Signal_Variance'] = signal_variance(data)
    data['SMA'] = simple_moving_average(data)
    data['WMA'] = weighted_moving_average(data)
    data['ROC'] = rate_of_change(data)
    data['Bullish_Engulfing'] = bullish_engulfing(data)
    data['Bearish_Engulfing'] = bearish_engulfing(data)

    upper_band, middle_band, lower_band = bollinger_bands(data)
    data['Upper_Band'] = upper_band.fillna(method='bfill')
    data['Middle_Band'] = middle_band.fillna(method='bfill')
    data['Lower_Band'] = lower_band.fillna(method='bfill')

    data['EMA'] = exponential_moving_average(data)
    data['ATR'] = average_true_range(data).fillna(method='bfill')

    return data



# Function to normalize data
def normalize_data(data, epsilon=1e-10):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + epsilon)
    return normalized_data
