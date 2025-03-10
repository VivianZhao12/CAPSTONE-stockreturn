from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange
from scipy import stats
import argparse


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prep_data(data, covariates, data_start, train=True):
    """
    Takes input data and transforms it into sliding windows for time series prediction
    Args:
        data: The stock return data
        covariates: Additional features including technical indicators
        data_start: When each series begins
        train: Boolean to indicate if processing training or test data
    """
    time_len = data.shape[0]  # Get the time length of the data
    input_size = window_size - stride_size  # Size of input portion (30-5=25)
    # Calculate how many windows can be generated
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    if train:
        windows_per_series -= (data_start+stride_size-1) // stride_size
    
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')  # Input features
    label = np.zeros((total_windows, window_size), dtype='float32')  # Prediction targets
    v_input = np.zeros((total_windows, 2), dtype='float32')  # Scaling factors
    
    count = 0
    if not train:
        covariates = covariates[-time_len:]  # testdata: only use covariate portion for test data
        
    for series in trange(num_series):
        # Calculate z-score of time features
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        
        # Calculate start and end position for each window - for data with all trading days, data_start has no effect
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            
            # Set target variable (daily return)
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            # Set all covariates
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            # Set series identifier
            x_input[count, :, -1] = series
            # Set labels
            label[count, :] = data[window_start:window_end, series]
            # Calculate scaling factors
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()  # Calculate number of non-zero values
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                # Calculate average of non-zero values and add 1
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum)+1
                # Normalize input data using scaling factor - converts data of different ranges to similar scales
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    # If training data, apply the same normalization to labels
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
            
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)
    
def gen_covariates(times, price_data, num_covariates, with_sentiment=False):
    """
    Creates additional features including technical indicators
    Args:
        times: DatetimeIndex for the time series
        price_data: DataFrame containing OHLCV data
        num_covariates: Number of features to generate
    Returns:
        ndarray containing all computed features
    """
    covariates = np.zeros((len(times), num_covariates))
    
    # 1. Time-based features
    covariates[:, 0] = stats.zscore([t.weekday() for t in times])  # weekday
    covariates[:, 1] = stats.zscore([t.month for t in times])  # month
    
    # 2. Price-based features
    # Close price has significant relationships with returns and other variables
    # Volume shows relationships with volatility and trading value
    covariates[:, 2] = stats.zscore(price_data['Close'].shift(5).values)
    covariates[:, 3] = stats.zscore(price_data['Volume'].shift(5).values)

    # 3. Technical indicators
    # Intraday return (significant relationship with returns)
    intraday_return = (price_data['Close'] - price_data['Open']) / price_data['Open']
    covariates[:, 4] = stats.zscore(intraday_return.shift(5).values)

    # MA5 (significant relationship with price dynamics)
    ma5 = price_data['Close'].rolling(window=5).mean()
    covariates[:, 5] = stats.zscore((price_data['Close'] - ma5).values)

    # MACD (strong contemporaneous and lagged relationships with returns)
    exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    covariates[:, 6] = stats.zscore(macd.shift(2).values)  # lag-2 MACD (significant)

    # Volatility (significant relationships in the system)
    volatility = (price_data['High'] - price_data['Low']) / price_data['Close']
    covariates[:, 7] = stats.zscore(volatility.values)

    if with_sentiment:
        covariates[:, 8] = stats.zscore(price_data['Sentiment_Score'].shift(5).values)
    
    # Fill NaN values with 0
    covariates = np.nan_to_num(covariates)
    return covariates

def visualize(data, week_start):
    """Helper function to visualize the time series data"""
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Data Preprocessing Script")
    parser.add_argument('--with_sentiment', action='store_true', help="Include sentiment analysis in feature set")
    parser.add_argument('ticker', type=str, help="Stock ticker symbol (e.g., AAPL, MSFT, ABT)")

    args = parser.parse_args()
    ticker = args.ticker 
    
    # Configuration
    save_path = ''
    name = f'{ticker}_stock_data.csv'
    save_name = f'{ticker}_stock_processed'
    window_size = 30    # Size of each data window
    stride_size = 5    # How far to move the window each time
    
    num_covariates = 8 if not args.with_sentiment else 9  # Number of features (2 time + 2 OHLCV + 4 technical)
    train_start = '2020-06-02'
    train_end = '2023-09-25'
    test_start = '2023-09-18'
    test_end = '2025-02-24'
    pred_days = 5       # Prediction horizon
    given_days = 25     # Historical data window

    # Create save directory
    save_path = os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load and prepare data
    if args.with_sentiment:
        csv_path = f'../data/stock/{ticker}_stock_wsenti.csv'
    else:
        csv_path = f'../data/stock/{ticker}_stock_data.csv'
    data_frame = pd.read_csv(csv_path, parse_dates=True)
    
    # Process date column
    data_frame['Date'] = pd.to_datetime(data_frame['Date']).dt.date
    data_frame.set_index("Date", inplace=True, drop=True)
    data_frame.index = pd.to_datetime(data_frame.index)
    
    # Filter date range
    data_frame = data_frame[pd.to_datetime(train_start):pd.to_datetime(test_end)]
    
    # Fill missing values
    data_frame.fillna(method='ffill', inplace=True)  # Forward fill
    data_frame.fillna(0, inplace=True)  # Fill remaining with 0
    
    # Generate features
    # pre-select features using PCMCI
    if args.with_sentiment:
        price_data = data_frame[['High', 'Low', 'Open', 'Close', 'Volume', 'Sentiment_Score']]
    else:
        price_data = data_frame[['High', 'Low', 'Open', 'Close', 'Volume']]
    covariates = gen_covariates(data_frame[train_start:test_end].index, price_data, num_covariates, with_sentiment=args.with_sentiment)

    # Split data
    train_data = data_frame[train_start:train_end]['Daily Return'].values.reshape(-1, 1)
    test_data = data_frame[test_start:test_end]['Daily Return'].values.reshape(-1, 1)

    # Find first non-zero value for each series
    data_start = (train_data != 0).argmax(axis=0)

    # Get dimensions
    total_time = data_frame.shape[0]
    num_series = 1  # Since we're only looking at daily returns

    # Process data
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)