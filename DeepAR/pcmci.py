from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import sys
from tigramite import data_processing as pp
from tigramite import pcmci
from tigramite.independence_tests import parcorr

def generate_all_features(price_data):
    """Generate all potential features for causality testing"""
    features = pd.DataFrame(index=price_data.index)
    
    # Time-based features
    features['age'] = np.arange(len(price_data))
    features['weekday'] = [t.weekday() for t in price_data.index]
    features['month'] = [t.month for t in price_data.index]
    
    # Price-based features
    features['Open'] = price_data['Open'].shift(5)
    features['Close'] = price_data['Close'].shift(5)
    features['High'] = price_data['High'].shift(5)
    features['Low'] = price_data['Low'].shift(5)
    features['Volume'] = price_data['Volume'].shift(5)
    
    # Technical indicators
    features['Volatility'] = (price_data['High'].shift(5) - price_data['Low'].shift(5)) / price_data['Close'].shift(5)
    features['Trading_Value'] = price_data['Volume'].shift(5) * price_data['Close'].shift(5)
    features['MA5'] = price_data['Close'].shift(5).rolling(window=5).mean()
    features['MA10'] = price_data['Close'].shift(5).rolling(window=10).mean()
    features['MA20'] = price_data['Close'].shift(5).rolling(window=20).mean()
    features['Intraday_Return'] = (price_data['Close'].shift(5) - price_data['Open'].shift(5)) / price_data['Open'].shift(5)
    
    # RSI
    prices = price_data['Close'].values
    returns = np.diff(prices, prepend=prices[0])
    gains = np.maximum(returns, 0)
    losses = -np.minimum(returns, 0)
    avg_gains = pd.Series(gains).rolling(window=14).mean()
    avg_losses = pd.Series(losses).rolling(window=14).mean()
    features['RSI'] = 100 - (100 / (1 + avg_gains / avg_losses))
    
    # MACD
    exp1 = price_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_data['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = exp1 - exp2
    
    return features

def run_pcmci_test(data_frame, features_df, log_file):
    data = pd.concat([features_df, data_frame['Daily Return']], axis=1)
    data = data.ffill().fillna(0)
    data_array = data.values
    dataframe = pp.DataFrame(data_array, var_names=data.columns, datatime=data.index)
    parcorr_test = parcorr.ParCorr()
    
    pcmci_obj = pcmci.PCMCI(dataframe=dataframe, cond_ind_test=parcorr_test, verbosity=1)
    
    original_stdout = sys.stdout
    with open(log_file, 'w') as f:
        sys.stdout = f
        results = pcmci_obj.run_pcmciplus(tau_min=0, tau_max=5, pc_alpha=0.05)
        print("\nSignificant causal relationships (p < 0.05):")
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                if i != j:
                    for tau in range(5):
                        if results['p_matrix'][i, j, tau] < 0.05:
                            print(f"{data.columns[i]} ->({tau}) {data.columns[j]}: p={results['p_matrix'][i, j, tau]:.4f}")
    
    sys.stdout = original_stdout
    print(f"PCMCI results have been saved to {log_file}")
    
    return results

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(current_dir, 'pcmci_result.log')
    
    # Configuration
    train_start = '2020-06-02'
    train_end = '2023-09-25'
    
    # Load data
    csv_path = '../data/stock/cvs_stock_wsenti.csv'
    data_frame = pd.read_csv(csv_path, parse_dates=['Date'])
    data_frame.set_index('Date', inplace=True)
    
    # Filter date range
    data_frame = data_frame[train_start:train_end]
    
    # Fill missing values
    data_frame.fillna(method='ffill', inplace=True)  # Forward fill
    data_frame.fillna(0, inplace=True)  # Fill remaining with 0
    
    # Generate features
    price_data = data_frame[['Open', 'Close', 'High', 'Low', 'Volume']]
    features_df = generate_all_features(price_data)
    
    significant_relationships = run_pcmci_test(data_frame, features_df, log_file)
    
    