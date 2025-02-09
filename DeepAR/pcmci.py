from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from tigramite import data_processing as pp
from tigramite import pcmci
from tigramite.independence_tests import parcorr


def generate_all_features(price_data):
    """Generate all potential features for Granger causality testing"""
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

# def run_granger_tests(data_frame, features_df):
#     """Run Granger causality tests between features and Daily Return"""
#     significant_features = []
#     maxlag = 5
   
#     for col in features_df.columns:
#         data = pd.DataFrame({
#             'y': data_frame['Daily_Return'],
#             'x': features_df[col]
#         }).dropna()
       
#         result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
#         p_value = result[maxlag][0]['ssr_ftest'][1]
        
#         if p_value < 0.05:
#             significant_features.append(col)
#             print(f"{col}: p-value = {p_value:.4f} - Significant")
#         else:
#             print(f"{col}: p-value = {p_value:.4f}")
           
#     return significant_features

def run_pcmci_test(data_frame, features_df):
    data = pd.concat([features_df, data_frame['Daily_Return']], axis=1)
    data = data.ffill().fillna(0)
    data_array = data.values
    dataframe = pp.DataFrame(data_array, var_names=data.columns, datatime=data.index)

    parcorr_test = parcorr.ParCorr()
    pcmci_obj = pcmci.PCMCI(dataframe=dataframe, cond_ind_test=parcorr_test, verbosity=1)
    
    # PCMCI+
    results = pcmci_obj.run_pcmciplus(tau_min=0, tau_max=5, pc_alpha=0.05)

    # print result
    print("\nSignificant causal relationships (p < 0.05):")
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i != j:
                for tau in range(5):
                    if results['p_matrix'][i, j, tau] < 0.05:
                        print(f"{data.columns[i]} ->({tau}) {data.columns[j]}: p={results['p_matrix'][i, j, tau]:.4f}")
    return results

if __name__ == '__main__':
    # Configuration
    train_start = '2019-12-03'
    train_end = '2023-09-30'
    
    # Load data
    csv_path = '../Data/cvs_stock.csv'
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
    
    # Run Granger tests
    # print("\nRunning Granger causality tests...")
    # significant_features = run_granger_tests(data_frame, features_df)
    # print(f"\nNumber of significant features: {len(significant_features)}")
    # print("Significant features:", significant_features)
    
    # Run PCMCI tests
    significant_relationships = run_pcmci_test(data_frame, features_df)
    
    