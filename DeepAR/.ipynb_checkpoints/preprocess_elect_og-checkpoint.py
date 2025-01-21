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

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prep_data(data, covariates, data_start, train = True):
    # Takes input data and transforms it into sliding windows for time series prediction
    # data: The stock return data
    # covariates: Additional features
    # data_start: When each series begins
    # train: Boolean to indicate if processing training or test data
    
    #print("train: ", train)
    time_len = data.shape[0]
    #print("time_len: ", time_len)
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    #print("windows pre: ", windows_per_series.shape)
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    #print("data_start: ", data_start.shape)
    #print(data_start)
    #print("windows: ", windows_per_series.shape)
    #print(windows_per_series)
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':
    # Update dataset path and name
    save_path = ''
    name = 'amazon_stock.csv'
    save_name = 'amazon_stock_processed'
    window_size = 35
    stride_size = 24
    num_covariates = 3
    train_start = '2022-01-01'
    train_end = '2022-08-31'
    test_start = '2022-08-25'  # Need additional 7 days as given info
    test_end = '2022-12-30'
    pred_days = 7
    given_days = 7

    # Ensure the processed save path exists
    save_path = os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Update the path to your dataset
    csv_path = os.path.join('https://raw.githubusercontent.com/VivianZhao12/CAPSTONE-stockreturn/refs/heads/master/Data/', name)

    data_frame = pd.read_csv(csv_path, parse_dates=True)
    data_frame = data_frame.drop(columns = ['Ticker', 'Industry'])
    data_frame['Date'] = pd.to_datetime(data_frame['Date']).dt.date
    data_frame.set_index("Date", inplace= True, drop=True)
    data_frame.index = pd.to_datetime(data_frame.index)
    data_frame.drop(columns = data_frame.columns[0], inplace=True)


    # Ensure data is in the required date range
    data_frame = data_frame[pd.to_datetime(train_start):pd.to_datetime(test_end)]

    # Fill missing values
    data_frame.fillna(0, inplace=True)

    # Generate covariates (adjust function `gen_covariates` to fit your needs)
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)

    # Split data into training and testing
    train_data = data_frame[train_start:train_end].values
    test_data = data_frame[test_start:test_end].values

    # Find the first nonzero value in each time series
    data_start = (train_data != 0).argmax(axis=0)

    # Get data dimensions
    total_time = data_frame.shape[0]
    num_series = data_frame.shape[1]

    # Prepare data for training and testing (adjust `prep_data` as needed)
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)
