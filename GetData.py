import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt
import requests
import os
import numpy as np
import random
import math
import sys
import time
import csv
from bs4 import BeautifulSoup as bs
from pandas_datareader import data
from Node import Node
from FeedForwardNet import FeedForwardNet

data_source = 'alphavantage'
api_key = 'S13H3V357VQ534EE'
ticker = "SPY"
random_seed = int(time.time())

def get_historical_data(name, number_of_days):
    data = []
    url = "https://finance.yahoo.com/quote/" + name + "/history/"
    rows = bs(requests.get(url, verify = True).text, features = "lxml").findAll('table')[0].tbody.findAll('tr')

    for each_row in rows:
        divs = each_row.findAll('td')

        if divs[1].span.text != 'Divident':
            data.append({
                'Date': divs[0].span.text, 
                'Open': float(divs[1].span.text.replace(',','')), 
                'Close': float(divs[4].span.text.replace(',','')),
                'High': float(divs[2].span.text.replace(',','')),
                'Low': float(divs[3].span.text.replace(',','')),
                'Volume': float(divs[6].span.text.replace(',','')),
                'Adj_Closing': float(divs[5].span.text.replace(',',''))})

    return data[:number_of_days]

def get_alphavantage_data(ticker):

    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    file_to_save = 'stock_market_data-%s.csv'%ticker

    if not os.path.exists(file_to_save):
        with requests.get(url_string, verify = False) as r:
            data = r.json()
            #print(data)

            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns = ['Date','Low','High','Close','Open','Volume'])

            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),
                            float(v['3. low']),
                            float(v['2. high']),
                            float(v['4. close']),
                            float(v['1. open']),
                            float(v['5. volume'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1

        df = df.sort_values('Date')
        print('Data saved to : %s' %file_to_save)
        df.to_csv(file_to_save)

    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

    return df

def Seed_RNG(seed_val):
    print("RANDOM SEED: ", seed_val)
    random.seed(random_seed)
    np.random.seed(random_seed)

def examine_data_fram(df):
    for name in df.columns:
        print("----------")
        print(df[name].dtype)

        if df[name].dtype is np.dtype('O'):
            print(df[name].value_counts())
            print("Name: ", name)
        else:
            print(df[name].describe())

def get_date_range(start, end, csvFile):
    df = pd.DataFrame(columns = ['Date','Low','High','Close','Open','Volume'])

    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        start = dt.datetime.strptime(start, "%Y-%m-%d")
        end = dt.datetime.strptime(end, "%Y-%m-%d")

        for row in csv_reader:
            if row[1] != "Date":
                tempDate = dt.datetime.strptime(row[1], "%Y-%m-%d")
                if tempDate >= start and tempDate <= end:
                    df.loc[-1,:] = row[1:]
                    df.index = df.index + 1 
    
    return df

def scale_on_lookback_window(num_of_days, variable, dataframe):
    scaled_var = variable + "_scaled"
    dataframe[scaled_var] = np.nan
    var_array = dataframe.as_matrix(columns = [variable])

    for i in range(num_of_days, len(dataframe[scaled_var])):
        data_slice = var_array[(i-num_of_days):i]
        data_avg = np.mean(data_slice)
        data_std = np.std(data_slice)
        dataframe[scaled_var][i] = (dataframe[variable][i] - data_avg) / (2.0*data_std)


begin = "2013-01-01"
stop = "2017-01-01"

spy_df = get_date_range(begin, stop, "stock_market_data-SPY.csv")
spy_df[['Open', 'Close', 'Volume']] = spy_df[['Open', 'Close', 'Volume']].apply(pd.to_numeric)

scale_window = 30
scale_on_lookback_window(scale_window, "Open", spy_df)
scale_on_lookback_window(scale_window, "Close", spy_df)
scale_on_lookback_window(scale_window, "Volume", spy_df)

spy_df.drop(spy_df.index[0:scale_window], inplace = True)
spy_df.reset_index(drop=True, inplace=True)

spy_df["tomorrow_close"] = np.nan
spy_df["week_close"] = np.nan
spy_df["tomorrow_up"] = np.nan
spy_df["week_up"] = np.nan

for i in range(1, len(spy_df["tomorrow_close"])):
    spy_df["tomorrow_close"][i-1] = spy_df["Close_scaled"][i]
    if spy_df["Close_scaled"][i] > spy_df["Close_scaled"][i-1]:
        spy_df["tomorrow_up"][i-1] = 1
    else:
        spy_df["tomorrow_up"][i-1] = 0

for i in range(5, len(spy_df["tomorrow_close"])):
    spy_df["week_close"][i-5] = spy_df["Close_scaled"][i]
    if spy_df["Close_scaled"][i] > spy_df["Close_scaled"][i-5]:
        spy_df["week_up"][i-5] = 1
    else:
        spy_df["week_up"][i-5] = 0

spy_df.drop(spy_df.index[-5:], inplace = True)

Seed_RNG(random_seed)

num_of_iterations = 250
hidden_layers = 1
lookback_window = 3

daily_residuals = [-1.0 for i in range(lookback_window)]
weekly_residuals = [-1.0 for i in range(lookback_window)]

for i in range(lookback_window, len(spy_df["tomorrow_close"])):
    stock_net = FeedForwardNet(3, 1, [7], 2, 0.35)
    for iterations in range(num_of_iterations):
        for x in range(lookback_window - 1):
            idx = i - lookback_window + x
            training_vector = np.array([float(spy_df["Open_scaled"][idx]), float(spy_df["Close_scaled"][idx]),
                                        float(spy_df["Volume_scaled"][idx])])
            training_output = [float(spy_df["tomorrow_close"][idx]), float(spy_df["week_close"][idx])]
            stock_net.FeedForward(training_vector, training_output, Training = True)

    pred_vector = [float(spy_df["Open_scaled"][idx]), float(spy_df["Close_scaled"][i]), float(spy_df["Volume_scaled"][i])]
    pred_closes = stock_net.FeedForward(pred_vector)

    if pred_closes[0] > spy_df["Close_scaled"][i]:
        daily_residuals.append(1)
    else:
        daily_residuals.append(0)

    if pred_closes[1] > spy_df["Close_scaled"][i]:
        weekly_residuals.append(1)
    else:
        weekly_residuals.append(0)

    if i%100 == 0:
        print(i, "iterations")

print(len(weekly_residuals))

spy_df["daily_residuals"] = daily_residuals
spy_df["weekly_residuals"] = weekly_residuals

print("Stats for daily predictions")
error_count = 0
false_pos_count = 0
false_neg_count = 0
up_preds = 0
down_preds = 0

for i in range(0, len(spy_df["tomorrow_up"])):
    if spy_df["daily_residuals"][i] != spy_df["tomorrow_up"][i]:
        error_count+=1
    if spy_df["daily_residuals"][i] > spy_df["tomorrow_up"][i]:
        error_count+=1
