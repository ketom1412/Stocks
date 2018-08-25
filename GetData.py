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
from Performance import Stats
from RunData import runData

data_source = 'alphavantage'
api_key = '' 
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

def StandardData(scale_window, start, end, csvFile):
    spy_df = get_date_range(start, end, csvFile)
    spy_df[['Open', 'Close', 'Volume']] = spy_df[['Open', 'Close', 'Volume']].apply(pd.to_numeric)

    scale_on_lookback_window(scale_window, "Open", spy_df)
    scale_on_lookback_window(scale_window, "Close", spy_df)
    scale_on_lookback_window(scale_window, "Volume", spy_df)

    spy_df.drop(spy_df.index[0:scale_window], inplace = True)
    spy_df.reset_index(drop=True, inplace=True)

    return spy_df

def AddVar(num_days, var_to_pred, TickerDF):
    var = "%s_%d_day_forecast" %(var_to_pred, num_days)
    TickerDF[var] = np.nan
    
    for i in range(num_days, len(TickerDF[var])):
        TickerDF[var][i - num_days] = TickerDF[var_to_pred][i]

        if TickerDF[var_to_pred][i] > TickerDF[var_to_pred][i - num_days]:
            TickerDF[(var + "_up")][i - num_days] = 1
        else:
            TickerDF[(var + "_up")][i - num_days] = 0
    
    TickerDF.drop(TickerDF.index[-num_days:], inplace = True)
    TickerDF.reset_index(drop = True, inplace = True)

    return TickerDF

# start = '2013-01-01'
# end = '2017-01-01'
# window = 30

# df = StandardData(window, start, end, "stock_market_data-SPY.csv")
# df = AddVar(10, "Close_scaled", df)
# Seed_RNG(random_seed)
# df = runData(df, 3)



