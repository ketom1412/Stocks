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
from bs4 import BeautifulSoup as bs
from pandas_datareader import data

data_source = 'alphavantage'
api_key = 'S13H3V357VQ534EE'

class Node(objec):
    def __init__(self, numbers_of_input):
        self.inputs = numbers_of_input
        self.bias = random.uniform(0.0, 1.0)
        self.weights = np.array([random.uniform(0.0, 1.0)] * numbers_of_input)
        self.outputs = 0.0

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

get_alphavantage_data('googl')
#for i in get_historical_data('googl', 10):
#    print(i)

