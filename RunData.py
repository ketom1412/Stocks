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

def runData(df, input_vars, output_vars, hidden_nodes_list, learning_rate, training_window, iterations):

    residuals = []
    
    for x in range(len(output_vars)):
        residuals.append([-1.0 for i in range(training_window)])
    
    for i in range(training_window, len(df["Close_scaled"])):
        stock_net = FeedForwardNet(len(input_vars), len(hidden_nodes_list), hidden_nodes_list, len(output_vars), learning_rate)

        for iterations in range(iterations):
            for x in range(training_window - 1):
                idx = i - training_window + x
                training_list = []

                for var in input_vars:
                    training_list.append(float(df[var][idx]))
                
                training_vector = np.array(training_list)
                out_list = []

                for var in output_vars:
                    out_list.append(float(df[var][idx]))

                training_output = np.array(out_list)
                stock_net.FeedForward(training_vector, training_output, Training = True)
        
        pred_list = []
        
        for var in input_vars:
            pred_list.append(float(df[var][idx]))

        pred_vector = np.array(pred_list)
        pred_closes = stock_net.FeedForward(pred_vector)

        for x in range(len(output_vars)):
            if pred_closes[x] > df["Close_scaled"][i]:
                residuals[x].append(1)
            else:
                residuals[x].append(0)

    for x in range(len(output_vars)):
        df[(output_vars[x] + "_residuals")] = residuals[x]

    return df