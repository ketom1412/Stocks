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

def Stats(TickDF, residuals, actuals):
    print("Stats for ", residuals)
    error_count = 0
    false_pos_count = 0
    false_neg_count = 0
    up_preds = 0
    down_preds = 0

    for i in range(0, len(TickDF[actuals])):
        if TickDF[residuals][i] != TickDF[actuals][i]:
            error_count+=1
        if TickDF[residuals][i] > TickDF[actuals][i]:
            false_pos_count+=1
        if TickDF[residuals][i] < TickDF[actuals][i]:
            false_neg_count+=1
        if TickDF[residuals][i] == 1.0:
            up_preds+=1
        else:
            down_preds+=1

    error_rate = error_count / float(len(TickDF[residuals]))
    false_pos_rate = false_pos_count / float(up_preds)
    false_neg_rate = false_neg_count / float(down_preds)

    print(error_count, len(TickDF[residuals]))
    print("Error rate: ", error_rate)
    print("Accuracy: ", 1 - error_rate)
    print("False Positive Rate: ", false_pos_rate)
    print("False Negative Rate: ", false_neg_rate)
    print("Count of upward predictions: ", up_preds)
    print("Count of down predictions: ", down_preds)
