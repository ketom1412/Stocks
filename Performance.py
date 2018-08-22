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

def DailyPerform(TickDF):
    print("Stats for daily predictions")
    error_count = 0
    false_pos_count = 0
    false_neg_count = 0
    up_preds = 0
    down_preds = 0

    for i in range(0, len(TickDF["tomorrow_up"])):
        if TickDF["daily_residuals"][i] != TickDF["tomorrow_up"][i]:
            error_count+=1
        if TickDF["daily_residuals"][i] > TickDF["tomorrow_up"][i]:
            false_pos_count+=1
        if TickDF["daily_residuals"][i] < TickDF["tomorrow_up"][i]:
            false_neg_count+=1
        if TickDF["daily_residuals"][i] == 1.0:
            up_preds+=1
        else:
            down_preds+=1

    error_rate = error_count / float(len(TickDF["daily_residuals"]))
    false_pos_rate = false_pos_count / float(up_preds)
    false_neg_rate = false_neg_count / float(down_preds)

    print(error_count, len(TickDF["daily_residuals"]))
    print("Error rate: ", error_rate)
    print("Accuracy: ", 1 - error_rate)
    print("False Positive Rate: ", false_pos_rate)
    print("False Negative Rate: ", false_neg_rate)
    print("Count of upward predictions: ", up_preds)
    print("Count of down predictions: ", down_preds)

def WeekPerform(TickDF):
    print("Stats for weekly predictions")
    error_count = 0
    false_pos_count = 0
    false_neg_count = 0
    up_preds = 0
    down_preds = 0

    for i in range(0, len(TickDF["week_up"])):
        if TickDF["weekly_residuals"][i] != TickDF["week_up"][i]:
            error_count+=1
        if TickDF["weekly_residuals"][i] > TickDF["week_up"][i]:
            false_pos_count+=1
        if TickDF["weekly_residuals"][i] < TickDF["week_up"][i]:
            false_neg_count+=1
        if TickDF["weekly_residuals"][i] == 1.0:
            up_preds+=1
        else:
            down_preds+=1

    error_rate = error_count / float(len(TickDF["weekly_residuals"]))
    false_pos_rate = false_pos_count / float(up_preds)
    false_neg_rate = false_neg_count / float(down_preds)

    print(error_count, len(TickDF["weekly_residuals"]))
    print("Error rate: ", error_rate)
    print("Accuracy: ", 1 - error_rate)
    print("False Positive Rate: ", false_pos_rate)
    print("False Negative Rate: ", false_neg_rate)
    print("Count of upward predictions: ", up_preds)
    print("Count of down predictions: ", down_preds)