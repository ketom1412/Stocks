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
from Node import Node
from FeedForwardNet import FeedForwardNet

def testCase1():
    test_network = FeedForwardNet(5, 2, [7, 10], 1, 0.1)

    training_vector = [0.45, 0.5, 0.55, 0.6, 0.65]
    training_output = [1.0]
    test_network.debug_info()

    for x in range(5000):
        test_network.FeedForward(training_vector, training_output, Training = True)
        if(x%1000 == 0):
            test_network.debug_info()

def testCase2():
    test_network = FeedForwardNet(5, 2, [7, 10], 2, 0.1)

    training_vector = [0.45, 0.5, 0.55, 0.6, 0.65]
    training_output = [1.0, 0.5]
    test_network.debug_info()

    for x in range(5000):
        test_network.FeedForward(training_vector, training_output, Training = True)
        if(x%1000 == 0):
            test_network.debug_info()

    test_network.debug_info()

def testCase3():
    test_network = FeedForwardNet(5, 1, [7], 2, 0.15)

    training_vector = [0.45, 0.5, 0.55, 0.6, 0.65]
    training_output = [1.0, 0.75]
    test_network.debug_info()

    for x in range(5000):
        test_network.FeedForward(training_vector, training_output, Training = True)
        if(x%1000 == 0):
            test_network.debug_info()

    test_network.debug_info()

testCase2()