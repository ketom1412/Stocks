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

class Node(object):
    def __init__(self, number_of_input):
        self.inputs = number_of_input
        self.bias = random.uniform(0.0, 1.0)
        self.weights = np.array([random.uniform(0.0, 1.0)] * number_of_input)
        self.output = 0.0

    def output(self):
        return self.output

    def debug_info(self):
        info = "Bias: %f ; Weights:" %(self.bias)
        for w in self.weights:
            info+="%f," %(w)
        return info

    def getWeightAtIdx(self, idx):
        return self.weights[idx]

    def getBias(self):
        return self.bias

    def calculateActivity(self, input_vector):
        #linear basis function
        activity = self.bias
        activity += np.dot(input_vector, self.weights)
        return activity

    def activationFunction(self, input_value):
        #sigmoid activation
        return 1.0/(1.0 + math.exp(-input_value))

    def calculate(self, input_vector):
        activity_value = self.calculateActivity(input_vector)
        self.output = self.activationFunction(activity_value)

    def updateWeights(self, alpha, delta):
        adjustments = self.output * alpha * delta
        self.bias = self.bias + adjustments
        self.weights = self.weights + adjustments