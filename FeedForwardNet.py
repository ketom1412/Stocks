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

class FeedForwardNet(object):
    def __init__(self, no_of_inputs, no_of_hidden_layers, nodes_in_hidden, no_of_outputs, learning_rate):
        self.number_of_inputs = no_of_inputs
        self.number_of_hidden_layers = no_of_hidden_layers
        self.hidden_nodes = []
        self.hidden_outputs = []
        self.hidden_nodes.append(np.array([Node(no_of_inputs) for x in range(nodes_in_hidden[0])]))
        self.hidden_outputs.append(np.array([0.0 for x in range(nodes_in_hidden[0])]))

        if no_of_hidden_layers > 1:
            for i in range(1, len(nodes_in_hidden)):
                self.hidden_nodes.append(np.array([Node(nodes_in_hidden[i-1]) for x in range(nodes_in_hidden[i])]))
                self.hidden_outputs.append(np.array([0.0 for x in range(nodes_in_hidden[i])]))

        self.hidden_node_list = nodes_in_hidden
        self.output_layer = np.array([Node(nodes_in_hidden[-1]) for i in range(no_of_outputs)])
        self.number_of_outputs = no_of_outputs
        self.network_output = np.array([0.0 for i in range(no_of_outputs)])
        self.errors = np.array([0.0 for i in range(no_of_outputs)])
        self.alpha = learning_rate