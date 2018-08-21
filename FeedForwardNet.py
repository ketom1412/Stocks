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

    def getNetOutputs(self):
        return self.network_output

    def debug_info(self):
        print("Number of Inputs: ", self.number_of_inputs)
        print("Number of Hidden Nodes: ", self.hidden_node_list)
        print("Number of Outputs: ", self.number_of_outputs)

        print("Hidden Layer Node Weights:")
        count = 1
        for layer in self.hidden_nodes:
            print("Hidden Layer", count, ": ")
            count+=1
            for node in layer:
                print(node.debug_info())

        print("Output Layer Node Weights:")
        for node in self.output_layer:
            node.debug_info()

        print("Output from network:")
        print(self.network_output)
        print("Network Errors:")
        print(self.errors)

    def FeedForward(self, input_vector, true_outputs = None, Training = False):
        for y in range(len(self.hidden_nodes)):
            layer = self.hidden_nodes[y]
            output = self.hidden_outputs[y]

            for x in range(len(layer)):
                layer[x].calculate(input_vector)
                output[x] = layer[x].output
            
            input_vector = output

        hidden_outputs = self.hidden_outputs[-1]

        for x in range(self.number_of_outputs):
            self.output_layer[x].calculate(hidden_outputs)
            self.network_output[x] = self.output_layer[x].output
            self.errors[x] = true_outputs[x] - self.output_layer[x].output

        if Training:
            self.BackPropagate()
        else:
            return self.network_output

    def BackPropagate(self):
        deltas_for_layer = []

        for i in range(self.number_of_outputs):
            output = self.network_output[i]
            delta_o = self.errors[i] * (output * (1.0 - output))
            self.output_layer[i].updateWeights(self.alpha, delta_o)
            deltas_for_layer.append(delta_o)

        prev_layer = self.output_layer

        for y in range(len(self.hidden_nodes)):
            layer = self.hidden_nodes[-(1+y)]
            prev_layer_factor = 0
            current_layer_deltas = []

            for j in range(len(layer)):
                output = layer[j].output
                
                for x in range(len(prev_layer)):
                    prev_layer_factor += prev_layer[x].getWeightAtIdx(j) * deltas_for_layer[x]

                delta_h = (output * (1.0 - output)) * prev_layer_factor
                current_layer_deltas.append(delta_h)
                layer[j].updateWeights(self.alpha, delta_h)
            
            prev_layer = layer
            deltas_for_layer = current_layer_deltas
