
import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp
from random import seed
from random import random
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
class mlp(object):

    def __init__(self, input_size, lr=0.1, epochs=1000,n_inputs=2, n_hidden=2, n_outputs=1):
        self.W = np.zeros(input_size+1)
        self.E = np.zeros(input_size + 1)
        #add one for bias
        self.epochs = epochs
        self.lr = lr
        self.erros_gragh = []
        network = list()
        hidden_layer = [{'weights': np.array([random() for i in range(n_inputs + 1)])} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights': np.array([random() for i in range(n_hidden + 1)])} for i in range(n_outputs)]
        network.append(output_layer)

    def sum_weights(self, weight, x):
        return weight.T.dot(x)

    def activate(weights, inputs):
        return weights.T.dot(inputs)

    # Transfer neuron activation
    def transfer(activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def predict(self, x):
        return self.W.T.dot(x)

    # Forward propagate input to a network output
    def forward_propagate(self,network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                x = np.insert(inputs,0, 1)
                activation = self.activate(neuron['weights'],x)
                new_inputs.append(self.transfer(activation))
            inputs = np.array(new_inputs)
        return inputs

    def pattern(self, X, d):
        for k in range(self.epochs):
            new_inputs = []
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1) #insert 1 pos em 0 [1,X[i]]
                y = self.predict(x)
                y = self.transfer(y)
                e = (d[i]-y)
                self.erros_gragh.append(e)
                self.W = self.W + self.lr *e* x

    def batch(self, X, d):
        for k in range(self.epochs):
            self.E = np.zeros(4)
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)  # insert 1 pos em 0 [1,X[i]]
                y = self.predict(x)
                e = (d[i] - y)
                self.erros_gragh.append(e)
                self.E = self.E + self.lr * e * x
            self.W = self.W + self.E/d.shape[0]

    # Train a network for a fixed number of epochs
    def train_network(self,network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(network, row)
                expected = np.array([0 for i in range(n_outputs)])
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                backward_propagate_error(network, expected)
                update_weights(network, row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    def printTrain(self,X,d):
        for i in range(d.shape[0]):
            x = np.insert(X[i], 0, 1)  # insert 1 pos em 0 [1,X[i]]
            y = self.predict(x)
            print(str(X[i])+" = "+str(y))

    def graph_err(self , title):
        squart_err =[]
        plt.figure()
        plt.plot(self.erros_gragh)
        plt.plot(math.sqrt(math.pow(self.erros_gragh,2)))
        plt.xscale('log')
        plt.xlabel('Escala logaritima de iterações')
        plt.ylabel('Valor do erro')
        plt.title('Erros durante treinamento '+title)
        plt.grid(True)
        plt.legend()
        plt.show()