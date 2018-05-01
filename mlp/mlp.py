
import numpy as np
import matplotlib.pyplot as plt
import math
from random import random, seed


class mlp(object):

    def __init__(self,n_inputs=2, n_hidden=2, n_outputs=1):
        self.network = list()
        hidden_layer = [{'weights': np.array([random() for i in range(n_inputs + 1)])} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': np.array([random() for i in range(n_hidden + 1)])} for i in range(n_outputs)]
        self.network.append(output_layer)


    def sum_weights(self, weight, x):
        return weight.T.dot(x)

    def activate(self,weights, inputs):
        return weights.T.dot(inputs)

    # Transfer neuron activation
    def transfer(self,activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def predict(self, x):
        return self.W.T.dot(x)

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                x = np.insert(inputs,0, 1)
                activation = self.activate(neuron['weights'],x)
                new_inputs.append(self.transfer(activation))
                neuron['output'] = self.transfer(activation)
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

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # Update self.network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs =  row
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][0] += l_rate * neuron['delta']

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                        errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected - neuron['output'])
            for j in range(len(layer)):
                    neuron = layer[j]
                    neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    # Train a self.network for a fixed number of epochs
    def train_network(self, train, expected , l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            for r, row in enumerate(train):
                outputs = self.forward_propagate(row)
                sum_error += sum(- outputs)**2
                self.backward_propagate_error(float(expected[r]))
                self.update_weights(row, l_rate)
           # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    def printTrain(self,X,d):
        for i in range(d.shape[0]):
            x = np.insert(X[i], 0, 1)  # insert 1 pos em 0 [1,X[i]]
            y = self.predict(x)
            print(str(X[i])+" = "+str(y))

    def graph_err(self,title):
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

# Test training backprop algorithm
if __name__ == '__main__':

    seed(1)
    dataset = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    expected = np.array([0,1,1,0])
    casa = mlp(2, 2, 1)
    casa.train_network(dataset,expected, 0.001, 10000000)
    for row in dataset:
        print(casa.forward_propagate(row))
    for layer in casa.network:
        print(layer)