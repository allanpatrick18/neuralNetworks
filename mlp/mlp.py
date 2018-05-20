
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
        self.output = []
        self.backprop = []
        self.backpropsaida = []
        self.Ws =[0,0,0]
        self.Wh1 = [0, 0, 0]
        self.Wh2 = [0, 0, 0]

    def sum_weights(self, weight, x):
        return weight.T.dot(x)

    def activate(self,weights, inputs):
        return weights.T.dot(inputs)

    # Transfer neuron activation
    def transfer(self,activation):
        return 1.0 / (1.0 + math.exp(-activation))

    def predict(self, x):
        return self.W.T.dot(x)

    # Forward propagate input to a network output

    def forward_propagate(self, row):
        inputs = row
        self.output = []
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                x = np.insert(inputs,0, 1)
                activation = self.activate(neuron['weights'],x)
                new_inputs.append(self.transfer(activation))
                neuron['output'] = self.transfer(activation)
                self.output.append(self.transfer(activation))
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
    def update_weights(self, row, l_rate,erro):

        for i in range(len(self.network)):
            inputs = row
            if i != 0:
              for neuron in self.network[i]:
                for k, back in enumerate(self.backprop):
                    neuron['weights'][k+1] += l_rate * back * self.output[k]
                neuron['weights'][0] += l_rate * erro
            else:
              for neuron in self.network[i]:
                for j, back in enumerate(self.backprop):
                    neuron['weights'][j+1] += l_rate * back * inputs[j]
                neuron['weights'][0] += l_rate * erro

    # Backpropagate error and store in neurons
    def backward_propagate_error(self,erro):
        self.backprop = []
        back =[]
        for i in range(len(self.network)-1):
              for neuron in self.network[i]:
                for k in range(3):
                    back.append(neuron['weights'][k] * erro * self.transfer_derivative(self.output[-1]))
        self.backprop = np.array(back)
    # Train a self.network for a fixed number of epochs
    def train_network(self, train, expected , l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            for r, row in enumerate(train):
                outputs = self.forward_propagate(row)
                erro = expected[r] - self.output[-1]
                self.backward_propagate_error(float(expected[r]) ,row)
                self.update_weights(row, l_rate,erro)

    def train_network_bacth(self, train, expected, l_rate, n_epoch):
        for epoch in range(n_epoch):
            self.Ws = np.array([0.0, 0.0, 0.0])
            self.Wh1 = np.array([0.0, 0.0, 0.0,0.0, 0.0, 0.0])
            self.Wh2 = np.array([0.0, 0.0, 0.0])
            re = []
            sum_erro =0.0
            for r, row in enumerate(train):
                self.forward_propagate(row)
                erro = expected[r] - self.output[-1]
                for i in range(len(self.output) - 1):
                    self.Ws[i+1]+= erro * self.transfer_derivative(self.output[-1])*self.output[i]*l_rate
                    self.Ws[0]+= erro * self.transfer_derivative(self.output[-1])*1*l_rate
                # implementar o erro back probagado
                # ebp = e*g'*w1
                # wh1 = wh1 = ebp*g'(output(-1))*x[i]
                x = np.insert(row, 0, 1)  # insert 1 pos em 0 [1,X[i]]
                x = np.insert(x,3,x)
                self.backward_propagate_error(erro)
                for i, ebp in enumerate(self.backprop):
                    self.Wh1[i]+= ebp*self.transfer_derivative(self.output[-1])*x[i]*l_rate
                sum_erro += erro**2
            for i, layer in enumerate(self.network):
                for j, neuron in enumerate(layer):
                    if(i==0):
                       index = [0, 1, 2]
                       if len( self.Wh1)>5:
                        self.Wh2 = self.Wh1[0:3]
                        self.Wh1 = np.delete(self.Wh1,index)
                       else:
                        self.Wh2 = self.Wh1
                       neuron['weights'] += self.Wh2/4
                    else:
                       neuron['weights'] += self.Ws/4
            print("erros: "+ str(sum_erro))




# Test training backprop algorithm
if __name__ == '__main__':

    seed(1)
    dataset = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    expected = np.array([0.1,0.9,0.9,0.1])
    casa = mlp(2, 2, 1)
    casa.train_network_bacth(dataset,expected, 1, 10000)
    for row in dataset:
        print(casa.forward_propagate(row))
    for layer in casa.network:
        print(layer)