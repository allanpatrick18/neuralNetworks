import numpy as np
import matplotlib.pyplot as plt

class mlp(object):

    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        #add one for bias
        self.epochs = epochs
        self.lr = lr
        self.erros_gragh = []

    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def activate(self,weights, inputs):
        return self.weights.T.dot(inputs)

    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def forward_propagate(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                x = np.insert(inputs, 0, 1)  # insert 1 pos em 0 [1,X[i]]
                activation = self.activate(neuron['weights'], x)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = np.array(new_inputs)
        return inputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self,network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    # Train a network for a fixed number of epochs
    def train_network(self, network, train, l_rate, n_epoch, n_outputs, expected):
        for epoch in range(n_epoch):
            sum_error = 0
            for k, row in train:
                outputs = self.forward_propagate(network, row)
                sum_error += sum(expected[k] - outputs)
                backward_propagate_error(network, expected)
                update_weights(network, row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


    def fit(self, X, d):
        for k in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1) #insert 1 pos em 0 [1,X[i]]
                y = self.predict(x)
                e = d[i] - y
                self.erros_gragh.append(e)
                self.W = self.W + self.lr * e * x

    def batelada(self, X, d):
        sum_err=0
        for k in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1) #insert 1 pos em 0 [1,X[i]]
                y = self.predict(x)
                e = d[i] - y
                sum_err +=e*x
                self.erros_gragh.append(e)
            self.W = self.W + self.lr * sum_err

    def printTrain(self,X,d):
        for i in range(d.shape[0]):
            x = np.insert(X[i], 0, 1)  # insert 1 pos em 0 [1,X[i]]
            y = self.predict(x)
            print(str(X[i])+" = "+str(y))

    def graph_err(self):
        plt.figure()
        plt.plot(self.erros_gragh)
        plt.xscale('log')
        plt.xlabel('Epocas log')
        plt.ylabel('Valor do erro')
        plt.title('Erros durante treinamento')
        plt.grid(True)
        plt.legend()
        plt.show()