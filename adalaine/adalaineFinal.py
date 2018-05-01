import numpy as np
import matplotlib.pyplot as plt
import math

class adalineFinal(object):

    def __init__(self, input_size, lr=0.1, epochs=1000):
        self.W = np.zeros(input_size+1)
        self.E = np.zeros(input_size + 1)
        #add one for bias
        self.epochs = epochs
        self.lr = lr
        self.erros_gragh = []

    def predict(self, x):
        return self.W.T.dot(x)

    def pattern(self, X, d):
        for k in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1) #insert 1 pos em 0 [1,X[i]]
                y = self.predict(x)
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

