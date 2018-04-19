import numpy as np
import matplotlib.pyplot as plt

class perceptron(object):

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



