import numpy as np

from neuro.perceptron import perceptron

if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 0, 0, 1])

    perceptro = perceptron(input_size=2)
    perceptro.fit(X, d)
    print(perceptro.W)