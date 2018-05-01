from random import seed
import numpy as np
from mlp.mlp import mlp

if __name__ == '__main__':

    seed(1)
    dataset = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    network = mlp(2, 2, 1)
    network.train_network(network, dataset, 0.5, 20, 1)
    for layer in network:
        print(layer)
