import numpy as np

from neuro.perceptron import perceptron

if __name__ == '__main__':
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    result_and = np.array([0, 0, 0, 1])
    result_or = np.array([0, 1, 1, 1])


    while(1):
        print("=================")
        print("1-Porta AND Padrão")
        print("2-Porta AND Batelada")
        print("3-Porta OR Padrão")
        print("4-Porta OR Batelada")
        q = int(input())

        if q == 1:
            print("Porta AND Padrão")
            port_and = perceptron(input_size=2)
            port_and.fit(X, result_and)
            port_and.printTrain(X, result_and)
            port_and.graph_err()
            print(port_and.W)

        elif q == 2:
            print("==========")
            print("2-Porta AND Batelada")
            port_and = perceptron(input_size=2)
            port_and.batelada(X, result_and)
            port_and.printTrain(X, result_and)
            port_and.graph_err()
            print(port_and.W)

        elif q == 3:
            print("3-Porta OR Padrão")
            print("==========")
            port_or = perceptron(input_size=2)
            port_or.fit(X, result_or)
            port_or.printTrain(X, result_or)
            port_and.graph_err()
            print(port_or.W)
        elif q == 4:
            print("4-Porta OR Batelada")
            print("==========")
            port_or = perceptron(input_size=2)
            port_or.batelada(X, result_or)
            port_or.printTrain(X, result_or)
            port_and.graph_err()
            print(port_or.W)

        else:
            nome = None
            print("Valor invalido")

