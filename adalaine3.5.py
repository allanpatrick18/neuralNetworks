import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def neuronio(x, weights):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * x[i+2]
	return activation

def training_weights(train):
    error=0.0
    sum_error =1
    epoch=1
    while(sum_error/15>0.0001 and epoch<10000):
            epoch += 1
            sum_error = 0
            for row in train:
               error = row[-1]-neuronio(row,weight)
               weight[-1] = weight[-1] + step * error
               sum_error+= error**2
               for i in range(len(weight)-1):
                   weight[i] = weight[i] + step * error * row[i+2]
    print("Epocas: "+str(epoch))
    # print(weight)

def training_weights_bateladas(train):
        error = 1.0
        epoch=0
        p = len(train)
        while(error/p>0.000001 and epoch<100000):
            sum_error = [0.0, 0.0, 0.0]
            error = 0.0
            lim_error = 0.0
            p=0
            epoch+=1
            for row in train:
                error = row[-1] - neuronio(row, weight)
                lim_error += step * error
                for i in range(len(weight)-1):
                    sum_error[i] = sum_error[i] + step * error * row[i + 2]
                p+=1
            for i in range(len(weight)-1):
                weight[i] = weight[i] + sum_error[i]/p
                error += sum_error[i]**2
            weight[-1] = weight[-1] + lim_error/p

        print("Epocas: " + str(epoch))
        # print(weight)

def test_validacao():
    dataset_treinamento =create_dataset(0, 2 * np.pi, 30)
    dataset_validacao = create_dataset(0.32, 2.1 * np.pi, 10)
    error =[]
    index=0
    result_test=[]
    training_weights_bateladas(dataset_treinamento)
    weight_batelada = weight[:]
    z1 = np.linspace(0.32, 2.1 * np.pi, 10)
    f = (-1) * np.pi + 0.565 * np.sin(z1) + 2.674 * np.cos(z1) + 0.674 * z1

    for row in dataset_validacao:
        result_test.append(neuronio(row, weight_batelada))
        error.append(math.sqrt((f[index]-neuronio(row, weight_batelada))**2))
        index += 1

    print(max(error))
    plt.figure(2)
    plt.plot(z1, f,label='f(z)')
    plt.plot(z1, result_test, label='Resuldado treinamento')
    plt.plot(z1, error, label='Erro trinemento e f(z)')
    plt.legend()
    plt.show()


def create_dataset(init,end,qtd):
    z1 = np.linspace(init,end,qtd)
    dataset_test = []
    for k in range(len(z1)):
        x_1 = np.sin(z1[k])
        x_2 = np.cos(z1[k])
        x_3 = z1[k]
        f = (-1) * np.pi + 0.565 * x_1 + 2.674 * x_2 + 0.674 * x_3
        dat = [k + 1, round(z1[k], 3), round(x_1, 3), round(x_2, 3), round(x_3, 3), round(f, 3)]
        print(dat)
        dataset_test.append(dat)

dataset=[]
z = np.linspace(0, 2*np.pi, 15)
for k in range(len(z)):
    x_1 = np.sin(z[k])
    x_2 = np.cos(z[k])
    x_3 = z[k]
    f = (-1)*np.pi + 0.565*x_1 + 2.674* x_2 + 0.674*x_3
    dat=[k+1,round(z[k],3),round(x_1,3),round(x_2,3),round(x_3,3),round(f,3)]
    # print(dat)
    dataset.append(dat)

step =0.1
weight=[0.2,-0.1,0.1,0.5]
print("-------------------------------")
print("3.5 Refaça o treinamento com 50 padrões")
print("Pesos iniciais:")
print(weight)
weight_batelada= weight[:]
print("Pesos Ajustado:")
# test_validacao()
print(weight_batelada)
test_validacao()
