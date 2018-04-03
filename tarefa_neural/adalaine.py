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

def test_function():
    z1 = np.linspace(0.2, 2.1 * np.pi, 15)
    dataset_tes=[]
    for k in range(len(z)):
        x_1 = np.sin(z1[k])
        x_2 = np.cos(z1[k])
        x_3 = z1[k]
        f = (-1) * np.pi + 0.565 * x_1 + 2.674 * x_2 + 0.674 * x_3
        dat = [k + 1, round(z1[k], 3), round(x_1, 3), round(x_2, 3), round(x_3, 3), round(f, 3)]
        # print(dat)
        dataset_tes.append(dat)
    f = (-1) * np.pi + 0.565 * np.sin(z1) + 2.674 * np.cos(z1) + 0.674 * z1
    error =[]
    index=0
    result_test=[]
    for row in dataset_tes:
        result_test.append(neuronio(row, weight_padrao))
        error.append(math.sqrt((f[index]-neuronio(row, weight_padrao))**2))
        index += 1
    print(max(error))
    plt.figure(2)
    plt.plot(z1, f,label='f(z)')
    plt.plot(z1, result_test, label='Resuldado treinamento')
    plt.plot(z1, error, label='Erro trinemento e f(z)')
    plt.legend()
    plt.show()

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
        dataset_test.append(dat)

    return dataset_test

def create_fun():
    f = (-1) * np.pi + 0.565 * np.sin(z) + 2.674 * np.cos(z) + 0.674 * z
    plt.figure(1)
    plt.subplot(311)
    plt.plot(z, f)

    plt.xlabel('')
    plt.ylabel('')
    plt.title('função de f(z)')
    plt.grid(True)

    result = []
    for row in dataset:
        result.append(neuronio(row, weight_padrao))

    plt.subplot(312)
    plt.ylabel('padrao')
    plt.plot(z, result)
    plt.grid(True)
    plt.xlabel(
        'w1 =' + str(round(weight_padrao[0], 2)) + " w2= " + str(round(weight_padrao[1], 2)) + " w3= " + str(round(weight_padrao[2], 2)))
    result2 = []
    for row in dataset:
        result2.append(neuronio(row, weight_batelada))

    plt.subplot(313)
    plt.plot(z, result2)
    plt.grid(True)
    plt.ylabel('batelada')
    plt.xlabel(
        'w1 =' + str(round(weight_batelada[0], 2)) + " w2= " + str(round(weight_batelada[1], 2)) + " w3= " + str(round(weight_batelada[2], 2)))
    plt.show()

    error1 = []
    error2 = []
    for e in range(len(result)):
        error1.append(math.sqrt((f[e] - result[e])**2))
        error2.append(math.sqrt((f[e] - result2[e])**2))

    norm1 = [float(i) / sum(error1) for i in error1]
    norm2 = [float(i) / sum(error2) for i in error2]
    norm1 = [float(i) / max(error1) for i in error1]
    norm2 = [float(i) / max(error2) for i in error2]


    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(norm1))
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, norm1, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Erro Padrao')

    rects2 = plt.bar(index + bar_width, norm2, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Erro Batelada')

    plt.xlabel('Erros por pontos')
    plt.ylabel('Erro norm')
    plt.title('Erros entre treinamento vs pontos f(z)')
    plt.legend()

    plt.tight_layout()
    plt.show()


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

print("Trainning Adalaine Por Padrão:")
print("Pesos iniciais:")
weight=[0.2,-0.1,0.1,0.5]
print(weight)
print("Pesos Ajustado:")
step =0.1
training_weights(dataset)
weight_padrao = weight[:]
print(weight_padrao)

print("-------------------------------")
print("Trainning Adalaine Por Batelada:")
print("Pesos iniciais:")
weight=[0.2,-0.1,0.1,0.5]
print(weight)
print("Pesos Ajustado:")
training_weights_bateladas(dataset)
weight_batelada = weight[:]
print(weight_batelada)

print("-------------------------------")
print("3.4 Teste com padrões diferentes dos treinados:")
print("Pesos iniciais:")
weight=[0.2,-0.1,0.1,0.5]
print(weight)
print("Pesos Ajustado:")
training_weights_bateladas(dataset)
weight_batelada = weight[:]
print(weight_batelada)
# create_fun()
test_function()

print("-------------------------------")
print("3.5 Refaça o treinamento com 50 padrões")
print("Pesos iniciais:")
weight=[0.2,-0.1,0.1,0.5]
print(weight)
print("Pesos Ajustado:") test_validacao()
print(weight_batelada)



