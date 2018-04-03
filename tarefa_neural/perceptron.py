
def port_and(x):
	activation = weights[-1]
	for i in range(len(x)-2):
		activation += weights[i] * x[i]
	return fun_activation(activation)

def port_or(x):
    activation = weights[-2]
    for i in range(len(x)-2):
        activation += weights[i+2] * x[i]
    return fun_activation(activation)

def fun_activation(sum):
    return 1.0 if sum >= 0 else 0.0


def training_weights(train):
    error=[0,0]
    sum_error1 = 1
    sum_error2 = 1
    epoch=1
    while(sum_error1!=0 and sum_error2!=0):
            sum_error1 = 0
            sum_error2 = 0
            epoch += 1
            for row in train:
               error[0] = row[-1] -port_and(row)
               error[1] = row[-2] -port_or(row)
               weights[-1] = weights[-1] + step * error[0]
               weights[-2] = weights[-2] + step * error[1]
               sum_error1 += (error[1])** 2
               sum_error2 += (error[1])** 2
               for i in range(len(error)):
                   weights[i] = weights[i] + step * error[0] * row[i]
                   weights[i+2] = weights[i+2] + step * error[1] * row[i]

    print(epoch)
    print(weights)

def training_weights_batelada(train):
        sum_error = [0, 0, 0, 0]
        error = [0, 0]
        for p in range(repeat):
            for row in train:
                error[0] = row[-1] - port_and(row)
                error[1] = row[-2] - port_or(row)
                weights[-1] = weights[-1] + step * error[0]
                weights[-2] = weights[-2] + step * error[1]
                for i in range(len(error)):
                    sum_error[i] = sum_error[i] + step * error[0] * row[i]
                    sum_error[i + 2] = sum_error[i + 2] + step * error[1] * row[i]

        for i in range(len(sum_error)):
            weights[i] = weights[i] + (sum_error[i]/repeat)

        print(weights)






# Calculate weights
dataset = [[0, 0, 0, 0],
           [0, 1, 0, 1],
           [1, 0, 0, 1],
           [1, 1, 1, 1]]


theta=0
weights=[1.0,1.0,1.0,1.0,1,1]
step =0.1
repeat =20


training_weights(dataset)
for col in dataset:
    print("| "+ str(col[0])+ " | "+str(col[1])+" | "+str(port_or(col)))

print("------------")
for col in dataset:
    print("| "+ str(col[0])+ " | "+str(col[1])+" | "+str(port_and(col)))
# training_weights_batelada(dataset)
