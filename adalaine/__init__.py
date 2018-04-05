
import numpy as np

from adalaine.adalaineFinal import adalineFinal

if __name__ == '__main__':


    def create_dataset(init, end, qtd):
        z1 = np.linspace(init, end, qtd)
        dataset = []
        wished =[]
        for k in range(len(z1)):
            x_1 = np.sin(z1[k])
            x_2 = np.cos(z1[k])
            x_3 = z1[k]
            f = (-1) * np.pi + 0.565 * x_1 + 2.674 * x_2 + 0.674 * x_3
            wished.append(f)
            dat = [x_1, x_2, x_3]
            dataset.append(dat)
        return dataset
        # wished_result = np.array(wished)
        # dataset_test = np.array(dataset)

    def create_datasetw(init, end, qtd):
        z1 = np.linspace(init, end, qtd)
        dataset = []
        wished =[]
        for k in range(len(z1)):
            x_1 = np.sin(z1[k])
            x_2 = np.cos(z1[k])
            x_3 = z1[k]
            f = (-1) * np.pi + 0.565 * x_1 + 2.674 * x_2 + 0.674 * x_3
            wished.append(f)
            dat = [x_1, x_2, x_3]
            dataset.append(dat)
        return wished

    dataset_test = np.array(create_dataset(0,2*np.pi,15))
    wished_result = np.array(create_datasetw(0,2*np.pi,15))
    adalaine = adalineFinal(input_size=3)
    adalaine.fit(dataset_test, wished_result)
    adalaine.batelada(dataset_test, wished_result)
    adalaine.printTrain(dataset_test, wished_result)
    adalaine.graph_err()
    print(adalaine.W)



    