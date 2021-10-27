import CNN
import data
from sklearn.metrics import classification_report
import numpy as np


def get_optimization_data(training, validation, test):
    loss_test = []
    acc_test = []
    for layers in range(1):
        for neurons in range(5, 20000, 1000):
            neural_network = CNN.ConvolutionalNeuralNetwork(neurons)
            history = neural_network.train(training, validation, 100)
            final_res = neural_network.evaluate(test)

            acc_test.append(final_res[1])
            loss_test.append(final_res[0])
    print(acc_test)
    print(loss_test)


if __name__ == '__main__':
    training, validation, test, class_names = data.get_training_data_10sec()

    neural_network = CNN.ConvolutionalNeuralNetwork(2005)
    history = neural_network.train(training, validation, 100)
    y_prob = neural_network.predict(test)
    y_classes = y_prob.argmax(axis=-1)
    print(classification_report(np.array(test.labels), np.array(y_classes), labels=np.unique(y_classes), target_names=class_names))
