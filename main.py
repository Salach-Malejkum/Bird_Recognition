import CNN
import data

if __name__ == '__main__':
    training, validation, test = data.get_training_data()
    neural_network = CNN.ConvolutionalNeuralNetwork()
    batchX, batchy = training.next()
    print(batchX, batchy)