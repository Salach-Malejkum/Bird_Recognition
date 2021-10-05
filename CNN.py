import keras
from keras.models import Sequential, Input, Model
from keras.layers import Softmax, ReLU, LeakyReLU, Dense


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(LeakyReLU(alpha=0.1))
        self.network.add(Dense(4, activation="softmax"))  # output layer

    def train(self):
        pass

    def classify(self):
        pass
