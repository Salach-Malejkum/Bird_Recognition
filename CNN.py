import keras
from keras.models import Sequential, Input, Model
from keras.layers import Softmax, ReLU, LeakyReLU, Dense, Flatten


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(Flatten(input_shape=[1000, 400]))
        self.network.add(LeakyReLU(alpha=0.1))
        self.network.add(Dense(4, activation="softmax"))  # output layer

        self.network.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    def train(self):
        pass

    def classify(self):
        pass
