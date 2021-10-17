import keras
from keras.models import Sequential, Input, Model
from keras.layers import Softmax, ReLU, LeakyReLU, Dense, Flatten, Rescaling


class ConvolutionalNeuralNetwork:
    def __init__(self, layers, neurons):
        self.network = Sequential([
            Rescaling(1./255),
        ])
        self.network.add(Flatten(input_shape=[1000, 400]))

        for num in range(layers):
            self.network.add(Dense(neurons, activation=LeakyReLU(alpha=0.1)))

        self.network.add(Dense(5, activation="softmax"))


        self.network.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    def train(self, train, val, epochs):
        history = self.network.fit(train, epochs=epochs, validation_data=val)
        return history.history

    def evaluate(self, test):
        return self.network.evaluate(test)
