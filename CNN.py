import keras
from keras.models import Sequential, Input, Model
from keras.layers import Softmax, ReLU, LeakyReLU, Dense, Flatten


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(Flatten(input_shape=[1000, 400]))
        self.network.add(LeakyReLU(alpha=0.1))  # is it enough?
        self.network.add(Dense(4, activation="softmax"))

        self.network.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs):
        history = self.network.fit(X_train / 255, y_train, epochs=epochs, validation_data=(X_val / 255, y_val))
        return history.history

    def evaluate(self, X_test, y_test):
        return self.network.evaluate(X_test / 255, y_test)
