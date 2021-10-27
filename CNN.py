from tensorflow.keras.layers import LeakyReLU, Dense, Flatten
from keras.layers import Rescaling
from tensorflow.keras.models import Sequential
from tensorflow import keras


class ConvolutionalNeuralNetwork:
    def __init__(self, neurons1):
        self.network = Sequential([
            Rescaling(1. / 255)
        ])
        self.network.add(Flatten(input_shape=[100, 40]))
        self.network.add(Dense(neurons1, activation=LeakyReLU(alpha=0.1)))

        self.network.add(Dense(5, activation="softmax"))

        self.network.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    def train(self, train, val, epochs):
        history = self.network.fit(train, epochs=epochs, validation_data=val, verbose=1, batch_size=64)
        return history.history

    def evaluate(self, test):
        return self.network.evaluate(test)

    def predict(self, test):
        return self.network.predict(test)
