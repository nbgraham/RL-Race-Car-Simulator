import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np

from base.model import BaseModel
from convolutional.hyperparameters import alpha, gamma


class Model(BaseModel):
    def create_nn(self, name, input_shape):
        model_filename = "race_car_" + name + "h5"
        if os.path.exists(model_filename):
            print("Loading existing model")
            return load_model(model_filename)

        print("Input shape", input_shape)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(16, kernel_size=(4, 4), strides=(2, 2),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(len(self.action_set), activation='softmax'))

        adamax = Adamax()  # Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        model.summary()

        return model

    def get_action(self, state, eps, reward):
        max_index, network_output = self.sample_action(state, eps)

        action = self.action_set[max_index]
        change = 0

        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma * np.max(network_output)
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] += alpha * change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = network_output
        self.prev_argmax = max_index

        loss = change ** 2
        return action, loss

    def predict(self, state):
        return self.model.predict(state.reshape((-1, 90,90,1)), verbose=0)[0]

    def update(self, state, expected_output):
        self.model.fit(state.reshape(-1, 90, 90, 1), np.array(expected_output).reshape(-1, 11), epochs=1, verbose=0)