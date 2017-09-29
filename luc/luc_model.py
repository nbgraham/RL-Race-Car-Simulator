import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random

vector_size = 10*10 + 7 + 4
n_outputs = 11

def create_nn():
    if os.path.exists('race-car.h5'):
        return load_model('race-car.h5')

    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(n_outputs, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    model.summary()

    return model


class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # one feedforward nn for all actions.
        self.min_q = 0
        self.max_pi = 0

    def predict(self, state):
        return self.model.predict(state.reshape(-1, vector_size), verbose=0)[0]

    def update(self, state, expected_output):
        expect = (expected_output * self.max_pi) + self.min_q
        self.model.fit(state.reshape(-1, vector_size), np.array(expect).reshape(-1, 11), epochs=1, verbose=0)

    def sample_action(self, state, eps):
        qval = self.predict(state)
        self.min_q = np.min(qval)
        pi = qval - self.min_q
        self.max_pi = np.max(pi)
        pi /= self.max_pi

        if np.random.random() < eps:
            return random.randint(0, 10), pi
        else:
            return np.argmax(pi), pi
