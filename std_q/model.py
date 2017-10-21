import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from std_q.hyperparameters import gamma

vector_size = 10*10 + 7 + 4

def create_nn():
    if os.path.exists('race-car.h5'):
        return load_model('race-car.h5')

    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    model.summary()

    return model


class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # one feedforward nn for all actions.
        self.prev_state = None
        self.prev_qvals = None
        self.prev_argmax = None

    def predict(self, state):
        return self.model.predict(state.reshape(-1, vector_size), verbose=0)[0]

    def update(self, state, expected_output):
        self.model.fit(state.reshape(-1, vector_size), np.array(expected_output).reshape(-1, 11), epochs=1, verbose=0)

    def get_action(self, state, eps, reward):
        argmax_qvals, qvals = self.sample_action(state, eps)
        action = Model.convert_argmax_qval_to_env_action(argmax_qvals)

        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma*np.max(qvals)
            y = self.prev_qvals[:]
            y[self.prev_argmax] = G
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = qvals
        self.prev_argmax = argmax_qvals

        return action

    def sample_action(self, state, eps, softmax=False):
        qval = self.predict(state)

        if softmax:
            prob = [math.exp(q)/eps for q in qval]
            prob = prob / np.sum(prob)

            action_selection_index = np.random.choice(range(len(qval)), p=prob)

            return action_selection_index, qval
        else:
            if np.random.random() < eps:
                return random.randint(0, 10), qval
            else:
                return np.argmax(qval), qval

    def convert_argmax_qval_to_env_action(output_value):
        # we reduce the action space to 15 values.  9 for steering, 6 for gas/brake.
        # to reduce the action space, gas and brake cannot be applied at the same time.
        # as well, steering input and gas/brake cannot be applied at the same time.
        # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

        gas = 0.0
        brake = 0.0
        steering = 0.0

        # output value ranges from 0 to 10

        if output_value <= 8:
            # steering. brake and gas are zero.
            output_value -= 4
            steering = float(output_value)/4
        elif output_value >= 9 and output_value <= 9:
            output_value -= 8
            gas = float(output_value)/3 # 33%
        elif output_value >= 10 and output_value <= 10:
            output_value -= 9
            brake = float(output_value)/2 # 50% brakes
        else:
            print("error")

        white = np.ones((round(brake * 100), 10))
        black = np.zeros((round(100 - brake * 100), 10))
        brake_display = np.concatenate((black, white))*255

        white = np.ones((round(gas * 100), 10))
        black = np.zeros((round(100 - gas * 100), 10))
        gas_display = np.concatenate((black, white))*255

        return [steering, gas, brake]
