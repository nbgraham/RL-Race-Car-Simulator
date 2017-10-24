import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from std_q.model import Model as StdQModel
from pg.hyperparameters import gamma, action_selection_coeff, alpha
from action_selection import softmax_select, eps_select

vector_size = 10*10 + 7 + 4

def create_nn(name):
    model_filename = "race_car_" + name + "h5"
    if os.path.exists(model_filename):
        print("Loading existing model")
        return load_model(model_filename)

    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('sigmoid'))

    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    model.summary()

    return model

class Model(StdQModel):
    def __init__(self, env, name):
        self.env = env
        self.model = create_nn(name)  # one feedforward nn for all actions.
        self.prev_state = None
        self.prev_qvals = None
        self.prev_argmax = None


    def update(self, state, expected_output):
        self.model.fit(state.reshape(-1, vector_size), np.array(expected_output).reshape(-1, 11), epochs=1, verbose=0)

    def sample_action(self, state, action_selection_parameter, softmax=False):
        probs = self.predict(state)

        probs = probs / np.sum(probs)

        if softmax:
            return softmax_select(probs, action_selection_parameter), probs
        else:
            return eps_select(probs, action_selection_parameter), probs

    def get_action(self, state, eps, reward):
        arg_max_probs, probs = self.sample_action(state, eps)

        action = Model.convert_argmax_qval_to_env_action(arg_max_probs)
        change = 0
        
        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma*np.max(probs)
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] += alpha*change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = probs
        self.prev_argmax = arg_max_probs

        loss = change**2
        return action, loss
