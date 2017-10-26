from os import path
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from base.model import BaseModel
from pg.hyperparameters import gamma, action_selection_coeff, alpha
from action_selection import softmax_select, eps_select


class Model(BaseModel):
    def __init__(self, env, name, input_size, action_set):
        BaseModel.__init__(self, env, name, input_size, action_set, alpha=alpha, gamma=gamma)
        self.last_ten_rewards = [0]*10

    def sample_action(self, state, action_selection_parameter, softmax=False):
        probs = self.predict(state)

        probs = probs / np.sum(probs)

        if softmax:
            return softmax_select(probs, action_selection_parameter), probs
        else:
            return eps_select(probs, action_selection_parameter), probs

    def get_action(self, state, eps, reward):
        self.last_ten_rewards = self.last_ten_rewards[1:].append(reward)

        arg_max_probs, probs = self.sample_action(state, eps)

        action = self.action_set[arg_max_probs]
        change = 0
        
        if self.prev_state is not None and self.prev_net_output is not None and self.prev_action_index is not None:
            other_prob = 0
            choice_prob = 0
            if np.sum(self.last_ten_rewards) < 0:
                print("Discourage")
                uniform_prob = 1/(len(self.action_set) - 1)
                other_prob = uniform_prob
            else:
                print("Encourage")
                choice_prob = 1
            G_array = [other_prob]*len(self.action_set)
            G_array[self.prev_action_index] = choice_prob
            G = np.array(G_array)

            y = self.prev_net_output[:]
            change = G - y[self.prev_action_index]
            y += alpha*change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_net_output = probs
        self.prev_action_index = arg_max_probs

        loss = np.sum(change**2)
        return action, loss

    def create_nn(self, name, input_size):
        model_filename = "race_car_" + name + "h5"
        if path.exists(model_filename):
            print("Loading existing model")
            return load_model(model_filename)

        model = Sequential()
        model.add(Dense(512, init='lecun_uniform', input_shape=(input_size,)))  # 7x7 + 3.  or 14x14 + 3
        model.add(Activation('relu'))

        model.add(Dense(len(self.action_set), init='lecun_uniform'))
        model.add(Activation('softmax'))  # linear output so we can have range of real-valued outputs

        adamax = Adamax()  # Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        model.summary()

        return model
