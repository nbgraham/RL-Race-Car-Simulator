from os import path
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from base.model import BaseModel
from apprentice.hyperparameters import gamma, encourage_strength, alpha


class Model(BaseModel):
    def __init__(self, env, name, input_size, action_set):
        BaseModel.__init__(self, env, name, input_size, action_set, alpha=alpha, gamma=gamma)

    def encourage_action(self, state, action_index):
        probs = self.predict(state)

        change = 0
        if self.prev_state is not None and self.prev_net_output is not None and self.prev_action_index is not None:
            other_prob = 0
            choice_prob = 1

            G_array = [other_prob]*len(self.action_set)
            G_array[self.prev_action_index] = choice_prob
            G = np.array(G_array)

            y = self.prev_net_output[:]
            change = G - y[self.prev_action_index]
            y += alpha*change
            for i in range (encourage_strength):
                self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_net_output = probs
        self.prev_action_index = action_index

        loss = np.sum(change**2)
        return loss

    # softmax output layer activation for prob
    def create_nn(self, name, input_size):
        model_filename = "models/race_car_" + name + ".h5"
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
