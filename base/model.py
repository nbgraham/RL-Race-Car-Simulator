import numpy as np
from os import path
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam

from action_selection import softmax_select, eps_select
from base.hyperparameters import alpha, gamma

class BaseModel:
    def __init__(self, env, name, input_size, action_set):
        self.input_size = input_size
        self.action_set = action_set
        self.env = env
        self.model = self.create_nn(name, input_size)  # one feedforward nn for all actions.
        self.prev_state = None
        self.prev_qvals = None
        self.prev_argmax = None

    def predict(self, state):
        return self.model.predict(state.reshape(-1, self.input_size), verbose=0)[0]

    def update(self, state, expected_output):
        self.model.fit(state.reshape(-1, self.input_size), np.array(expected_output).reshape(-1, 11), epochs=1, verbose=0)

    def get_action(self, state, eps, reward):
        max_index, network_output = self.sample_action(state, eps)

        action = self.action_set[max_index]
        change = 0

        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma * np.max(network_output)
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] = (1 - alpha) * y[self.prev_argmax] + alpha * change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = network_output
        self.prev_argmax = max_index

        loss = change ** 2
        return action, loss

    def get_action_selection_parameter(cur_episode, total_episodes):
        return 5 / np.sqrt(cur_episode + 25) - 0.1

    def sample_action(self, state, action_selection_parameter, softmax=False):
        network_output = self.predict(state)

        if softmax:
            return softmax_select(network_output, action_selection_parameter), network_output
        else:
            return eps_select(network_output, action_selection_parameter), network_output

    def create_nn(self, name, input_size):
        model_filename = "race_car_" + name + "h5"
        if path.exists(model_filename):
            print("Loading existing model")
            return load_model(model_filename)

        model = Sequential()
        model.add(Dense(512, init='lecun_uniform', input_shape=(input_size,)))  # 7x7 + 3.  or 14x14 + 3
        model.add(Activation('relu'))

        model.add(Dense(len(self.action_set), init='lecun_uniform'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        adamax = Adamax()  # Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        model.summary()

        return model