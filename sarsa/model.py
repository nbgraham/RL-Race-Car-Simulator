import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random

from sarsa.hyperparameters import gamma, alpha
from base.model import BaseModel

class Model(BaseModel):
    def __init__(self, env, name, input_size, action_set):
        BaseModel.__init__(self, env, name, input_size, action_set, alpha=alpha, gamma=gamma)
    
    def get_action(self, state, eps, reward):
        argmax_qvals, qvals = self.sample_action(state, eps)

        action = self.action_set[argmax_qvals]
        change = 0

        if self.prev_state is not None and self.prev_net_output is not None and self.prev_action_index is not None:
            G = reward + self.gamma * qvals[argmax_qvals]
            y = self.prev_net_output[:]
            change = G - y[self.prev_action_index]
            y[self.prev_action_index] += alpha * change
            # self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_net_output = qvals
        self.prev_action_index = argmax_qvals

        loss = change ** 2
        return action, loss

        return action
    
    def get_action_selection_parameter(cur_episode, total_episodes):
        return 0
