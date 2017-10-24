import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random

from sarsa.hyperparameters import gamma, alpha
from std_q.model import Model as StdQModel

class Model(StdQModel):
    def get_action(self, state, eps, reward):
        argmax_qvals, qvals = self.sample_action(state, eps)

        action = Model.convert_argmax_qval_to_env_action(argmax_qvals)
        change = 0

        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma * qvals[argmax_qvals]
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] += alpha * change
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = qvals
        self.prev_argmax = argmax_qvals

        loss = change ** 2
        return action, loss

        return action
