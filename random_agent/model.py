import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random_agent
import math

from std_q.hyperparameters import gamma, action_selection_coeff, alpha

vector_size = 10*10 + 7 + 4


class Model:
    def __init__(self, env, name):
        self.env = env

    def predict(self, state):
        return np.zeros(11)

    def update(self, state, expected_output):
        pass

    def get_action(self, state, eps, reward):
        action = self.env.action_space.sample()
        return action, 0

    def get_action_selection_parameter(cur_episode, total_episodes):
        return 0

    def sample_action(self, state, action_selection_parameter, softmax=False):
        qval = self.predict(state)

        return 0, qval

    def convert_argmax_qval_to_env_action(self, output_value):
        # we reduce the action space to 15 values.  9 for steering, 6 for gas/brake.
        # to reduce the action space, gas and brake cannot be applied at the same time.
        # as well, steering input and gas/brake cannot be applied at the same time.
        # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

        action = self.env.action_space.sample()
        return action