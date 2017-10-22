import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random

from softmax.hyperparameters import gamma, alpha

from std_q.model import Model as StdQModel


def decrease_linearly(value, value_start, value_end, result_start, result_end):
    return result_start - (result_start-result_end)*(value-value_start)/(value_end - value_start)


class Model(StdQModel):
    def get_action(self, state, eps, reward):
        argmax_qvals, qvals = self.sample_action(state, eps, softmax=True)
        action = Model.convert_argmax_qval_to_env_action(argmax_qvals)
        change = 0

        if self.prev_state is not None and self.prev_qvals is not None and self.prev_argmax is not None:
            G = reward + gamma*np.max(qvals)
            y = self.prev_qvals[:]
            change = G - y[self.prev_argmax]
            y[self.prev_argmax] += alpha * change
            y[self.prev_argmax] = G
            self.update(self.prev_state, y)

        self.prev_state = state
        self.prev_qvals = qvals
        self.prev_argmax = argmax_qvals

        loss = change**2
        return action, loss

    def get_action_selection_parameter(cur_episode, total_episodes):
        if cur_episode < 500:
            return decrease_linearly(cur_episode, 0, 500, 100, 1)
        elif cur_episode < 900:
            return decrease_linearly(cur_episode, 500, 900, 1, 0.1)
        else:
            return 0.1


if __name__ == "__main__":
    for i in range(1001):
        print(i, Model.get_action_selection_parameter(i, 1001))
