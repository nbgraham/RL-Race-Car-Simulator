import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from std_q.model import Model as StdQModel

vector_size = 10*10 + 7 + 4
num_actions = 11

def create_nn(name):
    model_filename = "race_car_" + name + "h5"
    if os.path.exists(model_filename):
        print("Loading existing model")
        return load_model(model_filename)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu',
                     input_shape=(vector_size,)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=(1, 1), stride=(1,1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    return model


class Model(StdQModel):
    def __init__(self, env, name):
        self.env = env
        self.model = create_nn(name)  # one feedforward nn for all actions.
        self.prev_state = None
        self.prev_qvals = None
        self.prev_argmax = None