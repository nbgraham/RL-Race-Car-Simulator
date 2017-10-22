import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils
import numpy as np
import random
import math

from std_q.hyperparameters import gamma, action_selection_coeff, alpha
from std_q.model import Model as StdQModel

vector_size = 10*10 + 7 + 4

def create_nn(name):
    model_filename = "race_car_" + name + "h5"
    if os.path.exists(model_filename):
        print("Loading existing model")
        return load_model(model_filename)

    model = Sequential()
    model.add(Dense(128, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(64, init='lecun_uniform', input_shape=(vector_size,)))  # 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(32, init='lecun_uniform', input_shape=(vector_size,)))  # 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

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
        self.chose_max = []
        self.count = 0