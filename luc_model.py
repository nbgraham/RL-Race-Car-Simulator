import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam
from keras.utils import np_utils


def create_nn():
    if os.path.exists('race-car.h5'):
        return load_model('race-car.h5')

    model = Sequential()
    model.add(Dense(512, init='lecun_uniform', input_shape=(vector_size,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

    model.add(Dense(11, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    model.summary()

    return model


class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn()  # one feedforward nn for all actions.

    def predict(self, s):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def update(self, s, G):
        self.model.fit(s.reshape(-1, vector_size), np.array(G).reshape(-1, 11), nb_epoch=1, verbose=0)

    def sample_action(self, s, eps):
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0, 10), qval
        else:
            return np.argmax(qval), qval
