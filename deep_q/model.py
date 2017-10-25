import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation#, Dropout, Flatten, Convolution2D, MaxPooling2D, Merge, Embedding
from keras.optimizers import Adamax#,SGD, RMSprop, Adam


from base.model import BaseModel


class Model(BaseModel):
    def create_nn(self, name, input_size):
        model_filename = "models/race_car_" + name + "h5"
        if os.path.exists(model_filename):
            print("Loading existing model")
            return load_model(model_filename)

        model = Sequential()
        model.add(Dense(128, init='lecun_uniform', input_shape=(input_size,)))  # 7x7 + 3.  or 14x14 + 3
        model.add(Activation('relu'))

        model.add(Dense(64, init='lecun_uniform'))  # 7x7 + 3.  or 14x14 + 3
        model.add(Activation('relu'))

        model.add(Dense(32, init='lecun_uniform'))  # 7x7 + 3.  or 14x14 + 3
        model.add(Activation('relu'))

        model.add(Dense(11, init='lecun_uniform'))
        model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        adamax = Adamax()  # Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        model.summary()

        return model