import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

with tf.device('/cpu:0'):
    inputs1 = Input(shape=(3, 1))
    lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)
    model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
    # define input data
    data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
    # make and show prediction
    print(model.predict(data))