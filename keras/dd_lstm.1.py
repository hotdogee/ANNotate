'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import tensorflow as tf
import numpy as np
import keras
from datasets import dd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils import plot_model

validation_split = 0.00
maxlen = 256
batch_size = 64
epochs = 5000
embedding_dims = 10
embedding_dropout = 0.6
filters = 64
kernel_size = 17
lstm_output_size = 64
dense_size = 64
dense_dropout = 0.6

model_name = 'vs{:.0f}ml{}bs{}ed{}eo{:.0f}cf{}ks{}ls{}ds{}do{:.0f}'.format(
    validation_split * 100, maxlen, batch_size,
    embedding_dims, embedding_dropout * 100, filters, kernel_size, lstm_output_size,
    dense_size, dense_dropout * 100)
csvlog_dir = './csvlogs'
if not os.path.exists(csvlog_dir):
    try:
        os.makedirs(csvlog_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
model_dir = './models'
if not os.path.exists(model_dir):
    try:
        os.makedirs(model_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
csvlog_path = '{0}.{1}.csv'.format(os.path.abspath(os.path.join(csvlog_dir, os.path.splitext(os.path.basename(__file__))[0])), model_name)


with tf.device('/gpu:1'):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test), (aa_list, fold_list) = dd.load_data()
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    # print(fold_list)

    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')

    print('Pad sequences')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
        '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Embedding(len(aa_list) + 1,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(embedding_dropout))
    model.add(LSTM(lstm_output_size, return_sequences=True))
    model.add(Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_size))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    save_freq = 10
    for epoch in range(2440, epochs):
        model_path = '{0}.e{1}.{2}.h5'.format(os.path.abspath(os.path.join(model_dir, os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
        if os.path.exists(model_path):
            print('Loading model:', os.path.basename(model_path))
            model = load_model(model_path)
        else:
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epoch+1,
                                verbose=1,
                                validation_split=validation_split,
                                callbacks=[CSVLogger(csvlog_path, append=True), TensorBoard()],
                                initial_epoch=epoch)

            # save model
            if epoch % save_freq == 0:
                model.save(model_path)
        score = model.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])