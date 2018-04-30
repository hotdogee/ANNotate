'''Trains and evaluate a simple MLP on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import tensorflow as tf
import numpy as np
import keras
from datasets import pfam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

epochs = 3
num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 128
batch_size = 32
dense_size = 128
dense_dropout = 0.5

def checkdir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

csvlog_dir = './csvlogs'
checkdir(csvlog_dir)
model_dir = './models'
checkdir(model_dir)
modelplot_dir = './modelplots'
checkdir(modelplot_dir)

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ds{5}do{6:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    dense_size, dense_dropout * 100)

csvlog_path = '{0}.{1}.csv'.format(os.path.join(
    csvlog_dir, os.path.splitext(os.path.basename(__file__))[0]), model_name)


with tf.device('/gpu:1'):
    print('Loading data...')
    (x_train, y_train), (x_test, y_test), (aa_list, domain_list) = pfam.load_data(
        num_domain=num_domain, test_split=test_split)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    # print(domain_list)

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
    model.add(Dense(dense_size, input_shape=(maxlen,)))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    modelplot_path = '{0}.{1}.png'.format(os.path.join(
        modelplot_dir, os.path.splitext(os.path.basename(__file__))[0]), model_name)
    plot_model(model, to_file=modelplot_path)

    for epoch in range(epochs):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch + 1,
                            verbose=1,
                            validation_split=validation_split,
                            callbacks=[
                                CSVLogger(csvlog_path, append=True), TensorBoard()],
                            initial_epoch=epoch)
        score = model.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # save model
        model_path = '{0}.e{1}.{2}.h5'.format(os.path.join(
            model_dir, os.path.splitext(os.path.basename(__file__))[0]), epoch, model_name)
        model.save(model_path)
