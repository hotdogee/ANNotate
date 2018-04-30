'''Trains and evaluate a simple MLP on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import tensorflow as tf
import numpy as np
import keras
from datasets import pfam
from keras.models import load_model
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix

num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 1
embedding_dims = 10
embedding_dropout = 0.2
filters = 128
kernel_size = 5
dense_size = 128
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ed{6}eo{7:.0f}cf{8}ks{9}ds{10}do{11:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    epochs, embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100)
csvlog_dir = './csvlogs'
if not os.path.exists(csvlog_dir):
    try:
        os.makedirs(csvlog_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
csvlog_path = '{0}.{1}.csv'.format(os.path.abspath(os.path.join(csvlog_dir, 
    os.path.splitext(os.path.basename(__file__))[0])), model_name)
model_dir = './models'
if not os.path.exists(model_dir):
    try:
        os.makedirs(model_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
confusion_matrix_dir = './confusion_matrix'
if not os.path.exists(confusion_matrix_dir):
    try:
        os.makedirs(confusion_matrix_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
classification_report_dir = './classification_report'
if not os.path.exists(classification_report_dir):
    try:
        os.makedirs(classification_report_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

with tf.device('/gpu:0'):
    print('Loading data...')
    (x_train, y_train_class), (x_test, y_test_class), domain_list = pfam.load_data(
        num_domain=num_domain, test_split=test_split)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print(domain_list)
    num_classes = np.max(y_train_class) + 1
    print(num_classes, 'classes')

    print('Pad sequences')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
        '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train_class, num_classes)
    y_test = keras.utils.to_categorical(y_test_class, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Embedding(len(aa_list) + 1,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(embedding_dropout))
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

    save_freq = 1
    for epoch in range(0, epochs):
        model_path = '{0}.e{1}.{2}.h5'.format(os.path.abspath(os.path.join(model_dir, 
            os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
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
        y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        print(classification_report(y_test_class, y_pred_class, target_names=domain_list, digits=5))

        confusion_matrix_path = '{0}.e{1}.{2}.cm.csv'.format(os.path.abspath(os.path.join(confusion_matrix_dir, 
            os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
        if not os.path.exists(confusion_matrix_path):
            np.savetxt(confusion_matrix_path, confusion_matrix(y_test_class, y_pred_class), fmt='%d', delimiter=",")

        classification_report_path = '{0}.e{1}.{2}.cr.txt'.format(os.path.abspath(os.path.join(
            classification_report_dir, os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
        if not os.path.exists(classification_report_path):
            np.savetxt(classification_report_path, [classification_report(y_test_class, y_pred_class, 
                target_names=domain_list, digits=5)], fmt='%s')
