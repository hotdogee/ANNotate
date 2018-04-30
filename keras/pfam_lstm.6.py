'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import tensorflow as tf
import numpy as np
import keras
from datasets import pfam
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils import plot_model

num_domain = 100
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 256
epochs = 30
embedding_dims = 64
embedding_dropout = 0.2
filters = 256
kernel_size = 7
lstm_output_size = 64
dense_size = 256
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ed{6}eo{7:.0f}cf{8}ks{9}ls{12}ds{10}do{11:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    epochs, embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100,
    lstm_output_size)
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


with tf.device('/gpu:0'):
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

    save_freq = 1
    for epoch in range(11, epochs):
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

# epoch,acc,loss,val_acc,val_loss
# 0,0.901748952679,0.362492145061,0.983369283299,0.0532865253791
# 1,0.982909173596,0.0558807544744,0.990073799882,0.0311253481906
# 2,0.9883754703,0.0369531229886,0.992321194065,0.0241618901942
# 3,0.990641840662,0.0294281466501,0.99336328183,0.0207290719593
# 4,0.991861378376,0.0253997094171,0.993835360173,0.019357691605
# 5,0.99279465535,0.0224547707899,0.993935802326,0.019214584181
# 6,0.993400936625,0.0205504315585,0.994327526894,0.0177129472048
# 7,0.993771457349,0.0192689244748,0.994538455529,0.0174068771963
# 8,0.994108218278,0.0181030360444,0.994977890161,0.0166444657384
# 9,0.994412614445,0.0172047354529,0.994832248954,0.0168603560187
# 10,0.99469357406,0.0164078956573,0.994857359525,0.0169307597939
# 11,0.994831682312,0.0157850642919,0.995234017763,0.0163920411403
# 12,0.995027544924,0.0151698139753,0.994882470075,0.0170410652592
