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
embedding_dims = 10
embedding_dropout = 0.2
lstm_output_size = 64
filters = 128
kernel_size = 5
dense_size = 128
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
    for epoch in range(epochs):
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
# 0,0.766851508519,0.876756168522,0.940628617484,0.223782244376
# 1,0.936576226686,0.235743276728,0.966987160962,0.116938711003
# 2,0.958091818271,0.150329911102,0.976052069243,0.0820172337648
# 3,0.967442723439,0.114220386059,0.98047403693,0.0662938186803
# 4,0.973060241703,0.0932337359463,0.982977558684,0.0571333450123
# 5,0.976562332159,0.0802049192792,0.985332928195,0.049058511723
# 6,0.979003360633,0.0713474643967,0.986196731065,0.0456574681537
# 7,0.980784817578,0.0647178414202,0.986633654621,0.0433599621574
# 8,0.982266342461,0.0594861238031,0.987803806213,0.0396407213937
# 9,0.983249282604,0.0557349355966,0.988662587018,0.037312718564
# 10,0.98426123943,0.0523627971475,0.9889086704,0.0368611294336
# 11,0.984919694934,0.0498437815314,0.989415903497,0.0343146797598
# 12,0.985572012293,0.047566011732,0.989478679888,0.0341905261269
# 13,0.986102124775,0.0456620524429,0.989596699477,0.0337110137572
# 14,0.986653720762,0.0438355472871,0.990397725945,0.031586599189
# 15,0.987021172413,0.0425464438607,0.990842182666,0.0301198589252
# 16,0.987340913942,0.0411287953866,0.990925047482,0.0297003893045
# 17,0.987777280216,0.0399621831994,0.990726674165,0.0304550389921
# 18,0.988082234396,0.038850400635,0.991033022865,0.0295118157145
# 19,0.988414252214,0.0376939802872,0.991793872506,0.0274034874498
# 20,0.988568542846,0.0370735649879,0.991703474525,0.0270497571344
# 21,0.988759104333,0.0363187818465,0.991818983055,0.0267879379838
# 22,0.989025834613,0.035667687397,0.991507612241,0.027689465604
# 23,0.9892163961,0.0349266917459,0.991979690566,0.0262144032372
# 24,0.989357294417,0.034316508712,0.99196462424,0.0263395784672
# 25,0.989621513638,0.0335152421647,0.992135375993,0.0260071238038
