'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import numpy as np
import keras
from datasets import pfam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard

num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 3
embedding_dims = 10
embedding_dropout = 0.2
filters = 128
kernel_size = 5
dense_size = 128
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ep{5}ed{6}eo{7:.0f}cf{8}ks{9}ds{10}do{11:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    epochs, embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100,
    )

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
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D())
#model.add(LSTM(lstm_output_size))
model.add(Dense(dense_size, activation='relu'))
model.add(Dropout(dense_dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for epoch in range(epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epoch+1,
                        verbose=1,
                        validation_split=validation_split,
                        callbacks=[CSVLogger('training.log', append=True), TensorBoard()],
                        initial_epoch=epoch)
    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # save model
    model_dir = './models'
    model_path = '{0}.e{1}.{2}.h5'.format(os.path.abspath(os.path.join(model_dir, os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
    if not os.path.exists(os.path.dirname(model_path)):
        try:
            os.makedirs(os.path.dirname(model_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    model.save(model_path)