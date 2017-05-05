'''Trains and evaluate a simple MLP on the Pfam Domain classification task.
'''
from __future__ import print_function

import numpy as np
import keras
from datasets import pfam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing import sequence

num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 1
embedding_dims = 10
filters = 128
kernel_size = 5

print('Loading data...')
(x_train, y_train), (x_test, y_test), (aa_list, domain_list) = pfam.load_data(
    num_domain=num_domain, test_split=test_split)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(domain_list)
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
model.add(Dropout(0.2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
