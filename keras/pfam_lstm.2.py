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
from keras.utils import plot_model

num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 3
embedding_dims = 128
embedding_dropout = 0.1
recurrent_dropout = 0.1
filters = 128
kernel_size = 5
lstm_output_size = 64
dense_size = 32
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ep{5}ed{6}eo{7:.0f}ls{12}ds{10}do{11:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    epochs, embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100,
    lstm_output_size)

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
model.add(LSTM(lstm_output_size, dropout=embedding_dropout, recurrent_dropout=recurrent_dropout))
model.add(Dense(dense_size))
model.add(Activation('relu'))
model.add(Dropout(dense_dropout))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

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

# 780M
# 1071643/1071643 [==============================] 100.00% - 33032s - loss: 0.0915 - acc: 0.9725 - val_loss: 0.0167 - val_acc: 0.9948
# 297679/297679 [==============================] 100.00% - 2174s
# Test score: 0.0170753354441
# Test accuracy: 0.99463180137
# Train on 1071643 samples, validate on 119072 samples
# Epoch 2/2
# 1071643/1071643 [==============================] 100.00% - 39215s - loss: 0.0222 - acc: 0.9936 - val_loss: 0.0090 - val_acc: 0.9972
# 297679/297679 [==============================] 100.00% - 4358s
# Test score: 0.00908686943684
# Test accuracy: 0.997205043016
# Train on 1071643 samples, validate on 119072 samples
# Epoch 3/3
# 1071643/1071643 [==============================] 100.00% - 45244s - loss: 0.0166 - acc: 0.9954 - val_loss: 0.0073 - val_acc: 0.9978
# 297679/297679 [==============================] 100.00% - 4396s
# Test score: 0.0065186630298
# Test accuracy: 0.997970968728