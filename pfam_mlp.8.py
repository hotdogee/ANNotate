'''Trains and evaluate a simple MLP on the Pfam Domain classification task.
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
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard

num_domain = 100
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 3
embedding_dims = 64
embedding_dropout = 0.2
filters = 256
kernel_size = 7
dense_size = 256
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ep{5}ed{6}eo{7:.0f}cf{8}ks{9}ds{10}do{11:.0f}'.format(
    num_domain, test_split * 100, validation_split * 100, maxlen, batch_size,
    epochs, embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100)

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
model.add(Dropout(0.2))
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

# Epoch 1/16
# 3584145/3584145 [==============================] - 2254s - loss: 0.1980 - acc: 0.9440 - val_loss: 0.0834 - val_acc: 0.9753
# Epoch 2/16
# 3584145/3584145 [==============================] - 2256s - loss: 0.1352 - acc: 0.9624 - val_loss: 0.0972 - val_acc: 0.9738
# Epoch 3/16
# 3584145/3584145 [==============================] - 2274s - loss: 0.1424 - acc: 0.9624 - val_loss: 0.0914 - val_acc: 0.9755
# Epoch 4/16
# 3584145/3584145 [==============================] - 2299s - loss: 0.1536 - acc: 0.9616 - val_loss: 0.0956 - val_acc: 0.9752
# Epoch 5/16
# 3584145/3584145 [==============================] - 2279s - loss: 0.1663 - acc: 0.9601 - val_loss: 0.1131 - val_acc: 0.9715
# Epoch 6/16
# 3584145/3584145 [==============================] - 2229s - loss: 0.1805 - acc: 0.9586 - val_loss: 0.1285 - val_acc: 0.9726
# Epoch 7/16
# 3584145/3584145 [==============================] - 2302s - loss: 0.1974 - acc: 0.9564 - val_loss: 0.1256 - val_acc: 0.9716
# Epoch 8/16
# 3584145/3584145 [==============================] - 2195s - loss: 0.2148 - acc: 0.9545 - val_loss: 0.1489 - val_acc: 0.9683
# Epoch 9/16
# 3584145/3584145 [==============================] - 2425s - loss: 0.2365 - acc: 0.9518 - val_loss: 0.1654 - val_acc: 0.9655
# Epoch 10/16
# 3584145/3584145 [==============================] - 2193s - loss: 0.2625 - acc: 0.9490 - val_loss: 0.1817 - val_acc: 0.9688
# Epoch 11/16
# 3584145/3584145 [==============================] - 2195s - loss: 0.2936 - acc: 0.9450 - val_loss: 0.1769 - val_acc: 0.9677
# Epoch 12/16
# 3584145/3584145 [==============================] - 2191s - loss: 0.3325 - acc: 0.9402 - val_loss: 0.2349 - val_acc: 0.9637
# Epoch 13/16
# 3584145/3584145 [==============================] - 2232s - loss: 0.3801 - acc: 0.9351 - val_loss: 0.2543 - val_acc: 0.9596
# Epoch 14/16
# 3584145/3584145 [==============================] - 2204s - loss: 0.4340 - acc: 0.9293 - val_loss: 0.3555 - val_acc: 0.9556
# Epoch 15/16
# 3584145/3584145 [==============================] - 2303s - loss: 0.5033 - acc: 0.9233 - val_loss: 0.3712 - val_acc: 0.9501
# Epoch 16/16
# 3584145/3584145 [==============================] - 2352s - loss: 0.5703 - acc: 0.9179 - val_loss: 0.3745 - val_acc: 0.9483
# 995596/995596 [==============================] - 162s
# Test score: 0.379666776267
# Test accuracy: 0.947849328443
