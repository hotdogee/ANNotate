'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

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

num_domain = 100
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 32
epochs = 10
embedding_dims = 128
embedding_dropout = 0.1
recurrent_dropout = 0.1
filters = 128
kernel_size = 5
lstm_output_size = 64
dense_size = 32
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ed{6}eo{7:.0f}ls{12}ds{10}do{11:.0f}'.format(
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
        model.save(model_path)
    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

# 3584145/3584145 [==============================] 100.00% - 64326s - loss: 0.7251 - acc: 0.8179 - val_loss: 0.2827 - val_acc: 0.9252
# 995596/995596 [==============================] 100.00% - 3916s
# Test score: 0.284647500905
# Test accuracy: 0.924580854082
# Train on 3584145 samples, validate on 398239 samples
# Epoch 2/2
# 3584145/3584145 [==============================] 100.00% - 62876s - loss: 0.4380 - acc: 0.8863 - val_loss: 0.2265 - val_acc: 0.9390
# 995596/995596 [==============================] 100.00% - 3646s
# Test score: 0.226647564766
# Test accuracy: 0.939269543068
# Train on 3584145 samples, validate on 398239 samples
# Epoch 3/3
# 3584145/3584145 [==============================] 100.00% - 62293s - loss: 0.3832 - acc: 0.9000 - val_loss: 0.2065 - val_acc: 0.9443
# 995596/995596 [==============================] 100.00% - 3553s
# Test score: 0.205963518859
# Test accuracy: 0.944450359383

# 3584145/3584145 [==============================] 100.00% - 59371s - loss: 0.3543 - acc: 0.9072 - val_loss: 0.1958 - val_acc: 0.9466
# 995596/995596 [==============================] 100.00% - 3756s
# Test score: 0.194667454505
# Test accuracy: 0.947274798211
# Train on 3584145 samples, validate on 398239 samples
# Epoch 5/5
# 3584145/3584145 [==============================] 100.00% - 64517s - loss: 0.3357 - acc: 0.9121 - val_loss: 0.1802 - val_acc: 0.9509
# 995596/995596 [==============================] 100.00% - 3584s
# Test score: 0.180611414908
# Test accuracy: 0.950632585908
# Train on 3584145 samples, validate on 398239 samples
# Epoch 6/6
# 3584145/3584145 [==============================] 100.00% - 62338s - loss: 0.3238 - acc: 0.9149 - val_loss: 0.1825 - val_acc: 0.9507
# 995596/995596 [==============================] 100.00% - 3595s
# Test score: 0.183765810755
# Test accuracy: 0.950693855741
# Train on 3584145 samples, validate on 398239 samples
# Epoch 7/7
# 3584145/3584145 [==============================] 100.00% - 62363s - loss: 0.3168 - acc: 0.9168 - val_loss: 0.1703 - val_acc: 0.9537
# 995596/995596 [==============================] 100.00% - 3597s
# Test score: 0.17137971566
# Test accuracy: 0.953280246204
# Train on 3584145 samples, validate on 398239 samples
# Epoch 8/8
# 3584145/3584145 [==============================] 100.00% - 62329s - loss: 0.3103 - acc: 0.9187 - val_loss: 0.1666 - val_acc: 0.9552
# 995596/995596 [==============================] 100.00% - 3616s
# Test score: 0.167223894257
# Test accuracy: 0.954982744005
# Train on 3584145 samples, validate on 398239 samples
# Epoch 9/9
# 3075616/3584145 [========================>.....]  85.81% - ETA: 8638s - loss: 0.3060 - acc: 0.9195

# 3,0.907158332043,0.354256769187,0.946647616126,0.195791251799

# 4,0.912065778589,0.335732181008,0.950916409488,0.180163822234

# 5,0.91494456837,0.323828910322,0.950738124589,0.182485260202

# 6,0.916821445561,0.316810491472,0.95374385733,0.170330841548

# 7,0.918739894731,0.310250130799,0.95520278024,0.166594218057