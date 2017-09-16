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
filters = 256
kernel_size = 5
lstm_output_size = 64
dense_size = 64
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

# epoch,acc,loss,val_acc,val_loss
# 0,0.863968394136,0.530729407136,0.945530196689,0.201210024962
# 1,0.921428681039,0.293829488139,0.955883276119,0.159324902162
# Epoch 3/3
# 3584145/3584145 [==============================] 100.00% - 93996s - loss: 0.2565 - acc: 0.9308 - val_loss: 0.1422 - val_acc: 0.9603
# 995596/995596 [==============================] 100.00% - 5638s
# Test score: 0.144072181394
# Test accuracy: 0.960154520508
# Train on 3584145 samples, validate on 398239 samples
# Epoch 4/4
# 3584145/3584145 [==============================] 100.00% - 94632s - loss: 0.2403 - acc: 0.9349 - val_loss: 0.1373 - val_acc: 0.9616
# 995596/995596 [==============================] 100.00% - 5463s
# Test score: 0.138583087005
# Test accuracy: 0.961369872921
# Train on 3584145 samples, validate on 398239 samples
# Epoch 5/5
# 3584145/3584145 [==============================] 100.00% - 94513s - loss: 0.2309 - acc: 0.9374 - val_loss: 0.1339 - val_acc: 0.9623
# 995596/995596 [==============================] 100.00% - 5529s
# Test score: 0.135844261564
# Test accuracy: 0.961896190824
# Train on 3584145 samples, validate on 398239 samples
# Epoch 6/6
# 3584145/3584145 [==============================] 100.00% - 94843s - loss: 0.2245 - acc: 0.9390 - val_loss: 0.1249 - val_acc: 0.9650
# 995596/995596 [==============================] 100.00% - 5476s
# Test score: 0.126067731344
# Test accuracy: 0.96474574024
# Train on 3584145 samples, validate on 398239 samples
# Epoch 7/7
# 3584145/3584145 [==============================] 100.00% - 94258s - loss: 0.2196 - acc: 0.9402 - val_loss: 0.1198 - val_acc: 0.9662
# 995596/995596 [==============================] 100.00% - 5570s
# Test score: 0.122114796609
# Test accuracy: 0.965831522023
# Train on 3584145 samples, validate on 398239 samples
# Epoch 8/8
# 3584145/3584145 [==============================] 100.00% - 93858s - loss: 0.2156 - acc: 0.9413 - val_loss: 0.1194 - val_acc: 0.9663
# 995596/995596 [==============================] 100.00% - 5523s
# Test score: 0.121368168067
# Test accuracy: 0.965762216803
# Train on 3584145 samples, validate on 398239 samples
# Epoch 9/9
#  948832/3584145 [======>.......................]  26.47% - ETA: 67575s - loss: 0.2138 - acc: 0.9415