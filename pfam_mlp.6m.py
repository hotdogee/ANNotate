'''Trains and evaluate a simple MLP on the Pfam Domain classification task.
'''
from __future__ import print_function

from datasets import pfam
from keras.models import Sequential
from keras.layers import Embedding, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Activation

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

model_name = 'ed{0}eo{1:.0f}cf{2}ks{3}ds{4}do{5:.0f}'.format(
    embedding_dims, embedding_dropout * 100, filters, kernel_size, dense_size, dense_dropout * 100)

print('Building model...')
model = Sequential()
model.add(Embedding(len(pfam.aa_list) + 1,
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
model.add(Dense(num_domain))
model.add(Activation('softmax'))

pfam.sparse_train(model, model_name, __file__, num_domain, device='/gpu:0', batch_size=batch_size)
