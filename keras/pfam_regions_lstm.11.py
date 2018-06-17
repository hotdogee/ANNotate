'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function

from datasets import pfam_regions
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, TimeDistributed
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Dense, Activation, LSTM, Bidirectional
from keras.layers import CuDNNGRU, CuDNNLSTM

num_domain = 16712
test_split = 0.2
validation_split = 0.1
batch_size = 32
epochs = 300

embedding_dims = 32
embedding_dropout = 0.2

lstm_output_size = 24
filters = 24
kernel_size = 11
conv_dropout = 0.2
dense_size = 32
dense_dropout = 0.2

lstm_output_size2 = 16
filters2 = 16
kernel_size2 = 9
conv_dropout2 = 0.2
dense_size2 = 32
dense_dropout2 = 0.2

lstm_output_size3 = 12
filters3 = 12
kernel_size3 = 7
conv_dropout3 = 0.2
dense_size3 = 24
dense_dropout3 = 0.2

model_name = 'ed{}eo{:.0f}cf{}ks{}cd{:.0f}cf{}ks{}cd{:.0f}ls{}ls{}ds{}dd{:.0f}ds{}dd{:.0f}'.format(
    embedding_dims, embedding_dropout * 100, filters, kernel_size, conv_dropout * 100, 
    filters2, kernel_size2, conv_dropout2 * 100, 
    lstm_output_size, lstm_output_size2, dense_size, dense_dropout * 100, 
    dense_size2, dense_dropout2 * 100)

print('Building model...')
model = Sequential()
# (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dims)
model.add(Embedding(len(pfam_regions.aa_list) + 1,
                    embedding_dims,
                    input_length=None))
model.add(Dropout(embedding_dropout))
# Expected input batch shape: (batch_size, timesteps, data_dim)
# returns a sequence of vectors of dimension lstm_output_size
# model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))


model.add(Conv1D(filters,
                kernel_size,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout)))
model.add(Conv1D(filters2,
                kernel_size2,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout2)))

model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
model.add(Bidirectional(CuDNNGRU(lstm_output_size2, return_sequences=True)))

model.add(TimeDistributed(Dense(dense_size)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout)))
model.add(TimeDistributed(Dense(dense_size2)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout2)))

model.add(TimeDistributed(Dense(3 + num_domain, activation='softmax')))
model.summary()
epoch_start = 0
pfam_regions.sparse_train(model, model_name, __file__, num_domain, device='/gpu:0', 
    epoch_start=epoch_start, batch_size=batch_size, epochs=epochs,
    predicted_dir='C:/Users/Hotdogee/Documents/Annotate/predicted')


# Pfam-A.regions.uniprot.tsv.gz
#        Region Count:       88761542
#      Sequence Count:       54223493
#        Domain Count:          16712
# Region length:
#                 Min:              3
#              Median:            127
#             Average:            156
#                 Max:           2161
# Regions per sequence:
#                 Min:              1
#              Median:              1
#             Average:              1
#                 Max:            572
# Sequences per domain:
#                 Min:              2
#              Median:            863
#             Average:           5311
#                 Max:        1078482