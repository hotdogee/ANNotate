'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function

from datasets import pfam_regions
from keras.models import Sequential
from keras.layers import Embedding, Dropout, TimeDistributed
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Dense, Activation, LSTM, Bidirectional
from keras.layers import CuDNNGRU, CuDNNLSTM

num_domain = 10
test_split = 0.2
validation_split = 0.1
batch_size = 16
epochs = 300

embedding_dims = 32
embedding_dropout = 0.2

filters = 24
kernel_size = 11
conv_dropout = 0.2
lstm_output_size = 24
dense_size = 32
dense_dropout = 0.2

filters2 = 16
kernel_size2 = 9
conv_dropout2 = 0.2
lstm_output_size2 = 16
dense_size2 = 32
dense_dropout2 = 0.2

filters3 = 12
kernel_size3 = 7
conv_dropout3 = 0.2
lstm_output_size3 = 12
dense_size3 = 24
dense_dropout3 = 0.2

model_name = 'ed{}eo{:.0f}ls{}cf{}ks{}cd{:.0f}ds{}dd{:.0f}ls{}cf{}ks{}cd{:.0f}ds{}dd{:.0f}ls{}cf{}ks{}cd{:.0f}ds{}dd{:.0f}'.format(
    embedding_dims, embedding_dropout * 100, 
    lstm_output_size, filters, kernel_size, conv_dropout * 100, dense_size, dense_dropout * 100, 
    lstm_output_size2, filters2, kernel_size2, conv_dropout2 * 100, dense_size2, dense_dropout2 * 100, 
    lstm_output_size3, filters3, kernel_size3, conv_dropout3 * 100, dense_size3, dense_dropout3 * 100)

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
model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
model.add(TimeDistributed(Dense(dense_size)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout)))

model.add(Conv1D(filters2,
                kernel_size2,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout2)))
model.add(Bidirectional(CuDNNGRU(lstm_output_size2, return_sequences=True)))
model.add(TimeDistributed(Dense(dense_size2)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout2)))

model.add(Conv1D(filters3,
                kernel_size3,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout3)))
model.add(Bidirectional(CuDNNGRU(lstm_output_size3, return_sequences=True)))
model.add(TimeDistributed(Dense(dense_size3)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout3)))

model.add(TimeDistributed(Dense(3 + num_domain, activation='softmax')))
model.summary()
epoch_start = 0
pfam_regions.sparse_train(model, model_name, __file__, num_domain, device='/gpu:0', 
    epoch_start=epoch_start, batch_size=batch_size, epochs=epochs,
    predicted_dir='C:/Users/Hotdogee/Documents/Annotate/predicted')

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 32)          896
# _________________________________________________________________
# dropout_1 (Dropout)          (None, None, 32)          0
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, None, 24)          8472
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, None, 24)          0
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, None, 48)          7200
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, None, 32)          1568
# _________________________________________________________________
# time_distributed_3 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# conv1d_2 (Conv1D)            (None, None, 16)          4624
# _________________________________________________________________
# time_distributed_5 (TimeDist (None, None, 16)          0
# _________________________________________________________________
# bidirectional_2 (Bidirection (None, None, 32)          3264
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, None, 32)          1056
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_8 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# conv1d_3 (Conv1D)            (None, None, 12)          2700
# _________________________________________________________________
# time_distributed_9 (TimeDist (None, None, 12)          0
# _________________________________________________________________
# bidirectional_3 (Bidirection (None, None, 24)          1872
# _________________________________________________________________
# time_distributed_10 (TimeDis (None, None, 24)          600
# _________________________________________________________________
# time_distributed_11 (TimeDis (None, None, 24)          0
# _________________________________________________________________
# time_distributed_12 (TimeDis (None, None, 24)          0
# _________________________________________________________________
# time_distributed_13 (TimeDis (None, None, 13)          325
# =================================================================
# Total params: 32,577
# Trainable params: 32,577
# Non-trainable params: 0
# _________________________________________________________________

# 8000/107590 [=>............................] - ETA: 41:30:43 - loss: 0.2026 - acc: 0.9337(32, 582) (32, 582, 1) 582 582

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