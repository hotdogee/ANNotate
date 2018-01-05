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
batch_size = 32
epochs = 300

embedding_dims = 32
embedding_dropout = 0.2
lstm_output_size = 32
filters = 32
kernel_size = 7
conv_dropout = 0.2
dense_size = 32
dense_dropout = 0.2

model_name = 'ed{0}eo{1:.0f}ls{2}ds{3}dd{4:.0f}'.format(
    embedding_dims, embedding_dropout * 100, lstm_output_size, dense_size, 
    dense_dropout * 100)

print('Building model...')
model = Sequential()
# (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dims)
model.add(Embedding(len(pfam_regions.aa_list) + 1,
                    embedding_dims,
                    input_length=None))
model.add(Dropout(embedding_dropout))
# model.add(Conv1D(filters,
#                 kernel_size,
#                 padding='same',
#                 activation='relu',
#                 strides=1))
# model.add(TimeDistributed(Dropout(conv_dropout)))
# Expected input batch shape: (batch_size, timesteps, data_dim)
# returns a sequence of vectors of dimension lstm_output_size
model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
# model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))

model.add(TimeDistributed(Dense(dense_size)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(dense_dropout)))
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
# bidirectional_1 (Bidirection (None, None, 64)          12672
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, None, 32)          2080
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_3 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, None, 13)          429
# =================================================================
# Total params: 16,077
# Trainable params: 16,077
# Non-trainable params: 0
# _________________________________________________________________

#                 precision    recall  f1-score   support

#            PAD    1.00000   1.00000   1.00000 862358460
#      NO_DOMAIN    0.96755   0.96380   0.96567 157730093
# UNKNOWN_DOMAIN    0.99081   0.99494   0.99287  31941925
#        PF00005    0.99861   0.99797   0.99829  38679125
#        PF00115    0.92991   0.96485   0.94706  35285414
#        PF07690    0.80492   0.70804   0.75338   4328191
#        PF00400    0.72832   0.80425   0.76440   2389194
#        PF00096    0.98424   0.96635   0.97521  11183701
#        PF00072    0.97623   0.96702   0.97160  19053041
#        PF00528    0.75916   0.61719   0.68085   1474398
#        PF00106    0.95814   0.95516   0.95665  18042236
#        PF13561    0.96365   0.98093   0.97222   9156078

#    avg / total    0.99035   0.99037   0.99032 1191621856

#   9602/107590 [=>............................] - ETA: 4:31:43 - loss: 0.2199 - acc: 0.9257(32, 1679) (32, 1679, 1) 1679 1679

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