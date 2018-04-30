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
lstm_output_size = 16
lstm_output_size2 = 16
filters = 32
kernel_size = 7
conv_dropout = 0.2
dense_size = 32
dense_dropout = 0.2
dense_size2 = 24
dense_dropout2 = 0.2

model_name = 'ed{0}eo{1:.0f}ls{2}ls{3}ds{4}dd{5:.0f}ds{6}dd{7:.0f}'.format(
    embedding_dims, embedding_dropout * 100, lstm_output_size, lstm_output_size2, 
    dense_size, dense_dropout * 100, dense_size2, dense_dropout2 * 100)

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
model.add(Bidirectional(CuDNNGRU(lstm_output_size2, return_sequences=True)))
# model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))

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
    
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 32)          896
# _________________________________________________________________
# dropout_1 (Dropout)          (None, None, 32)          0
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, None, 32)          4800
# _________________________________________________________________
# bidirectional_2 (Bidirection (None, None, 32)          4800
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, None, 32)          1056
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_3 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, None, 24)          792
# _________________________________________________________________
# time_distributed_5 (TimeDist (None, None, 24)          0
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, None, 24)          0
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, None, 13)          325
# =================================================================
# Total params: 12,669
# Trainable params: 12,669
# Non-trainable params: 0
# _________________________________________________________________

# 1000/107590 [..............................] - ETA: 9:11:06 - loss: 0.4913 - acc: 0.8540(32, 1534) (32, 1534, 1) 1534 1534

#                 precision    recall  f1-score   support

#            PAD    1.00000   1.00000   1.00000 861332871
#      NO_DOMAIN    0.97612   0.96042   0.96821 157730248
# UNKNOWN_DOMAIN    0.99406   0.99643   0.99524  31941665
#        PF00005    0.99846   0.99859   0.99853  38678304
#        PF00115    0.91998   0.97814   0.94817  35286169
#        PF07690    0.82067   0.81509   0.81787   4328264
#        PF00400    0.78985   0.68823   0.73555   2389194
#        PF00096    0.97054   0.98709   0.97875  11183756
#        PF00072    0.97162   0.97337   0.97249  19052817
#        PF00528    0.79361   0.55782   0.65514   1474564
#        PF00106    0.93355   0.97975   0.95610  18042256
#        PF13561    0.98563   0.97281   0.97918   9156308

#    avg / total    0.99108   0.99106   0.99099 1190596416

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