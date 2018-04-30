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
batch_size = 64
epochs = 300

embedding_dims = 32
embedding_dropout = 0.2
lstm_output_size = 32
filters = 32
kernel_size = 7
conv_dropout = 0.2
dense_size = 32
dense_dropout = 0.2

model_name = 'ed{0}eo{1:.0f}ls{2}cf{3}ks{4}cd{5:.0f}'.format(
    embedding_dims, embedding_dropout * 100, lstm_output_size, filters, kernel_size, 
    conv_dropout * 100)

print('Building model...')
model = Sequential()
# (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dims)
model.add(Embedding(len(pfam_regions.aa_list) + 1,
                    embedding_dims,
                    input_length=None))
model.add(Dropout(embedding_dropout))
# Expected input batch shape: (batch_size, timesteps, data_dim)
# returns a sequence of vectors of dimension lstm_output_size
model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
# model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))

model.add(Conv1D(filters,
                kernel_size,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout)))
# model.add(TimeDistributed(Dense(dense_size)))
# model.add(TimeDistributed(Activation('relu')))
# model.add(TimeDistributed(Dropout(dense_dropout)))
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
# conv1d_1 (Conv1D)            (None, None, 32)          14368
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, None, 13)          429
# =================================================================
# Total params: 28,365
# Trainable params: 28,365
# Non-trainable params: 0
# _________________________________________________________________

#                 precision    recall  f1-score   support

#            PAD    1.00000   1.00000   1.00000 1189150651
#      NO_DOMAIN    0.97359   0.95725   0.96535 157723509
# UNKNOWN_DOMAIN    0.99411   0.99394   0.99402  31939394
#        PF00005    0.99772   0.99814   0.99793  38677140
#        PF00115    0.92723   0.96996   0.94811  35285154
#        PF07690    0.73101   0.84635   0.78446   4328226
#        PF00400    0.73268   0.80325   0.76634   2388777
#        PF00096    0.96593   0.97551   0.97070  11183501
#        PF00072    0.98465   0.95273   0.96843  19052443
#        PF00528    0.85369   0.52277   0.64845   1474544
#        PF00106    0.93291   0.97490   0.95344  18041302
#        PF13561    0.95117   0.98895   0.96969   9156191

#    avg / total    0.99252   0.99234   0.99235 1518400832


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
