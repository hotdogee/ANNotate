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
filters = 32
kernel_size = 7
lstm_output_size = 32
dense_size = 32
dense_dropout = 0.2

model_name = 'ed{0}eo{1:.0f}cf{2}ks{3}ls{4}ds{5}do{6:.0f}'.format(
    embedding_dims, embedding_dropout * 100, filters, kernel_size, 
    lstm_output_size, dense_size, dense_dropout * 100)

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
# model.add(TimeDistributed(Conv1D(filters,
#                 kernel_size,
#                 padding='valid',
#                 activation='relu',
#                 strides=1)))
# model.add(GlobalMaxPooling1D())
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
# time_distributed_1 (TimeDist (None, None, 13)          845
# =================================================================
# Total params: 14,413
# Trainable params: 14,413
# Non-trainable params: 0
# _________________________________________________________________

#                 precision    recall  f1-score   support

#            PAD    1.00000   1.00000   1.00000 862187725
#      NO_DOMAIN    0.96240   0.96527   0.96383 157729784
# UNKNOWN_DOMAIN    0.99311   0.98970   0.99140  31941877
#        PF00005    0.99765   0.99818   0.99792  38678560
#        PF00115    0.94330   0.94382   0.94356  35287282
#        PF07690    0.76129   0.78206   0.77153   4328264
#        PF00400    0.74828   0.76825   0.75813   2389194
#        PF00096    0.97725   0.95595   0.96648  11183619
#        PF00072    0.95968   0.97149   0.96555  19053029
#        PF00528    0.76038   0.60600   0.67446   1474548
#        PF00106    0.95933   0.94758   0.95342  18041648
#        PF13561    0.97345   0.96788   0.97066   9156086

#    avg / total    0.98974   0.98975   0.98973 1191451616

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

# 980 Ti Ran out of memory with batch size 64 at
# 50527/53795 [===========================>..] - ETA: 12:09 - loss: 0.0235 - acc: 0.9914(64, 1357) (64, 1357, 1) 1357 1357