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

model_name = 'ed{0}eo{1:.0f}cf{3}ks{4}cd{5:.0f}ls{2}'.format(
    embedding_dims, embedding_dropout * 100, lstm_output_size, filters, kernel_size, 
    conv_dropout * 100)

print('Building model...')
model = Sequential()
# (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dims)
model.add(Embedding(len(pfam_regions.aa_list) + 1,
                    embedding_dims,
                    input_length=None))
model.add(Dropout(embedding_dropout))
model.add(Conv1D(filters,
                kernel_size,
                padding='same',
                activation='relu',
                strides=1))
model.add(TimeDistributed(Dropout(conv_dropout)))
# Expected input batch shape: (batch_size, timesteps, data_dim)
# returns a sequence of vectors of dimension lstm_output_size
model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
# model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))

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
# conv1d_1 (Conv1D)            (None, None, 32)          7200
# _________________________________________________________________
# time_distributed_1 (TimeDist (None, None, 32)          0
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, None, 64)          12672
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, None, 13)          845
# =================================================================
# Total params: 21,613
# Trainable params: 21,613
# Non-trainable params: 0
# _________________________________________________________________

# 13000/53795 [======>.......................] - ETA: 7:24:26 - loss: 0.0776 - acc: 0.9745(64, 3163) (64, 3163, 1) 3163 3163
#                 precision    recall  f1-score   support

#            PAD    1.00000   1.00000   1.00000 1193413536
#      NO_DOMAIN    0.97528   0.96332   0.96926 157726252
# UNKNOWN_DOMAIN    0.99363   0.99630   0.99496  31941511
#        PF00005    0.99785   0.99845   0.99815  38676761
#        PF00115    0.93745   0.96213   0.94963  35284737
#        PF07690    0.86176   0.74662   0.80007   4328106
#        PF00400    0.76236   0.86592   0.81085   2389171
#        PF00096    0.98724   0.96933   0.97820  11183380
#        PF00072    0.96073   0.97682   0.96871  19051627
#        PF00528    0.77101   0.52253   0.62290   1474546
#        PF00106    0.91451   0.98523   0.94856  18041586
#        PF13561    0.97664   0.98877   0.98267   9155859

#    avg / total    0.99308   0.99305   0.99300 1522667072

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