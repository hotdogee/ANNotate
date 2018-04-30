'''Trains and evaluate a LSTM network on the Pfam Domain classification task.
'''
from __future__ import print_function
import os
import errno

import tensorflow as tf
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
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

num_domain = 10
test_split = 0.2
validation_split = 0.1
maxlen = 512
batch_size = 256
epochs = 150

embedding_dims = 10
embedding_dropout = 0.2
filters = 128
kernel_size = 5
lstm_output_size = 64
dense_size = 128
dense_dropout = 0.2

model_name = 'nd{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}ed{6}eo{7:.0f}cf{8}ks{9}ls{12}ds{10}do{11:.0f}'.format(
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
csvlog_path = '{0}.{1}.csv'.format(os.path.abspath(os.path.join(csvlog_dir,
    os.path.splitext(os.path.basename(__file__))[0])), model_name)
model_dir = './models'
if not os.path.exists(model_dir):
    try:
        os.makedirs(model_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
confusion_matrix_dir = './confusion_matrix'
if not os.path.exists(confusion_matrix_dir):
    try:
        os.makedirs(confusion_matrix_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
classification_report_dir = './classification_report'
if not os.path.exists(classification_report_dir):
    try:
        os.makedirs(classification_report_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with tf.device('/gpu:0'):
    print('Loading data...')
    (x_train, y_train_class), (x_test, y_test_class), (aa_list, domain_list) = pfam.load_data(
        num_domain=num_domain, test_split=test_split)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    # print(domain_list)

    num_classes = np.max(y_train_class) + 1
    print(num_classes, 'classes')

    print('Pad sequences')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
        '(for use with categorical_crossentropy)')
    y_train = keras.utils.to_categorical(y_train_class, num_classes)
    y_test = keras.utils.to_categorical(y_test_class, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Embedding(len(aa_list) + 1,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(embedding_dropout))
    model.add(LSTM(lstm_output_size, return_sequences=True))
    model.add(Conv1D(filters,
                    kernel_size,
                    padding='valid',
                    activation='relu',
                    strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(dense_size))
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    save_freq = 1
    epoch_start = 91
    for epoch in range(epoch_start, epochs):
        model_path = '{0}.e{1}.{2}.h5'.format(os.path.abspath(os.path.join(model_dir,
            os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
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
            if epoch % save_freq == 0:
                model.save(model_path)
        score = model.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        print(classification_report(y_test_class, y_pred_class, target_names=domain_list, digits=5))

        confusion_matrix_path = '{0}.e{1}.{2}.cm.csv'.format(os.path.abspath(os.path.join(confusion_matrix_dir, 
            os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
        if not os.path.exists(confusion_matrix_path):
            np.savetxt(confusion_matrix_path, confusion_matrix(y_test_class, y_pred_class), fmt='%d', delimiter=",")

        classification_report_path = '{0}.e{1}.{2}.cr.txt'.format(os.path.abspath(os.path.join(
            classification_report_dir, os.path.splitext(os.path.basename(__file__))[0])), epoch, model_name)
        if not os.path.exists(classification_report_path):
            np.savetxt(classification_report_path, [classification_report(y_test_class, y_pred_class, 
                target_names=domain_list, digits=5)], fmt='%s')

# epoch,acc,loss,val_acc,val_loss
# 297679/297679 [==============================] 100.00% - 2197s
# Test score: 0.0059061792766
# Test accuracy: 0.998199402712
# Loading model: pfam_lstm.1.e1.nd10ts20vs10ml512bs32ep3ed10eo20cf128ks5ls64ds128do20.h5
# 297679/297679 [==============================] 100.00% - 2206s
# Test score: 0.00137261035171
# Test accuracy: 0.999559928648
# Loading model: pfam_lstm.1.e2.nd10ts20vs10ml512bs32ep3ed10eo20cf128ks5ls64ds128do20.h5
# 297679/297679 [==============================] 100.00% - 2211s
# Test score: 0.00175688161021
# Test accuracy: 0.999418837069
# 1071643/1071643 [==============================] 100.00% - 33785s - loss: 0.0051 - acc: 0.9986 - val_loss: 0.0021 - val_acc: 0.9994
# 297679/297679 [==============================] 100.00% - 2206s
# Test score: 0.00175688161021
# Test accuracy: 0.999418837069
# 1071643/1071643 [==============================] 100.00% - 3840s - loss: 5.7477e-04 - acc: 0.9998 - val_loss: 4.3981e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000467091639998
# Test accuracy: 0.99989922026
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support
# 980Ti
#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99976   0.99997   0.99987     37300
# PF00072.23;Response_reg;    0.99976   1.00000   0.99988     28684
#      PF00069.24;Pkinase;    0.99974   1.00000   0.99987     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99966   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99993   0.99874   0.99933     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 32/32
# 1071643/1071643 [==============================] 100.00% - 3838s - loss: 5.7238e-04 - acc: 0.9998 - val_loss: 0.0010 - val_acc: 0.9997
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.00104897792174
# Test accuracy: 0.999778284621
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99982   0.99991     57079
#         PF00400.31;WD40;    1.00000   0.99972   0.99986     39257
#        PF07690.15;MFS_1;    0.99962   0.99997   0.99980     37300
# PF00072.23;Response_reg;    0.99923   1.00000   0.99962     28684
#      PF00069.24;Pkinase;    0.99985   0.99992   0.99989     26514
#    PF02518.25;HATPase_c;    0.99971   1.00000   0.99985     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99924   0.99962     23644
#      PF00096.25;zf-C2H2;    0.99979   1.00000   0.99989     23394
#         PF12796.6;Ank_2;    0.99987   1.00000   0.99993     22846
#  PF00501.27;AMP-binding;    0.99927   0.99840   0.99884     15032

#              avg / total    0.99978   0.99978   0.99978    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 33/33
# 1071643/1071643 [==============================] 100.00% - 3818s - loss: 6.7458e-04 - acc: 0.9998 - val_loss: 8.4656e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000867895172178
# Test accuracy: 0.999815237177
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99995   0.99998   0.99996     57079
#         PF00400.31;WD40;    0.99975   0.99997   0.99986     39257
#        PF07690.15;MFS_1;    0.99989   0.99987   0.99988     37300
# PF00072.23;Response_reg;    0.99930   1.00000   0.99965     28684
#      PF00069.24;Pkinase;    0.99985   0.99985   0.99985     26514
#    PF02518.25;HATPase_c;    0.99979   0.99996   0.99987     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99987   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99979   0.99989     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99973   0.99767   0.99870     15032

#              avg / total    0.99982   0.99982   0.99982    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 34/34
# 1071643/1071643 [==============================] 100.00% - 3856s - loss: 4.5552e-04 - acc: 0.9999 - val_loss: 7.8795e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 219s
# Test score: 0.000416740238528
# Test accuracy: 0.99988914229
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99979   0.99992   0.99985     37300
# PF00072.23;Response_reg;    0.99979   0.99986   0.99983     28684
#      PF00069.24;Pkinase;    0.99977   0.99992   0.99985     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99979   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99987   0.99996   0.99991     22846
#  PF00501.27;AMP-binding;    0.99967   0.99920   0.99943     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 35/35
# 1071643/1071643 [==============================] 100.00% - 3819s - loss: 5.1687e-04 - acc: 0.9999 - val_loss: 8.2907e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 215s
# Test score: 0.000806363028586
# Test accuracy: 0.999852189733
# 297679/297679 [==============================] 100.00% - 217s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    1.00000   0.99977   0.99989     39257
#        PF07690.15;MFS_1;    0.99973   1.00000   0.99987     37300
# PF00072.23;Response_reg;    0.99972   0.99983   0.99977     28684
#      PF00069.24;Pkinase;    0.99974   0.99992   0.99983     26514
#    PF02518.25;HATPase_c;    0.99996   0.99992   0.99994     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99970   0.99985     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99974   1.00000   0.99987     22846
#  PF00501.27;AMP-binding;    0.99927   0.99894   0.99910     15032

#              avg / total    0.99985   0.99985   0.99985    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 36/36
# 1071643/1071643 [==============================] 100.00% - 3777s - loss: 4.7919e-04 - acc: 0.9999 - val_loss: 4.6514e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 215s
# Test score: 0.000418875860445
# Test accuracy: 0.99990929823
# 297679/297679 [==============================] 100.00% - 217s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   1.00000   0.99999     57079
#         PF00400.31;WD40;    1.00000   0.99992   0.99996     39257
#        PF07690.15;MFS_1;    0.99984   0.99997   0.99991     37300
# PF00072.23;Response_reg;    0.99976   0.99990   0.99983     28684
#      PF00069.24;Pkinase;    0.99970   1.00000   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99970   0.99985     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99967   0.99920   0.99943     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 37/37
# 1071643/1071643 [==============================] 100.00% - 3823s - loss: 6.6698e-04 - acc: 0.9998 - val_loss: 6.3752e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000449767162859
# Test accuracy: 0.999905938906
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99996   0.99997     57079
#         PF00400.31;WD40;    0.99995   0.99995   0.99995     39257
#        PF07690.15;MFS_1;    0.99989   0.99995   0.99992     37300
# PF00072.23;Response_reg;    0.99990   0.99986   0.99988     28684
#      PF00069.24;Pkinase;    0.99974   0.99992   0.99983     26514
#    PF02518.25;HATPase_c;    1.00000   0.99979   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99987   0.99992     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99969   1.00000   0.99985     22846
#  PF00501.27;AMP-binding;    0.99980   0.99947   0.99963     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 38/38
# 1071643/1071643 [==============================] 100.00% - 3911s - loss: 6.3642e-04 - acc: 0.9999 - val_loss: 6.4751e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000813380068166
# Test accuracy: 0.99983875244
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99995   0.99995   0.99995     57079
#         PF00400.31;WD40;    0.99995   0.99995   0.99995     39257
#        PF07690.15;MFS_1;    0.99976   0.99989   0.99983     37300
# PF00072.23;Response_reg;    0.99983   0.99993   0.99988     28684
#      PF00069.24;Pkinase;    0.99943   0.99992   0.99968     26514
#    PF02518.25;HATPase_c;    0.99979   0.99992   0.99985     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99966   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99974   1.00000   0.99987     22846
#  PF00501.27;AMP-binding;    0.99987   0.99834   0.99910     15032

#              avg / total    0.99984   0.99984   0.99984    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 39/39
# 1071643/1071643 [==============================] 100.00% - 3893s - loss: 5.5694e-04 - acc: 0.9999 - val_loss: 5.6760e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 215s
# Test score: 0.000458884488759
# Test accuracy: 0.99988914229
# 297679/297679 [==============================] 100.00% - 217s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99991   0.99998   0.99995     57079
#         PF00400.31;WD40;    1.00000   0.99992   0.99996     39257
#        PF07690.15;MFS_1;    0.99981   0.99987   0.99984     37300
# PF00072.23;Response_reg;    0.99983   0.99983   0.99983     28684
#      PF00069.24;Pkinase;    0.99970   1.00000   0.99985     26514
#    PF02518.25;HATPase_c;    0.99992   0.99987   0.99990     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99975   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99960   0.99933   0.99947     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 40/40
# 1071643/1071643 [==============================] 100.00% - 3877s - loss: 3.9411e-04 - acc: 0.9999 - val_loss: 0.0010 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 215s
# Test score: 0.000753237263062
# Test accuracy: 0.999872345673
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99993   0.99995     57079
#         PF00400.31;WD40;    0.99990   0.99997   0.99994     39257
#        PF07690.15;MFS_1;    0.99979   0.99995   0.99987     37300
# PF00072.23;Response_reg;    0.99969   0.99990   0.99979     28684
#      PF00069.24;Pkinase;    0.99989   0.99977   0.99983     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99970   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99947   0.99927   0.99937     15032

#              avg / total    0.99987   0.99987   0.99987    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 41/41
# 1071643/1071643 [==============================] 100.00% - 3858s - loss: 5.8590e-04 - acc: 0.9999 - val_loss: 0.0011 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000916381290939
# Test accuracy: 0.999781643944
# 297679/297679 [==============================] 100.00% - 217s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99995   0.99996     57079
#         PF00400.31;WD40;    0.99992   0.99992   0.99992     39257
#        PF07690.15;MFS_1;    0.99987   0.99989   0.99988     37300
# PF00072.23;Response_reg;    0.99937   0.99997   0.99967     28684
#      PF00069.24;Pkinase;    0.99887   0.99992   0.99940     26514
#    PF02518.25;HATPase_c;    1.00000   0.99983   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99958   0.99977     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99987   0.99993     22846
#  PF00501.27;AMP-binding;    0.99953   0.99767   0.99860     15032

#              avg / total    0.99978   0.99978   0.99978    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 42/42
# 1071643/1071643 [==============================] 100.00% - 3914s - loss: 4.8680e-04 - acc: 0.9999 - val_loss: 7.8866e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000900008556225
# Test accuracy: 0.999801799884
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99992   0.99997   0.99995     39257
#        PF07690.15;MFS_1;    0.99984   0.99984   0.99984     37300
# PF00072.23;Response_reg;    0.99976   0.99990   0.99983     28684
#      PF00069.24;Pkinase;    0.99864   1.00000   0.99932     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99975   0.99983     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99996   0.99991   0.99993     22846
#  PF00501.27;AMP-binding;    1.00000   0.99747   0.99873     15032

#              avg / total    0.99980   0.99980   0.99980    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 43/43
# 1071643/1071643 [==============================] 100.00% - 3895s - loss: 5.9516e-04 - acc: 0.9999 - val_loss: 5.3544e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000511933074239
# Test accuracy: 0.999862267703
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99992   0.99971   0.99981     37300
# PF00072.23;Response_reg;    0.99958   0.99997   0.99977     28684
#      PF00069.24;Pkinase;    0.99970   0.99992   0.99981     26514
#    PF02518.25;HATPase_c;    0.99992   0.99987   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99958   0.99996   0.99977     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99996   0.99998     22846
#  PF00501.27;AMP-binding;    0.99967   0.99887   0.99927     15032

#              avg / total    0.99986   0.99986   0.99986    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 44/44
# 1071643/1071643 [==============================] 100.00% - 3879s - loss: 4.6592e-04 - acc: 0.9999 - val_loss: 6.2716e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000946045993275
# Test accuracy: 0.999855549057
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99987   1.00000   0.99994     39257
#        PF07690.15;MFS_1;    0.99992   0.99987   0.99989     37300
# PF00072.23;Response_reg;    0.99930   1.00000   0.99965     28684
#      PF00069.24;Pkinase;    0.99974   0.99992   0.99983     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99979   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99991   0.99996     22846
#  PF00501.27;AMP-binding;    0.99987   0.99834   0.99910     15032

#              avg / total    0.99986   0.99986   0.99986    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 45/45
# 1071643/1071643 [==============================] 100.00% - 3870s - loss: 4.6511e-04 - acc: 0.9999 - val_loss: 4.4556e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000432254742152
# Test accuracy: 0.999932813493
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99993   0.99998   0.99996     57079
#         PF00400.31;WD40;    1.00000   1.00000   1.00000     39257
#        PF07690.15;MFS_1;    0.99987   1.00000   0.99993     37300
# PF00072.23;Response_reg;    0.99993   0.99997   0.99995     28684
#      PF00069.24;Pkinase;    0.99989   0.99985   0.99987     26514
#    PF02518.25;HATPase_c;    0.99996   0.99983   0.99990     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99987   0.99994     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99967   0.99953   0.99960     15032

#              avg / total    0.99993   0.99993   0.99993    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 46/46
# 1071643/1071643 [==============================] 100.00% - 3882s - loss: 4.8568e-04 - acc: 0.9999 - val_loss: 4.6463e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000427382572405
# Test accuracy: 0.999916016876
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99996   0.99997     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99987   0.99997   0.99992     37300
# PF00072.23;Response_reg;    0.99993   0.99990   0.99991     28684
#      PF00069.24;Pkinase;    0.99981   0.99996   0.99989     26514
#    PF02518.25;HATPase_c;    1.00000   0.99983   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99987   0.99992     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99967   0.99927   0.99947     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 47/47
# 1071643/1071643 [==============================] 100.00% - 3903s - loss: 5.2835e-04 - acc: 0.9999 - val_loss: 6.5326e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000643199063036
# Test accuracy: 0.99985890838
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99991   0.99998   0.99995     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99981   0.99997   0.99989     37300
# PF00072.23;Response_reg;    0.99962   0.99997   0.99979     28684
#      PF00069.24;Pkinase;    0.99977   0.99966   0.99972     26514
#    PF02518.25;HATPase_c;    0.99996   0.99992   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99983   0.99989     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99969   1.00000   0.99985     22846
#  PF00501.27;AMP-binding;    0.99980   0.99847   0.99913     15032

#              avg / total    0.99986   0.99986   0.99986    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 48/48
# 1071643/1071643 [==============================] 100.00% - 3892s - loss: 4.5513e-04 - acc: 0.9999 - val_loss: 7.7686e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.00108944181612
# Test accuracy: 0.999801799884
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99996   0.99996     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99995   0.99981   0.99988     37300
# PF00072.23;Response_reg;    0.99965   1.00000   0.99983     28684
#      PF00069.24;Pkinase;    0.99876   1.00000   0.99938     26514
#    PF02518.25;HATPase_c;    1.00000   0.99979   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99979   0.99992   0.99985     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   0.99996   0.99996     22846
#  PF00501.27;AMP-binding;    0.99980   0.99721   0.99850     15032

#              avg / total    0.99980   0.99980   0.99980    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 49/49
# 1071643/1071643 [==============================] 100.00% - 3889s - loss: 3.8619e-04 - acc: 0.9999 - val_loss: 5.0300e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000350782263977
# Test accuracy: 0.999919376199
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99992   0.99995   0.99993     37300
# PF00072.23;Response_reg;    0.99983   0.99997   0.99990     28684
#      PF00069.24;Pkinase;    0.99981   0.99989   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   0.99987   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99987   0.99989     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99953   0.99933   0.99943     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 50/50
# 1071643/1071643 [==============================] 100.00% - 3972s - loss: 5.1754e-04 - acc: 0.9999 - val_loss: 5.5431e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 219s
# Test score: 0.000517285204137
# Test accuracy: 0.999885782966
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99984   1.00000   0.99992     37300
# PF00072.23;Response_reg;    0.99979   0.99997   0.99988     28684
#      PF00069.24;Pkinase;    0.99962   0.99992   0.99977     26514
#    PF02518.25;HATPase_c;    0.99987   0.99987   0.99987     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99983   0.99992     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99987   1.00000   0.99993     22846
#  PF00501.27;AMP-binding;    0.99987   0.99860   0.99923     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 51/51
# 1071643/1071643 [==============================] 100.00% - 3957s - loss: 5.0768e-04 - acc: 0.9999 - val_loss: 5.2784e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000572717872851
# Test accuracy: 0.999875704997
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   1.00000   0.99998     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99987   0.99968   0.99977     37300
# PF00072.23;Response_reg;    0.99979   0.99993   0.99986     28684
#      PF00069.24;Pkinase;    0.99981   0.99992   0.99987     26514
#    PF02518.25;HATPase_c;    0.99996   0.99992   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99949   0.99992   0.99970     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99980   0.99887   0.99933     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 52/52
# 1071643/1071643 [==============================] 100.00% - 3923s - loss: 4.4773e-04 - acc: 0.9999 - val_loss: 4.0037e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000563037247127
# Test accuracy: 0.999902579583
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99991   0.99998   0.99995     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99984   0.99997   0.99991     37300
# PF00072.23;Response_reg;    0.99986   0.99990   0.99988     28684
#      PF00069.24;Pkinase;    0.99977   0.99981   0.99979     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99983   0.99989     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99967   0.99920   0.99943     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 53/53
# 1071643/1071643 [==============================] 100.00% - 3893s - loss: 4.4963e-04 - acc: 0.9999 - val_loss: 3.7243e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000560942431674
# Test accuracy: 0.999875704997
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99993   0.99998   0.99996     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99981   0.99979   0.99980     37300
# PF00072.23;Response_reg;    0.99983   0.99997   0.99990     28684
#      PF00069.24;Pkinase;    0.99977   0.99989   0.99983     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99966   0.99992   0.99979     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99973   0.99874   0.99923     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 54/54
# 1071643/1071643 [==============================] 100.00% - 3894s - loss: 4.8627e-04 - acc: 0.9999 - val_loss: 4.5671e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000419254772534
# Test accuracy: 0.999932813493
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    1.00000   1.00000   1.00000     39257
#        PF07690.15;MFS_1;    0.99989   1.00000   0.99995     37300
# PF00072.23;Response_reg;    0.99986   0.99997   0.99991     28684
#      PF00069.24;Pkinase;    0.99981   0.99996   0.99989     26514
#    PF02518.25;HATPase_c;    1.00000   0.99992   0.99996     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99992   0.99996     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99982   1.00000   0.99991     22846
#  PF00501.27;AMP-binding;    0.99980   0.99927   0.99953     15032

#              avg / total    0.99993   0.99993   0.99993    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 55/55
# 1071643/1071643 [==============================] 100.00% - 3891s - loss: 3.6837e-04 - acc: 0.9999 - val_loss: 4.4152e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000385530047606
# Test accuracy: 0.999912657553
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99996   0.99997     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99984   0.99997   0.99991     37300
# PF00072.23;Response_reg;    0.99990   0.99997   0.99993     28684
#      PF00069.24;Pkinase;    0.99970   0.99992   0.99981     26514
#    PF02518.25;HATPase_c;    0.99992   0.99996   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99983   0.99989     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99980   0.99900   0.99940     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 56/56
# 1071643/1071643 [==============================] 100.00% - 3896s - loss: 4.1137e-04 - acc: 0.9999 - val_loss: 0.0013 - val_acc: 0.9997
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.00102226359278
# Test accuracy: 0.999791721914
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99995   0.99996     57079
#         PF00400.31;WD40;    1.00000   0.99997   0.99999     39257
#        PF07690.15;MFS_1;    0.99941   0.99997   0.99969     37300
# PF00072.23;Response_reg;    0.99983   0.99993   0.99988     28684
#      PF00069.24;Pkinase;    0.99970   0.99981   0.99975     26514
#    PF02518.25;HATPase_c;    0.99946   0.99992   0.99969     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99966   0.99979     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99969   1.00000   0.99985     22846
#  PF00501.27;AMP-binding;    0.99987   0.99734   0.99860     15032

#              avg / total    0.99979   0.99979   0.99979    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 57/57
# 1071643/1071643 [==============================] 100.00% - 3893s - loss: 4.8635e-04 - acc: 0.9999 - val_loss: 2.9014e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000457407073655
# Test accuracy: 0.999912657553
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   1.00000   0.99999     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99971   0.99995   0.99983     37300
# PF00072.23;Response_reg;    0.99993   0.99990   0.99991     28684
#      PF00069.24;Pkinase;    0.99989   0.99985   0.99987     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99970   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99987   1.00000   0.99993     22846
#  PF00501.27;AMP-binding;    0.99987   0.99940   0.99963     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 58/58
# 1071643/1071643 [==============================] 100.00% - 3925s - loss: 5.2073e-04 - acc: 0.9999 - val_loss: 5.0440e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000455195852262
# Test accuracy: 0.999885782966
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99989   0.99979   0.99984     37300
# PF00072.23;Response_reg;    0.99986   0.99993   0.99990     28684
#      PF00069.24;Pkinase;    0.99943   0.99996   0.99970     26514
#    PF02518.25;HATPase_c;    1.00000   0.99983   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99979   0.99987   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99980   0.99920   0.99950     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 59/59
# 1071643/1071643 [==============================] 100.00% - 3904s - loss: 6.6187e-04 - acc: 0.9999 - val_loss: 0.0010 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 219s
# Test score: 0.000727001169871
# Test accuracy: 0.999872345673
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99990   1.00000   0.99995     39257
#        PF07690.15;MFS_1;    0.99973   0.99992   0.99983     37300
# PF00072.23;Response_reg;    0.99997   0.99986   0.99991     28684
#      PF00069.24;Pkinase;    0.99985   0.99974   0.99979     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99975   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99948   1.00000   0.99974     22846
#  PF00501.27;AMP-binding;    0.99973   0.99894   0.99933     15032

#              avg / total    0.99987   0.99987   0.99987    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 60/60
# 1071643/1071643 [==============================] 100.00% - 3907s - loss: 4.1569e-04 - acc: 0.9999 - val_loss: 8.4980e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000555350039502
# Test accuracy: 0.99990929823
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    1.00000   1.00000   1.00000     39257
#        PF07690.15;MFS_1;    0.99987   1.00000   0.99993     37300
# PF00072.23;Response_reg;    0.99958   1.00000   0.99979     28684
#      PF00069.24;Pkinase;    0.99992   0.99977   0.99985     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99987   0.99994     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99967   0.99927   0.99947     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 61/61
# 1071643/1071643 [==============================] 100.00% - 3916s - loss: 4.2798e-04 - acc: 0.9999 - val_loss: 7.3201e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 219s
# Test score: 0.000738888466032
# Test accuracy: 0.99986898635
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99995   0.99996   0.99996     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99981   0.99989   0.99985     37300
# PF00072.23;Response_reg;    0.99986   0.99990   0.99988     28684
#      PF00069.24;Pkinase;    0.99959   0.99985   0.99972     26514
#    PF02518.25;HATPase_c;    1.00000   0.99992   0.99996     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99983   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99974   1.00000   0.99987     22846
#  PF00501.27;AMP-binding;    0.99980   0.99867   0.99923     15032

#              avg / total    0.99987   0.99987   0.99987    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 62/62
# 1071643/1071643 [==============================] 100.00% - 3938s - loss: 4.6249e-04 - acc: 0.9999 - val_loss: 4.5329e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000646494493749
# Test accuracy: 0.99989922026
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99989   0.99992   0.99991     37300
# PF00072.23;Response_reg;    0.99955   1.00000   0.99977     28684
#      PF00069.24;Pkinase;    0.99989   1.00000   0.99994     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99979   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99987   0.99874   0.99930     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 63/63
# 1071643/1071643 [==============================] 100.00% - 3932s - loss: 3.7731e-04 - acc: 0.9999 - val_loss: 0.0011 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.00149263791791
# Test accuracy: 0.999788362591
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99995   0.99946   0.99971     37300
# PF00072.23;Response_reg;    0.99882   1.00000   0.99941     28684
#      PF00069.24;Pkinase;    0.99985   0.99996   0.99991     26514
#    PF02518.25;HATPase_c;    1.00000   0.99987   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99924   0.99983   0.99953     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    1.00000   0.99991   0.99996     22846
#  PF00501.27;AMP-binding;    0.99973   0.99820   0.99897     15032

#              avg / total    0.99979   0.99979   0.99979    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 64/64
# 1071643/1071643 [==============================] 100.00% - 3978s - loss: 4.2866e-04 - acc: 0.9999 - val_loss: 5.1366e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 222s
# Test score: 0.000564799516107
# Test accuracy: 0.999875704997
# 297679/297679 [==============================] 100.00% - 221s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99971   0.99995   0.99983     37300
# PF00072.23;Response_reg;    0.99965   0.99997   0.99981     28684
#      PF00069.24;Pkinase;    0.99959   1.00000   0.99979     26514
#    PF02518.25;HATPase_c;    1.00000   0.99987   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99962   0.99975     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99993   0.99867   0.99930     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 65/65
# 1071643/1071643 [==============================] 100.00% - 3937s - loss: 3.7949e-04 - acc: 0.9999 - val_loss: 7.4827e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000507062224132
# Test accuracy: 0.999895860936
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99995   0.99996     57079
#         PF00400.31;WD40;    1.00000   0.99997   0.99999     39257
#        PF07690.15;MFS_1;    0.99965   0.99989   0.99977     37300
# PF00072.23;Response_reg;    0.99997   0.99993   0.99995     28684
#      PF00069.24;Pkinase;    0.99981   0.99996   0.99989     26514
#    PF02518.25;HATPase_c;    1.00000   0.99983   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99966   0.99975     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99996   0.99998     22846
#  PF00501.27;AMP-binding;    0.99953   0.99953   0.99953     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 66/66
# 1071643/1071643 [==============================] 100.00% - 3905s - loss: 5.1244e-04 - acc: 0.9999 - val_loss: 5.1149e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000586188605664
# Test accuracy: 0.999885782966
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99989   1.00000   0.99995     37300
# PF00072.23;Response_reg;    0.99944   0.99997   0.99970     28684
#      PF00069.24;Pkinase;    0.99962   1.00000   0.99981     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99987   0.99994     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99996   0.99998     22846
#  PF00501.27;AMP-binding;    0.99980   0.99840   0.99910     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 67/67
# 1071643/1071643 [==============================] 100.00% - 3914s - loss: 5.0759e-04 - acc: 0.9999 - val_loss: 7.0658e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.00051847017778
# Test accuracy: 0.99990929823
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99976   0.99997   0.99987     37300
# PF00072.23;Response_reg;    0.99993   0.99986   0.99990     28684
#      PF00069.24;Pkinase;    0.99970   1.00000   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   1.00000   1.00000     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99962   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99960   0.99947   0.99953     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 68/68
# 1071643/1071643 [==============================] 100.00% - 3912s - loss: 3.0536e-04 - acc: 0.9999 - val_loss: 8.6517e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000482357723589
# Test accuracy: 0.999926094846
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99992   0.99976   0.99984     37300
# PF00072.23;Response_reg;    0.99997   0.99993   0.99995     28684
#      PF00069.24;Pkinase;    0.99992   0.99977   0.99985     26514
#    PF02518.25;HATPase_c;    0.99992   1.00000   0.99996     23929
# PF00528.21;BPD_transp_1;    0.99962   0.99996   0.99979     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99973   0.99980   0.99977     15032

#              avg / total    0.99993   0.99993   0.99993    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 69/69
# 1071643/1071643 [==============================] 100.00% - 3915s - loss: 6.5657e-04 - acc: 0.9999 - val_loss: 8.8173e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 259s
# Test score: 0.000703188034744
# Test accuracy: 0.999895860936
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99965   1.00000   0.99983     37300
# PF00072.23;Response_reg;    0.99986   0.99993   0.99990     28684
#      PF00069.24;Pkinase;    0.99985   0.99970   0.99977     26514
#    PF02518.25;HATPase_c;    0.99996   1.00000   0.99998     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99962   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99960   0.99927   0.99943     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 70/70
# 1071643/1071643 [==============================] 100.00% - 3919s - loss: 3.6400e-04 - acc: 0.9999 - val_loss: 4.7619e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000594313245653
# Test accuracy: 0.999905938906
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99981   0.99995   0.99988     37300
# PF00072.23;Response_reg;    0.99983   0.99997   0.99990     28684
#      PF00069.24;Pkinase;    0.99966   1.00000   0.99983     26514
#    PF02518.25;HATPase_c;    0.99992   0.99996   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99966   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99996   0.99998     22846
#  PF00501.27;AMP-binding;    0.99993   0.99914   0.99953     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 71/71
# 1071643/1071643 [==============================] 100.00% - 3941s - loss: 5.5944e-04 - acc: 0.9999 - val_loss: 5.5558e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.00109174851402
# Test accuracy: 0.999801799884
# 297679/297679 [==============================] 100.00% - 221s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99986   0.99993     57079
#         PF00400.31;WD40;    0.99990   1.00000   0.99995     39257
#        PF07690.15;MFS_1;    0.99962   0.99995   0.99979     37300
# PF00072.23;Response_reg;    0.99927   0.99990   0.99958     28684
#      PF00069.24;Pkinase;    0.99977   0.99970   0.99974     26514
#    PF02518.25;HATPase_c;    1.00000   0.99979   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99975   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99982   1.00000   0.99991     22846
#  PF00501.27;AMP-binding;    0.99947   0.99820   0.99884     15032

#              avg / total    0.99980   0.99980   0.99980    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 72/72
# 1071643/1071643 [==============================] 100.00% - 3918s - loss: 4.0055e-04 - acc: 0.9999 - val_loss: 5.9654e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000719432932659
# Test accuracy: 0.999892501613
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99995   0.99997   0.99996     39257
#        PF07690.15;MFS_1;    0.99984   1.00000   0.99992     37300
# PF00072.23;Response_reg;    0.99993   0.99979   0.99986     28684
#      PF00069.24;Pkinase;    0.99932   0.99996   0.99964     26514
#    PF02518.25;HATPase_c;    1.00000   0.99992   0.99996     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99979   0.99989     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   0.99991   0.99993     22846
#  PF00501.27;AMP-binding;    0.99980   0.99920   0.99950     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 73/73
# 1071643/1071643 [==============================] 100.00% - 3996s - loss: 5.0643e-04 - acc: 0.9999 - val_loss: 8.3287e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 221s
# Test score: 0.000488376239018
# Test accuracy: 0.999905938906
# 297679/297679 [==============================] 100.00% - 224s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99995   0.99984   0.99989     37300
# PF00072.23;Response_reg;    0.99976   1.00000   0.99988     28684
#      PF00069.24;Pkinase;    0.99981   0.99996   0.99989     26514
#    PF02518.25;HATPase_c;    0.99996   1.00000   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99992   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99982   1.00000   0.99991     22846
#  PF00501.27;AMP-binding;    0.99973   0.99900   0.99937     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 74/74
# 1071643/1071643 [==============================] 100.00% - 3976s - loss: 4.2175e-04 - acc: 0.9999 - val_loss: 6.9391e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 226s
# Test score: 0.000627423968489
# Test accuracy: 0.999916016876
# 297679/297679 [==============================] 100.00% - 222s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99992   0.99997   0.99995     39257
#        PF07690.15;MFS_1;    0.99987   0.99995   0.99991     37300
# PF00072.23;Response_reg;    0.99990   0.99993   0.99991     28684
#      PF00069.24;Pkinase;    0.99970   0.99996   0.99983     26514
#    PF02518.25;HATPase_c;    0.99992   0.99996   0.99994     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99979   0.99989     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99980   0.99920   0.99950     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 75/75
# 1071643/1071643 [==============================] 100.00% - 3990s - loss: 6.7821e-04 - acc: 0.9999 - val_loss: 5.4335e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 223s
# Test score: 0.000520475608101
# Test accuracy: 0.999905938906
# 297679/297679 [==============================] 100.00% - 225s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99996   0.99997     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99973   0.99989   0.99981     37300
# PF00072.23;Response_reg;    0.99997   0.99993   0.99995     28684
#      PF00069.24;Pkinase;    0.99974   0.99996   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99966   0.99977     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99991   0.99996   0.99993     22846
#  PF00501.27;AMP-binding;    0.99987   0.99940   0.99963     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 76/76
# 1071643/1071643 [==============================] 100.00% - 3905s - loss: 6.4412e-04 - acc: 0.9999 - val_loss: 9.8579e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000701002611829
# Test accuracy: 0.99985890838
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99991   0.99995     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99968   1.00000   0.99984     37300
# PF00072.23;Response_reg;    0.99972   0.99997   0.99984     28684
#      PF00069.24;Pkinase;    0.99962   1.00000   0.99981     26514
#    PF02518.25;HATPase_c;    0.99987   1.00000   0.99994     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99970   0.99985     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    0.99987   1.00000   0.99993     22846
#  PF00501.27;AMP-binding;    0.99980   0.99807   0.99893     15032

#              avg / total    0.99986   0.99986   0.99986    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 77/77
# 1071643/1071643 [==============================] 100.00% - 3894s - loss: 4.8870e-04 - acc: 0.9999 - val_loss: 0.0012 - val_acc: 0.9997
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.00112152491435
# Test accuracy: 0.999774925298
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    1.00000   0.99869   0.99934     37300
# PF00072.23;Response_reg;    0.99993   0.99990   0.99991     28684
#      PF00069.24;Pkinase;    0.99951   1.00000   0.99975     26514
#    PF02518.25;HATPase_c;    0.99996   0.99983   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99835   0.99996   0.99915     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99940   0.99953   0.99947     15032

#              avg / total    0.99978   0.99977   0.99977    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 78/78
# 1071643/1071643 [==============================] 100.00% - 3896s - loss: 4.3570e-04 - acc: 0.9999 - val_loss: 4.9940e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000383342696616
# Test accuracy: 0.999926094846
# 297679/297679 [==============================] 100.00% - 222s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    1.00000   1.00000   1.00000     39257
#        PF07690.15;MFS_1;    0.99976   0.99997   0.99987     37300
# PF00072.23;Response_reg;    0.99979   0.99997   0.99988     28684
#      PF00069.24;Pkinase;    0.99981   0.99996   0.99989     26514
#    PF02518.25;HATPase_c;    1.00000   1.00000   1.00000     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99987   0.99992     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    1.00000   0.99900   0.99950     15032

#              avg / total    0.99993   0.99993   0.99993    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 79/79
# 1071643/1071643 [==============================] 100.00% - 3943s - loss: 4.2342e-04 - acc: 0.9999 - val_loss: 0.0014 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.0012349544541
# Test accuracy: 0.99983875244
# 297679/297679 [==============================] 100.00% - 223s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99974   1.00000   0.99987     57079
#         PF00400.31;WD40;    0.99997   0.99992   0.99995     39257
#        PF07690.15;MFS_1;    0.99987   0.99987   0.99987     37300
# PF00072.23;Response_reg;    0.99997   0.99927   0.99962     28684
#      PF00069.24;Pkinase;    0.99981   0.99989   0.99985     26514
#    PF02518.25;HATPase_c;    0.99983   0.99987   0.99985     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99970   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99907   0.99960   0.99933     15032

#              avg / total    0.99984   0.99984   0.99984    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 80/80
# 1071643/1071643 [==============================] 100.00% - 3937s - loss: 4.6826e-04 - acc: 0.9999 - val_loss: 3.7526e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 219s
# Test score: 0.000367848407922
# Test accuracy: 0.999926094846
# 297679/297679 [==============================] 100.00% - 224s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99989   0.99987   0.99988     37300
# PF00072.23;Response_reg;    0.99997   0.99993   0.99995     28684
#      PF00069.24;Pkinase;    0.99989   0.99989   0.99989     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99975   0.99983   0.99979     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99960   0.99967   0.99963     15032

#              avg / total    0.99993   0.99993   0.99993    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 81/81
# 1071643/1071643 [==============================] 100.00% - 3931s - loss: 3.8198e-04 - acc: 0.9999 - val_loss: 3.6210e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.00049300603301
# Test accuracy: 0.999922735523
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99998   0.99999     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99973   0.99997   0.99985     37300
# PF00072.23;Response_reg;    0.99993   0.99993   0.99993     28684
#      PF00069.24;Pkinase;    0.99977   1.00000   0.99989     26514
#    PF02518.25;HATPase_c;    0.99992   1.00000   0.99996     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99966   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99993   0.99927   0.99960     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 82/82
# 1071643/1071643 [==============================] 100.00% - 3915s - loss: 3.7757e-04 - acc: 0.9999 - val_loss: 8.2413e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000787324864048
# Test accuracy: 0.999875704997
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99990   0.99997   0.99994     39257
#        PF07690.15;MFS_1;    0.99960   1.00000   0.99980     37300
# PF00072.23;Response_reg;    0.99979   0.99990   0.99984     28684
#      PF00069.24;Pkinase;    0.99985   0.99992   0.99989     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99958   0.99979     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99987   0.99880   0.99933     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 83/83
# 1071643/1071643 [==============================] 100.00% - 3980s - loss: 4.4175e-04 - acc: 0.9999 - val_loss: 6.8601e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000680670878381
# Test accuracy: 0.999875704997
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99985   0.99997   0.99991     39257
#        PF07690.15;MFS_1;    0.99973   0.99995   0.99984     37300
# PF00072.23;Response_reg;    0.99990   0.99986   0.99988     28684
#      PF00069.24;Pkinase;    0.99974   0.99981   0.99977     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99966   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   0.99996   0.99996     22846
#  PF00501.27;AMP-binding;    0.99940   0.99933   0.99937     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 84/84
# 1071643/1071643 [==============================] 100.00% - 3962s - loss: 3.9513e-04 - acc: 0.9999 - val_loss: 8.0557e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000496565880903
# Test accuracy: 0.999916016876
# 297679/297679 [==============================] 100.00% - 218s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   1.00000   1.00000     57079
#         PF00400.31;WD40;    0.99990   0.99997   0.99994     39257
#        PF07690.15;MFS_1;    0.99984   0.99997   0.99991     37300
# PF00072.23;Response_reg;    0.99972   0.99993   0.99983     28684
#      PF00069.24;Pkinase;    0.99996   0.99989   0.99992     26514
#    PF02518.25;HATPase_c;    1.00000   0.99987   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99996   0.99970   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99973   0.99947   0.99960     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 85/85
# 1071643/1071643 [==============================] 100.00% - 3930s - loss: 5.0943e-04 - acc: 0.9999 - val_loss: 9.2638e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 216s
# Test score: 0.000774476878857
# Test accuracy: 0.99989922026
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99995   0.99997     57079
#         PF00400.31;WD40;    0.99995   0.99997   0.99996     39257
#        PF07690.15;MFS_1;    0.99979   1.00000   0.99989     37300
# PF00072.23;Response_reg;    0.99990   0.99979   0.99984     28684
#      PF00069.24;Pkinase;    0.99992   0.99970   0.99981     26514
#    PF02518.25;HATPase_c;    0.99987   0.99996   0.99992     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99975   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99920   0.99967   0.99943     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 86/86
# 1071643/1071643 [==============================] 100.00% - 3977s - loss: 5.1849e-04 - acc: 0.9999 - val_loss: 9.7915e-04 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.000729163916906
# Test accuracy: 0.999892501613
# 297679/297679 [==============================] 100.00% - 221s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99987   1.00000   0.99994     39257
#        PF07690.15;MFS_1;    0.99984   0.99989   0.99987     37300
# PF00072.23;Response_reg;    1.00000   0.99979   0.99990     28684
#      PF00069.24;Pkinase;    0.99977   1.00000   0.99989     26514
#    PF02518.25;HATPase_c;    0.99996   0.99992   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99979   0.99987   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99987   0.99994     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99947   0.99920   0.99933     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 87/87
# 1071643/1071643 [==============================] 100.00% - 3988s - loss: 5.5683e-04 - acc: 0.9999 - val_loss: 4.3612e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 222s
# Test score: 0.000598262810204
# Test accuracy: 0.999895860936
# 297679/297679 [==============================] 100.00% - 223s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99992   1.00000   0.99996     39257
#        PF07690.15;MFS_1;    0.99976   0.99992   0.99984     37300
# PF00072.23;Response_reg;    0.99972   0.99993   0.99983     28684
#      PF00069.24;Pkinase;    0.99985   0.99996   0.99991     26514
#    PF02518.25;HATPase_c;    1.00000   0.99992   0.99996     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99983   0.99985     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    0.99987   0.99880   0.99933     15032

#              avg / total    0.99990   0.99990   0.99990    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 88/88
# 1071643/1071643 [==============================] 100.00% - 3935s - loss: 3.9242e-04 - acc: 0.9999 - val_loss: 0.0010 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.00155727773308
# Test accuracy: 0.999764847328
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99995   0.99996     57079
#         PF00400.31;WD40;    0.99995   0.99990   0.99992     39257
#        PF07690.15;MFS_1;    0.99976   0.99997   0.99987     37300
# PF00072.23;Response_reg;    0.99868   1.00000   0.99934     28684
#      PF00069.24;Pkinase;    0.99955   0.99996   0.99975     26514
#    PF02518.25;HATPase_c;    0.99967   0.99992   0.99979     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99975   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   0.99996   0.99998     22846
#  PF00501.27;AMP-binding;    1.00000   0.99654   0.99827     15032

#              avg / total    0.99977   0.99976   0.99976    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 89/89
# 1071643/1071643 [==============================] 100.00% - 3905s - loss: 5.5385e-04 - acc: 0.9999 - val_loss: 0.0020 - val_acc: 0.9996
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.00154719751445
# Test accuracy: 0.999734613418
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99995   0.99998   0.99996     57079
#         PF00400.31;WD40;    0.99987   1.00000   0.99994     39257
#        PF07690.15;MFS_1;    1.00000   0.99863   0.99932     37300
# PF00072.23;Response_reg;    0.99965   0.99997   0.99981     28684
#      PF00069.24;Pkinase;    0.99974   0.99996   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   0.99987   0.99994     23929
# PF00528.21;BPD_transp_1;    0.99785   1.00000   0.99892     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99980   0.99854   0.99917     15032

#              avg / total    0.99973   0.99973   0.99973    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 90/90
# 1071643/1071643 [==============================] 100.00% - 4010s - loss: 4.5054e-04 - acc: 0.9999 - val_loss: 0.0013 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 227s
# Test score: 0.000750550429308
# Test accuracy: 0.999892501613
# 297679/297679 [==============================] 100.00% - 228s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99998   0.99997     57079
#         PF00400.31;WD40;    1.00000   0.99990   0.99995     39257
#        PF07690.15;MFS_1;    0.99989   0.99997   0.99993     37300
# PF00072.23;Response_reg;    0.99986   0.99990   0.99988     28684
#      PF00069.24;Pkinase;    0.99992   0.99974   0.99983     26514
#    PF02518.25;HATPase_c;    1.00000   0.99992   0.99996     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99992   0.99992     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99948   1.00000   0.99974     22846
#  PF00501.27;AMP-binding;    0.99960   0.99920   0.99940     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 91/91
# 1071643/1071643 [==============================] 100.00% - 3915s - loss: 4.0420e-04 - acc: 0.9999 - val_loss: 5.4796e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000534617548248
# Test accuracy: 0.999905938906
# 297679/297679 [==============================] 100.00% - 220s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99993   1.00000   0.99996     57079
#         PF00400.31;WD40;    0.99990   1.00000   0.99995     39257
#        PF07690.15;MFS_1;    0.99989   0.99992   0.99991     37300
# PF00072.23;Response_reg;    1.00000   0.99962   0.99981     28684
#      PF00069.24;Pkinase;    0.99992   0.99989   0.99991     26514
#    PF02518.25;HATPase_c;    0.99996   1.00000   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99992   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99978   1.00000   0.99989     22846
#  PF00501.27;AMP-binding;    0.99973   0.99940   0.99957     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 92/92
# 1071643/1071643 [==============================] 100.00% - 3934s - loss: 4.7738e-04 - acc: 0.9999 - val_loss: 8.3646e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 218s
# Test score: 0.000762481189014
# Test accuracy: 0.99987906432
# 297679/297679 [==============================] 100.00% - 219s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99995   0.99996   0.99996     57079
#         PF00400.31;WD40;    0.99995   1.00000   0.99997     39257
#        PF07690.15;MFS_1;    0.99981   0.99992   0.99987     37300
# PF00072.23;Response_reg;    0.99993   0.99976   0.99984     28684
#      PF00069.24;Pkinase;    0.99974   0.99992   0.99983     26514
#    PF02518.25;HATPase_c;    0.99979   1.00000   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99987   0.99985     23644
#      PF00096.25;zf-C2H2;    1.00000   0.99996   0.99998     23394
#         PF12796.6;Ank_2;    0.99991   1.00000   0.99996     22846
#  PF00501.27;AMP-binding;    0.99973   0.99880   0.99927     15032

#              avg / total    0.99988   0.99988   0.99988    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 93/93
#  623360/1071643 [================>.............]  58.17% - ETA: 1647s - loss: 5.6879e-04 - acc: 0.9999
# 1071643/1071643 [==============================] 100.00% - 3602s - loss: 7.1314e-04 - acc: 0.9999 - val_loss: 6.2498e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 220s
# Test score: 0.000532608797944
# Test accuracy: 0.999916016876
# 297679/297679 [==============================] 100.00% - 213s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99998   0.99998   0.99998     57079
#         PF00400.31;WD40;    0.99997   0.99995   0.99996     39257
#        PF07690.15;MFS_1;    0.99981   0.99995   0.99988     37300
# PF00072.23;Response_reg;    0.99986   0.99986   0.99986     28684
#      PF00069.24;Pkinase;    0.99989   0.99996   0.99992     26514
#    PF02518.25;HATPase_c;    0.99996   0.99987   0.99992     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99983   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99973   0.99947   0.99960     15032

#              avg / total    0.99992   0.99992   0.99992    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 94/94
# 1071643/1071643 [==============================] 100.00% - 3635s - loss: 5.0171e-04 - acc: 0.9999 - val_loss: 8.9196e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 220s
# Test score: 0.000931838499162
# Test accuracy: 0.999892501613
# 297679/297679 [==============================] 100.00% - 221s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    1.00000   1.00000   1.00000     39257
#        PF07690.15;MFS_1;    0.99987   0.99968   0.99977     37300
# PF00072.23;Response_reg;    0.99972   0.99997   0.99984     28684
#      PF00069.24;Pkinase;    0.99989   0.99996   0.99992     26514
#    PF02518.25;HATPase_c;    0.99996   1.00000   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99949   0.99979   0.99964     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99980   0.99927   0.99953     15032

#              avg / total    0.99989   0.99989   0.99989    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 95/95
# 1071643/1071643 [==============================] 100.00% - 3655s - loss: 3.5910e-04 - acc: 0.9999 - val_loss: 9.2676e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 217s
# Test score: 0.00046247043852
# Test accuracy: 0.999912657553
# 297679/297679 [==============================] 100.00% - 216s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99998   0.99997     57079
#         PF00400.31;WD40;    0.99997   0.99997   0.99997     39257
#        PF07690.15;MFS_1;    0.99981   0.99992   0.99987     37300
# PF00072.23;Response_reg;    0.99990   0.99990   0.99990     28684
#      PF00069.24;Pkinase;    0.99970   1.00000   0.99985     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99987   0.99979   0.99983     23644
#      PF00096.25;zf-C2H2;    0.99996   1.00000   0.99998     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99993   0.99920   0.99957     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 96/96
# 1071643/1071643 [==============================] 100.00% - 3657s - loss: 5.1080e-04 - acc: 0.9999 - val_loss: 0.0042 - val_acc: 0.9995
# 297679/297679 [==============================] 100.00% - 210s
# Test score: 0.00371749338539
# Test accuracy: 0.999556569283
# 297679/297679 [==============================] 100.00% - 208s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99993   0.99996     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99786   1.00000   0.99893     37300
# PF00072.23;Response_reg;    0.99854   0.99997   0.99925     28684
#      PF00069.24;Pkinase;    0.99974   1.00000   0.99987     26514
#    PF02518.25;HATPase_c;    0.99996   0.99996   0.99996     23929
# PF00528.21;BPD_transp_1;    1.00000   0.99924   0.99962     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99996   1.00000   0.99998     22846
#  PF00501.27;AMP-binding;    1.00000   0.99282   0.99639     15032

#              avg / total    0.99956   0.99956   0.99956    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 97/97
# 1071643/1071643 [==============================] 100.00% - 3626s - loss: 5.6623e-04 - acc: 0.9999 - val_loss: 0.0011 - val_acc: 0.9998
# 297679/297679 [==============================] 100.00% - 212s
# Test score: 0.00128235091142
# Test accuracy: 0.99984883041
# 297679/297679 [==============================] 100.00% - 205s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99991   0.99996     57079
#         PF00400.31;WD40;    0.99997   0.99992   0.99995     39257
#        PF07690.15;MFS_1;    0.99968   0.99997   0.99983     37300
# PF00072.23;Response_reg;    0.99941   0.99997   0.99969     28684
#      PF00069.24;Pkinase;    0.99985   0.99981   0.99983     26514
#    PF02518.25;HATPase_c;    0.99996   0.99983   0.99990     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99970   0.99981     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    0.99987   1.00000   0.99993     22846
#  PF00501.27;AMP-binding;    0.99967   0.99874   0.99920     15032

#              avg / total    0.99985   0.99985   0.99985    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 98/98
# 1071643/1071643 [==============================] 100.00% - 3615s - loss: 4.3106e-04 - acc: 0.9999 - val_loss: 5.1984e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 215s
# Test score: 0.00061013521595
# Test accuracy: 0.999912657553
# 297679/297679 [==============================] 100.00% - 212s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    0.99996   0.99998   0.99997     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99984   0.99992   0.99988     37300
# PF00072.23;Response_reg;    0.99983   0.99990   0.99986     28684
#      PF00069.24;Pkinase;    0.99974   0.99996   0.99985     26514
#    PF02518.25;HATPase_c;    0.99996   1.00000   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99992   0.99983   0.99987     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99987   0.99907   0.99947     15032

#              avg / total    0.99991   0.99991   0.99991    297679

# Train on 1071643 samples, validate on 119072 samples
# Epoch 99/99
# 1071643/1071643 [==============================] 100.00% - 3620s - loss: 4.0979e-04 - acc: 0.9999 - val_loss: 4.7885e-04 - val_acc: 0.9999
# 297679/297679 [==============================] 100.00% - 210s
# Test score: 0.00042304649045
# Test accuracy: 0.999929454169
# 297679/297679 [==============================] 100.00% - 210s
#                           precision    recall  f1-score   support

#     PF00005.26;ABC_tran;    1.00000   0.99996   0.99998     57079
#         PF00400.31;WD40;    0.99997   1.00000   0.99999     39257
#        PF07690.15;MFS_1;    0.99989   0.99987   0.99988     37300
# PF00072.23;Response_reg;    0.99986   0.99997   0.99991     28684
#      PF00069.24;Pkinase;    0.99989   0.99996   0.99992     26514
#    PF02518.25;HATPase_c;    1.00000   0.99996   0.99998     23929
# PF00528.21;BPD_transp_1;    0.99983   0.99983   0.99983     23644
#      PF00096.25;zf-C2H2;    1.00000   1.00000   1.00000     23394
#         PF12796.6;Ank_2;    1.00000   1.00000   1.00000     22846
#  PF00501.27;AMP-binding;    0.99967   0.99953   0.99960     15032

#              avg / total    0.99993   0.99993   0.99993    297679