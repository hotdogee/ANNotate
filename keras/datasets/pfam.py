# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from six.moves import zip
import os
import gzip
import json
import errno
import struct
import warnings
import numpy as np
from collections import defaultdict

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, TensorBoard
from keras.utils import plot_model, get_file, Progbar


def _gzip_size(filename):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _pfam_fa_to_domain_dict(fa_path):
    """Parse a Pfam-A.fasta.gz file into domain_dict[domain] = [(seq_id, sequence), ...]
    """
    domain_dict = defaultdict(list)
    seq_id, domain, sequence = '', '', ''
    target = _gzip_size(fa_path)
    while target < os.path.getsize(fa_path):
        # the uncompressed size can't be smaller than the compressed size, so add 4GB
        target += 2**32
    prog = Progbar(target)
    current = 0
    with gzip.open(fa_path, 'r') as fa_f:
        for line in fa_f:
            # if (int(fa_f.tell()/target*100) > current):
            #     current = int(fa_f.tell()/target*100)
            #     print('{}/{} ({:.2f}%)'.format(fa_f.tell(), target, current))
            if target < fa_f.tell():
                target += 2**32
            prog.update(fa_f.tell())
            line = line.strip().decode('utf-8')
            if len(line) > 0 and line[0] == '>':
                if sequence:
                    domain_dict[domain].append((seq_id, sequence))
                    seq_id, domain, sequence = '', '', ''
                # parse header
                seq_id = line.split()[0]
                domain = line.split()[2]
            else:
                sequence += line
        prog.update(fa_f.tell(), force=True)
    return domain_dict

aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

def load_data(fname='Pfam-A.fasta.gz',
              origin_base='ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam31.0/',
              num_domain=10,
              test_split=0.2,
              seed=113,
              index_from=1,
              cache_subdir='datasets',
              cache_dir=None,
              **kwargs):
    """Loads the Pfam classification dataset.

    # Arguments
        fname: name of the file to download (relative to origin_base).
        origin_base: base URL download location.
        num_domain: max number of domains to include. Domains are
            ranked by how many sequences they have.
        test_split: Fraction of the dataset to be used as test data.
        seed: random seed for sample shuffling.
        index_from: index amino acids with this index and higher.
            Set to 1 because 0 is usually the padding character.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    path = get_file(fname, origin_base + fname,
                    cache_subdir=cache_subdir, cache_dir=cache_dir)

    # check cache
    cache_path = '{0}-d{1}.npz'.format(os.path.splitext(
        os.path.splitext(path)[0])[0], num_domain)
    if os.path.exists(cache_path):
        print('from {0}'.format(cache_path))
        f = np.load(cache_path)
        domain_list = f['domain_list']
        domains = f['domains']
        sequences = f['sequences']
    else:
        aa_index = dict(
            zip(aa_list, range(index_from, index_from + len(aa_list))))
        domain_dict = _pfam_fa_to_domain_dict(path)
        domain_list = []
        domains = []
        sequences = []
        di = 0
        for dom in sorted(domain_dict, key=lambda k: (len(domain_dict[k]), k), reverse=True):
            domain_list.append(dom)
            for seq_id, seq in domain_dict[dom]:
                domains.append(di)
                try:
                    sequences.append([aa_index[a] for a in seq])
                except KeyError as e:
                    print('{0} parsing {1}: {2}'.format(e, seq_id, seq))
                    raise e
            di += 1
            if num_domain and di >= num_domain:
                break

        # save cache
        np.savez(cache_path, domain_list=domain_list,
                 domains=domains, sequences=sequences)

    np.random.seed(seed)
    np.random.shuffle(domains)
    np.random.seed(seed)
    np.random.shuffle(sequences)

    x_train = np.array(sequences[:int(len(sequences) * (1 - test_split))])
    y_train = np.array(domains[:int(len(sequences) * (1 - test_split))])

    x_test = np.array(sequences[int(len(sequences) * (1 - test_split)):])
    y_test = np.array(domains[int(len(sequences) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test), domain_list


def dir_check(dr):
    if not os.path.exists(dr):
        try:
            os.makedirs(dr)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def train(model, model_name, file_name, num_domain, epoch_start=0, save_freq=1, 
          test_split=0.2, validation_split=0.1, maxlen=512, batch_size=256, 
          epochs=150, device='/gpu:0', csvlog_dir='./csvlogs',
          model_dir='./models', confusion_matrix_dir='./confusion_matrix',
          classification_report_dir='./classification_report', tensorboard_logs_dir='./tfevents'):
    trainer_name = 'cl{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}'.format(
        num_domain, test_split * 100, validation_split * 100, maxlen, batch_size)
    
    # make sure directories we're going to write into exist
    dir_check(csvlog_dir)
    dir_check(model_dir)
    dir_check(confusion_matrix_dir)
    dir_check(classification_report_dir)
    dir_check(tensorboard_logs_dir)

    csvlog_path = '{}.{}.{}.csv'.format(
        os.path.abspath(os.path.join(csvlog_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    tensorboard_logs_path = '{}.{}.{}/'.format(
        os.path.abspath(os.path.join(tensorboard_logs_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    with tf.device(device):
        print('Loading data...')
        (x_train, y_train_class), (x_test, y_test_class), domain_list = load_data(
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

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        for epoch in range(epoch_start, epochs):
            model_path = '{}.e{}.{}.{}.h5'.format(
                os.path.abspath(os.path.join(model_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if os.path.exists(model_path):
                print('Loading model:', os.path.basename(model_path))
                model = load_model(model_path)
            else:
                history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch+1,
                    verbose=1,
                    validation_split=validation_split,
                    callbacks=[CSVLogger(csvlog_path, append=True), 
                               TensorBoard(log_dir=tensorboard_logs_path, histogram_freq=0, write_grads=True, write_images=True, embeddings_freq=1)],
                    initial_epoch=epoch)

                # save model
                if epoch % save_freq == 0:
                    model.save(model_path)
            # score = model.evaluate(x_test, y_test,
            #                     batch_size=batch_size, verbose=1)
            # print('Test score:', score[0])
            # print('Test accuracy:', score[1])
            y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
            y_pred_class = np.argmax(y_pred, axis=1)
            del y_pred
            print('Test accuracy:', accuracy_score(y_test_class, y_pred_class))
            classification_report_txt = classification_report(y_test_class, y_pred_class, target_names=domain_list, digits=5)
            if len(y_pred_class) <= 15:
                print(classification_report_txt)

            # save classification_report
            classification_report_path = '{}.e{}.{}.{}.cr.txt'.format(
                os.path.abspath(os.path.join(classification_report_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(classification_report_path):
                np.savetxt(classification_report_path, [classification_report_txt], fmt='%s')
            else:
                print('Classification Report File Exists: {}'.format(classification_report_path))

            # save confusion_matrix
            confusion_matrix_path = '{}.e{}.{}.{}.cm.csv'.format(
                os.path.abspath(os.path.join(confusion_matrix_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(confusion_matrix_path):
                np.savetxt(confusion_matrix_path, confusion_matrix(y_test_class, y_pred_class), fmt='%d', delimiter=",")
            else:
                print('Confusion Matrix File Exists: {}'.format(confusion_matrix_path))


def sparse_train(model, model_name, file_name, num_domain, epoch_start=0, save_freq=1, 
          test_split=0.2, validation_split=0.1, maxlen=512, batch_size=256, 
          epochs=150, device='/gpu:0', csvlog_dir='./csvlogs',
          model_dir='./models', confusion_matrix_dir='./confusion_matrix',
          classification_report_dir='./classification_report', tensorboard_logs_dir='./tfevents'):
    trainer_name = 'cs{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}'.format(
        num_domain, test_split * 100, validation_split * 100, maxlen, batch_size)
    
    # make sure directories we're going to write into exist
    dir_check(csvlog_dir)
    dir_check(model_dir)
    dir_check(confusion_matrix_dir)
    dir_check(classification_report_dir)
    dir_check(tensorboard_logs_dir)

    csvlog_path = '{}.{}.{}.csv'.format(
        os.path.abspath(os.path.join(csvlog_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    tensorboard_logs_path = '{}.{}.{}/'.format(
        os.path.abspath(os.path.join(tensorboard_logs_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    with tf.device(device):
        print('Loading data...')
        (x_train, y_train_class), (x_test, y_test_class), domain_list = load_data(
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
        print('y_train shape:', y_train_class.shape)
        print('y_test shape:', y_test_class.shape) # (297679, )

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        for epoch in range(epoch_start, epochs):
            model_path = '{}.e{}.{}.{}.h5'.format(
                os.path.abspath(os.path.join(model_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if os.path.exists(model_path):
                print('Loading model:', os.path.basename(model_path))
                model = load_model(model_path)
            else:
                history = model.fit(x_train, y_train_class,
                    batch_size=batch_size,
                    epochs=epoch+1,
                    verbose=1,
                    validation_split=validation_split,
                    callbacks=[CSVLogger(csvlog_path, append=True), 
                               TensorBoard(log_dir=tensorboard_logs_path, histogram_freq=0, write_grads=True, write_images=True, embeddings_freq=1)],
                    initial_epoch=epoch)

                # save model
                if epoch % save_freq == 0:
                    model.save(model_path)
            # score = model.evaluate(x_test, y_test,
            #                     batch_size=batch_size, verbose=1)
            # print('Test score:', score[0])
            # print('Test accuracy:', score[1])
            y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
            # print('y_pred shape:', y_pred.shape) # (297679, 10)
            y_pred_class = np.argmax(y_pred, axis=1)
            del y_pred
            print('Test accuracy:', accuracy_score(y_test_class, y_pred_class))
            classification_report_txt = classification_report(y_test_class, y_pred_class, target_names=domain_list, digits=5)
            if len(y_pred_class) <= 15:
                print(classification_report_txt)

            # save classification_report
            classification_report_path = '{}.e{}.{}.{}.cr.txt'.format(
                os.path.abspath(os.path.join(classification_report_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(classification_report_path):
                np.savetxt(classification_report_path, [classification_report_txt], fmt='%s')
            else:
                print('Classification Report File Exists: {}'.format(classification_report_path))

            # save confusion_matrix
            confusion_matrix_path = '{}.e{}.{}.{}.cm.csv'.format(
                os.path.abspath(os.path.join(confusion_matrix_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(confusion_matrix_path):
                np.savetxt(confusion_matrix_path, confusion_matrix(y_test_class, y_pred_class), fmt='%d', delimiter=",")
            else:
                print('Confusion Matrix File Exists: {}'.format(confusion_matrix_path))
                

def sparse_train_vs(model, model_name, file_name, num_domain, epoch_start=0, save_freq=1, 
          test_split=0.2, validation_split=0.1, maxlen=512, batch_size=256, 
          epochs=150, device='/gpu:0', csvlog_dir='./csvlogs',
          model_dir='./models', confusion_matrix_dir='./confusion_matrix',
          classification_report_dir='./classification_report', tensorboard_logs_dir='./tfevents',
          acc_list_file='./vs/acc_list.csv'):
    trainer_name = 'cs{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}'.format(
        num_domain, test_split * 100, validation_split * 1000, maxlen, batch_size)
    
    # make sure directories we're going to write into exist
    dir_check(csvlog_dir)
    dir_check(model_dir)
    dir_check(confusion_matrix_dir)
    dir_check(classification_report_dir)
    dir_check(tensorboard_logs_dir)

    csvlog_path = '{}.{}.{}.csv'.format(
        os.path.abspath(os.path.join(csvlog_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    tensorboard_logs_path = '{}.{}.{}/'.format(
        os.path.abspath(os.path.join(tensorboard_logs_dir, os.path.splitext(os.path.basename(file_name))[0])), 
        trainer_name, model_name)

    with tf.device(device):
        print('Loading data...')
        (x_train, y_train_class), (x_test, y_test_class), domain_list = load_data(
            num_domain=num_domain, test_split=test_split)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        # print(domain_list)

        num_classes = np.max(y_train_class) + 1
        print(num_classes, 'classes')

        # calculation total number of sequences we need
        total_seq = int(len(x_train) * (1. - validation_split) / 0.9) + 1
        x_train = x_train[:total_seq]
        y_train_class = y_train_class[:total_seq]

        print('Pad sequences')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_train shape:', y_train_class.shape)
        print('y_test shape:', y_test_class.shape) # (297679, )

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        acc_list = [str(validation_split)]
        for epoch in range(epoch_start, epochs):
            model_path = '{}.e{}.{}.{}.h5'.format(
                os.path.abspath(os.path.join(model_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if os.path.exists(model_path):
                print('Loading model:', os.path.basename(model_path))
                model = load_model(model_path)
            else:
                history = model.fit(x_train, y_train_class,
                    batch_size=batch_size,
                    epochs=epoch+1,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[CSVLogger(csvlog_path, append=True), 
                               TensorBoard(log_dir=tensorboard_logs_path, histogram_freq=0, write_grads=True, write_images=True, embeddings_freq=1)],
                    initial_epoch=epoch)

                # save model
                if epoch % save_freq == 0:
                    model.save(model_path)
            # score = model.evaluate(x_test, y_test,
            #                     batch_size=batch_size, verbose=1)
            # print('Test score:', score[0])
            # print('Test accuracy:', score[1])
            y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
            # print('y_pred shape:', y_pred.shape) # (297679, 10)
            y_pred_class = np.argmax(y_pred, axis=1)
            del y_pred
            print('Test accuracy:', accuracy_score(y_test_class, y_pred_class))
            acc_list.append(str(accuracy_score(y_test_class, y_pred_class)))
            classification_report_txt = classification_report(y_test_class, y_pred_class, target_names=domain_list, digits=5)
            if len(y_pred_class) <= 15:
                print(classification_report_txt)

            # save classification_report
            classification_report_path = '{}.e{}.{}.{}.cr.txt'.format(
                os.path.abspath(os.path.join(classification_report_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(classification_report_path):
                np.savetxt(classification_report_path, [classification_report_txt], fmt='%s')
            else:
                print('Classification Report File Exists: {}'.format(classification_report_path))

            # save confusion_matrix
            confusion_matrix_path = '{}.e{}.{}.{}.cm.csv'.format(
                os.path.abspath(os.path.join(confusion_matrix_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if not os.path.exists(confusion_matrix_path):
                np.savetxt(confusion_matrix_path, confusion_matrix(y_test_class, y_pred_class), fmt='%d', delimiter=",")
            else:
                print('Confusion Matrix File Exists: {}'.format(confusion_matrix_path))
                
        with open(acc_list_file, 'a') as f:
            f.write(','.join(acc_list) + '\n')
