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
from keras.utils import GeneratorEnqueuer


def _gzip_size(filename):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _fa_gz_to_dict(fa_path):
    """Parse a FASTA.gz file into fa_dict[seq_id] = sequence
    """
    print('Parsing {}'.format(os.path.basename(fa_path)))
    fa_dict = {}
    seq_id, sequence = '', ''
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
                prog.target = target
            prog.update(fa_f.tell())
            line = line.strip().decode('utf-8')
            if len(line) > 0 and line[0] == '>':
                if sequence:
                    fa_dict[seq_id] = sequence
                    seq_id, sequence = '', ''
                # parse header
                seq_id = line.split()[0][1:]
            else:
                sequence += line
        if sequence:
            fa_dict[seq_id] = sequence
        prog.update(fa_f.tell(), force=True)
    return fa_dict


def _pfam_regions_tsv_gz_to_dict(tsv_path):
    """Parse a Pfam-A.regions.uniprot.tsv.gz file into 
    domain_regions_dict[pfamA_acc] = [(uniprot_acc + '.' + seq_version, seq_start, seq_end), ...]
    """
    print('Parsing {}'.format(os.path.basename(tsv_path)))
    domain_regions_dict = defaultdict(list)
    target = _gzip_size(tsv_path)
    while target < os.path.getsize(tsv_path):
        # the uncompressed size can't be smaller than the compressed size, so add 4GB
        target += 2**32
    prog = Progbar(target)
    # current = 0
    line_num = 0
    with gzip.open(tsv_path, 'r') as tsv_f:
        for line in tsv_f:
            if target < tsv_f.tell():
                target += 2**32
                prog.target = target
            prog.update(tsv_f.tell())
            line_num += 1
            if line_num == 1: continue # skip header
            tokens = line.strip().decode('utf-8').split()
            seq_id = '{}.{}'.format(tokens[0], tokens[1])
            domain_regions_dict[tokens[4]].append((seq_id, int(tokens[5]), int(tokens[6])))
        prog.update(tsv_f.tell(), force=True)
    return domain_regions_dict

aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

def load_data(uniprot_file='uniprot.gz',
              regions_file='Pfam-A.regions.uniprot.tsv.gz',
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
        uniprot_file: name of the uniprot file to download (relative to origin_base).
        regions_file: name of the regions file to download (relative to origin_base).
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

    uniprot_path = get_file(uniprot_file, origin_base + uniprot_file,
                    cache_subdir=cache_subdir, cache_dir=cache_dir)
    regions_path = get_file(regions_file, origin_base + regions_file,
                    cache_subdir=cache_subdir, cache_dir=cache_dir)

    # check cache
    seq_dom_split_cache_path = '{0}-d{1}-s{2}.npz'.format(os.path.splitext(
        os.path.splitext(regions_path)[0])[0], num_domain, test_split * 100)
    if os.path.exists(seq_dom_split_cache_path):
        print('from {0}'.format(seq_dom_split_cache_path))
        f = np.load(seq_dom_split_cache_path)
        x_train = f['x_train']
        y_train = f['y_train']
        maxlen_train = f['maxlen_train'].tolist()
        x_test = f['x_test']
        y_test = f['y_test']
        maxlen_test = f['maxlen_test'].tolist()
        domain_list = f['domain_list']
    else:
        seq_dom_cache_path = '{0}-d{1}.npz'.format(os.path.splitext(
            os.path.splitext(regions_path)[0])[0], num_domain)
        if os.path.exists(seq_dom_cache_path):
            print('from {0}'.format(seq_dom_cache_path))
            f = np.load(seq_dom_cache_path)
            domain_list = f['domain_list']
            domains = f['domains']
            sequences = f['sequences']
        else:
            # seq_dict[seq_id] = sequence
            seq_dict_cache_path = '{0}.json'.format(os.path.splitext(uniprot_path)[0])
            if os.path.exists(seq_dict_cache_path):
                print('from {0}'.format(seq_dict_cache_path))
                with open(seq_dict_cache_path, 'r') as f:
                    seq_dict = json.load(f)
            else:
                seq_dict = _fa_gz_to_dict(uniprot_path)
                with open(seq_dict_cache_path, 'w') as f:
                    json.dump(seq_dict, f)

            domain_regions_dict_cache_path = '{0}.json'.format(os.path.splitext(regions_path)[0])
            if os.path.exists(domain_regions_dict_cache_path):
                print('from {0}'.format(domain_regions_dict_cache_path))
                with open(domain_regions_dict_cache_path, 'r') as f:
                    domain_regions_dict = json.load(f)
            else:
                domain_regions_dict = _pfam_regions_tsv_gz_to_dict(regions_path)
                with open(domain_regions_dict_cache_path, 'w') as f:
                    json.dump(domain_regions_dict, f)
                
            print('seq_dict[{}]'.format(len(seq_dict)))
            print('domain_regions_dict[{}]'.format(len(domain_regions_dict)))

            domain_list = []
            # build seq_regions_dict[seq_id] = [(pfamA_acc, seq_start, seq_end), ...]
            seq_regions_dict = defaultdict(list)
            # domains with the most sequences first
            for pfamA_acc in sorted(domain_regions_dict, key=lambda k: (len(domain_regions_dict[k]), k), reverse=True):
                domain_list.append(pfamA_acc)
                for seq_id, seq_start, seq_end in domain_regions_dict[pfamA_acc]:
                    seq_regions_dict[seq_id].append((pfamA_acc, seq_start, seq_end))
                if num_domain and len(domain_list) >= num_domain:
                    break

            domain_list = ['PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'] + domain_list
            # build domain to id mapping
            domain_id = dict([(d, i) for i, d in enumerate(domain_list)])

            sequences = []
            domains = []
            aa_index = dict(zip(aa_list, range(index_from, index_from + len(aa_list))))
            for seq_id in seq_regions_dict:
                try:
                    sequences.append(np.array([aa_index[a] for a in seq_dict[seq_id]]))
                except KeyError as e:
                    print('{0} parsing {1}: {2}'.format(e, seq_id, seq_dict[seq_id]))
                    raise e
                # initialize domain with 'NO_DOMAIN'
                domain = [domain_id['NO_DOMAIN']] * len(seq_dict[seq_id])
                for pfamA_acc, seq_start, seq_end in seq_regions_dict[seq_id]:
                    domain = domain[:seq_start-1] + [domain_id[pfamA_acc]] * (seq_end - seq_start + 1) + domain[seq_end:]
                domains.append(np.array(domain))

            # save cache
            print('Save sequence domain data...')
            np.savez(seq_dom_cache_path, domain_list=domain_list,
                    domains=domains, sequences=sequences)

        print('Shuffle data...')
        np.random.seed(seed)
        np.random.shuffle(domains)
        np.random.seed(seed)
        np.random.shuffle(sequences)

        print('Test split...')
        x_train = np.array(sequences[:int(len(sequences) * (1 - test_split))])
        y_train = np.array(domains[:int(len(sequences) * (1 - test_split))])

        x_test = np.array(sequences[int(len(sequences) * (1 - test_split)):])
        y_test = np.array(domains[int(len(sequences) * (1 - test_split)):])

        print('Get max length...')
        maxlen_train = max([len(x) for x in x_train])
        maxlen_test = max([len(x) for x in x_test])

        # save cache
        print('Save split data...')
        np.savez(seq_dom_split_cache_path, x_train=x_train, y_train=y_train, maxlen_train=maxlen_train,
                x_test=x_test, y_test=y_test, maxlen_test=maxlen_test, domain_list=domain_list)

    print(len(x_train), 'train sequences') # 3442895 train sequences
    print(len(x_test), 'test sequences') # 860724 test sequences
    # print(domain_list)
    num_classes = len(domain_list)
    print(num_classes, 'classes') # 13 classes
    print('maxlen_train:', maxlen_train) # d10: 25572
    print('maxlen_test:', maxlen_test) # d10: 22244

    return (x_train, y_train, maxlen_train), (x_test, y_test, maxlen_test), domain_list


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

        num_classes = num_domain + 3
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

def sequence_batch_generator(x, y, batch_size, maxlen):
    print(x.shape, y.shape, batch_size, maxlen)
    while True:
        # loop once per epoch
        num_seqs = x.shape[0]
        indices = np.random.permutation(np.arange(num_seqs))
        num_batches = num_seqs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            maxlen_x_batch = max([len(xb) for xb in x[sids]])
            maxlen_y_batch = max([len(yb) for yb in y[sids]])
            x_batch = sequence.pad_sequences(x[sids], maxlen=maxlen_x_batch)
            y_batch = sequence.pad_sequences(y[sids], maxlen=maxlen_x_batch).reshape(len(sids), maxlen_x_batch, 1)
            print(x_batch.shape, y_batch.shape, maxlen_x_batch, maxlen_y_batch)
            yield x_batch, y_batch

def sparse_train(model, model_name, file_name, num_domain, epoch_start=0, save_freq=1, 
          test_split=0.2, validation_split=0.1, maxlen=36805, batch_size=256, 
          epochs=150, device='/gpu:0', csvlog_dir='./csvlogs',
          model_dir='./models', predicted_dir='./predicted', confusion_matrix_dir='./confusion_matrix',
          classification_report_dir='./classification_report', tensorboard_logs_dir='./tfevents'):
    trainer_name = 'cs{0}ts{1:.0f}vs{2:.0f}ml{3}bs{4}'.format(
        num_domain, test_split * 100, validation_split * 100, maxlen, batch_size)
    
    # make sure directories we're going to write into exist
    dir_check(csvlog_dir)
    dir_check(model_dir)
    dir_check(predicted_dir)
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
        (x_train, y_train_class, maxlen_train), (x_test, y_test_class, maxlen_test), domain_list = load_data(
            num_domain=num_domain, test_split=test_split)
        # maxlen = 1000

        # seq_dict[71201428]
        # domain_regions_dict[16712]
        # 3442895 train sequences
        # 860724 test sequences
        # 13 classes
        # maxlen_train: 25572
        # maxlen_test: 22244
        # x_train shape: (3442895,)
        # x_test shape: (860724,)
        # y_train shape: (3442895,)
        # y_test shape: (860724,)

        # print('Pad sequences')
        # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        # y_train_class = sequence.pad_sequences(y_train_class, maxlen=maxlen)
        # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        # y_test_class = sequence.pad_sequences(y_test_class, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_train shape:', y_train_class.shape)
        print('y_test shape:', y_test_class.shape) # (297679, )
        # x_train = x_train.reshape(len(x_train),-1,1)
        # x_test = x_test.reshape(len(x_test),-1,1)
        # y_train_class = y_train_class.reshape(len(y_train_class), maxlen, 1)
        # y_test_class = y_test_class.reshape(len(y_test_class), maxlen, 1)
        # print('x_train shape:', x_train.shape)
        # print('x_test shape:', x_test.shape)
        # print('y_train shape:', y_train_class.shape)
        # print('y_test shape:', y_test_class.shape) # (297679, )
        train_steps = x_train.shape[0] // batch_size
        train_gen = sequence_batch_generator(x_train, y_train_class, batch_size, maxlen_train)
        test_steps = x_test.shape[0] // batch_size
        test_gen = sequence_batch_generator(x_test, y_test_class, batch_size, maxlen_test)

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
                history = model.fit_generator(train_gen,
                    steps_per_epoch=train_steps,
                    epochs=epoch+1,
                    verbose=1,
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
            y_pred_path = '{}.e{}.{}.{}.npz'.format(
                os.path.abspath(os.path.join(predicted_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            y_pred_text_path = '{}.e{}.{}.{}.txt'.format(
                os.path.abspath(os.path.join(predicted_dir, os.path.splitext(os.path.basename(file_name))[0])), 
                epoch, trainer_name, model_name)
            if os.path.exists(y_pred_path):
                print('from {0}'.format(y_pred_path))
                f = np.load(y_pred_path)
                y_test_class = f['y_test_class']
                y_pred_class = f['y_pred_class']
            else:
                steps = test_steps
                steps_done = 0
                enqueuer = GeneratorEnqueuer(test_gen, use_multiprocessing=False, wait_time=0.01)
                enqueuer.start(workers=1, max_queue_size=10)
                output_generator = enqueuer.get()
                progbar = Progbar(target=steps)
                y_test_class_list = []
                y_pred_class_list = []
                write_text = False
                if not os.path.exists(y_pred_text_path):
                    write_text = True
                    f = open(y_pred_text_path, 'w')
                while steps_done < steps:
                    x, y = next(output_generator)
                    y_pred = model.predict_on_batch(x)
                    # y_pred.shape = (32, ?, 13)
                    y_c = y.reshape(y.shape[0], y.shape[1])
                    y_pred_c = [np.argmax(s, axis=1) for s in y_pred]
                    if write_text:
                        for i in range(len(x)):
                            # sequence
                            f.write('>{}\n'.format(''.join([aa_list[a-1] for a in x[i] if a != 0])))
                            # expected
                            f.write('@{}\n'.format(''.join(['{:X}'.format(d) for d in y_c[i] if d != 0])))
                            # predicted
                            f.write('${}\n'.format(''.join(['{:X}'.format(d) for d in y_pred_c[i] if d != 0])))
                    y_test_class_list.append(np.concatenate(y_c))
                    y_pred_class_list.append(np.concatenate(y_pred_c))
                    steps_done += 1
                    progbar.update(steps_done)
                if write_text:
                    f.close()
                
                # y_pred = model.predict_generator(test_gen, steps=test_steps, verbose=1)
                # y_pred = [b.tolist() for b in y_pred]
                # with open(y_pred_path, 'w') as f:
                #     json.dump(y_pred, f)
                # y_pred = np.array(y_pred)
                print('y_pred_class_list len:', len(y_pred_class_list)) # (297679, 10)
                y_test_class = np.concatenate(y_test_class_list)
                y_pred_class = np.concatenate(y_pred_class_list)
                print('y_pred_class shape:', y_pred_class.shape) # (297679, 10)
                np.savez_compressed(y_pred_path, y_test_class=y_test_class, y_pred_class=y_pred_class)
            # y_pred_class = np.argmax(y_pred, axis=1)
            # print('y_pred_class shape:', y_pred_class.shape) # (297679, 10)
            # del y_pred
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

        num_classes = num_domain + 3
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
