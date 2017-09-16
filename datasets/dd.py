# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from six.moves import zip
import os
import re
import gzip
import json
import struct
import warnings
import numpy as np
from collections import defaultdict

from keras.utils import get_file, Progbar


def _dd_to_fold_dict(dd_path):
    fold_dict = defaultdict(list)
    fold, seq_id, sequence = '', '', ''
    target = os.path.getsize(dd_path)
    prog = Progbar(target)
    current = 0
    with open(dd_path, 'rb') as dd_f:
        for line in dd_f:
            # if (int(dd_f.tell()/target*100) > current):
            #     current = int(dd_f.tell()/target*100)
            #     print('{}/{} ({:.2f}%)'.format(dd_f.tell(), target, current))
            prog.update(dd_f.tell())
            line = line.strip().decode('utf-8')
            if not len(line) or line[:2] == '--' or  line[:2] == '**': continue
            if line[:4] == 'TYPE':
                fold = line
            elif line[0] == '>':
                if sequence:
                    fold_dict[fold].append((seq_id, sequence))
                    seq_id, sequence = '', ''
                # parse header
                seq_id = line.split()[0]
            else:
                sequence += line.replace(' ', '')
        prog.update(dd_f.tell(), force=True)
    return fold_dict


def load_data(origin_train='http://binfo.shmtu.edu.cn/profold/files/DD-train.dataset',
              origin_test='http://binfo.shmtu.edu.cn/profold/files/DD-test.dataset',
              seed=113,
              index_from=1,
              cache_subdir='datasets',
              cache_dir=None,
              **kwargs):
    """Loads the DD-dataset proposed by Ding and Dubchak in 2001 and modified by Shen and Chou in 2006.
    There are 311 protein sequences in the training set and 386 protein sequences in the testing set with 
    no two proteins having more than 35% of sequence identity. The protein sequences in DD-dataset were 
    selected from 27 SCOP folds comprehensively, which belong to different structural classes containing 
    α, β, α/β, and α + β.

    C.H. Ding and I. Dubchak, “Multi-class protein fold recognition using support vector machines and 
    neural networks,” Bioinformatics, vol. 17, no. 4, pp. 349–358, 2001. 
    H.-B. Shen and K.-C. Chou, “Ensemble classifier for protein fold pattern recognition,” Bioinformatics, 
    vol. 22, no. 14, pp. 1717–1722, 2006.

    # Arguments
        origin_train: train dataset download location.
        origin_test: test dataset download location.
        seed: random seed for sample shuffling.
        index_from: index amino acids with this index and higher.
            Set to 1 because 0 is usually the padding character.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test), (aa_list, fold_list)`.
    """
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    train_path = get_file(os.path.basename(origin_train), origin_train,
                          cache_subdir=cache_subdir, cache_dir=cache_dir)
    test_path = get_file(os.path.basename(origin_test), origin_test,
                         cache_subdir=cache_subdir, cache_dir=cache_dir)

    aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'
    # check cache
    cache_path = os.path.join(os.path.split(train_path)[0], 'dd.npz')
    if os.path.exists(cache_path):
        print('from {0}'.format(cache_path))
        f = np.load(cache_path)
        fold_list = f['fold_list']
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']
    else:
        aa_index = dict(zip(aa_list, range(index_from, index_from + len(aa_list))))
        train_dict = _dd_to_fold_dict(train_path)
        test_dict = _dd_to_fold_dict(test_path)
        fold_list = []
        x_train, y_train, x_test, y_test = [], [], [], []
        fi = 0
        for fold in sorted(train_dict, key=lambda k: int(re.findall(r'\((\d+)\)', k)[0])):
            fold_list.append(fold)
            for seq_id, seq in train_dict[fold]:
                y_train.append(fi)
                try:
                    x_train.append([aa_index[a] for a in seq])
                except KeyError as e:
                    print('{0} parsing {1}: {2}'.format(e, seq_id, seq))
                    raise e
            for seq_id, seq in test_dict[fold]:
                y_test.append(fi)
                try:
                    x_test.append([aa_index[a] for a in seq])
                except KeyError as e:
                    print('{0} parsing {1}: {2}'.format(e, seq_id, seq))
                    raise e
            fi += 1

        # save cache
        np.savez(cache_path, fold_list=fold_list,
                 x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test)), (aa_list, fold_list)
