# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from six.moves import zip
import os
import gzip
import json
import struct
import warnings
import numpy as np
from collections import defaultdict

from keras.utils import get_file, Progbar


def _gzip_size(filename):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _pfam_fa_to_domain_dict(fa_path):
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

    path = get_file(fname, origin_base + fname, cache_subdir=cache_subdir, cache_dir=cache_dir)

    aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'
    # check cache
    cache_path = '{0}-d{1}.npz'.format(os.path.splitext(os.path.splitext(path)[0])[0], num_domain)
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

    return (x_train, y_train), (x_test, y_test), (aa_list, domain_list)
