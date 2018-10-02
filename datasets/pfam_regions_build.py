
import os
import time
import math
import gzip
import ujson
import msgpack
import struct
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf

def _gzip_size(filename):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _fa_gz_to_dict(fa_path):
    """Parse a FASTA.gz file into fa_dict[seq_id] = sequence
    """
    fa_dict = {}
    seq_id, sequence = '', ''
    target = _gzip_size(fa_path)
    while target < os.path.getsize(fa_path):
        # the uncompressed size can't be smaller than the compressed size, so add 4GB
        target += 2**32
    prog = tf.keras.utils.Progbar(target)
    # current = 0
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
        prog.update(fa_f.tell())
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
    prog = tf.keras.utils.Progbar(target)
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
        prog.update(tsv_f.tell())
    return domain_regions_dict

aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

def load_data(uniprot_file='uniprot.gz',
              regions_file='Pfam-A.regions.uniprot.tsv.gz',
              origin_base='ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam31.0/',
              num_domain=10,
              test_split=0.2,
              max_seq_per_class_in_train=None,
              max_seq_per_class_in_test=None,
              seed=113,
              index_from=1,
              cache_subdir='datasets',
              cache_dir='~',
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
            defaults to '~/.keras'.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    # ~/.keras/datasets/uniprot.gz
    uniprot_path = tf.keras.utils.get_file(uniprot_file, origin_base + uniprot_file,
                    cache_subdir=cache_subdir, cache_dir=cache_dir)

    # ~/.keras/datasets/Pfam-A.regions.uniprot.tsv.gz
    regions_path = tf.keras.utils.get_file(regions_file, origin_base + regions_file,
                    cache_subdir=cache_subdir, cache_dir=cache_dir)

    # check cache
    # seq_dom_split_cache_path = '{0}-d{1}-s{2}.npz'.format(os.path.splitext(
    #     os.path.splitext(regions_path)[0])[0], num_domain or 0, int(test_split * 100))
    # if os.path.exists(seq_dom_split_cache_path):
    #     print('Loading {0}'.format(seq_dom_split_cache_path))
    #     f = np.load(seq_dom_split_cache_path)
    #     x_train = f['x_train']
    #     y_train = f['y_train']
    #     maxlen_train = f['maxlen_train'].tolist()
    #     x_test = f['x_test']
    #     y_test = f['y_test']
    #     maxlen_test = f['maxlen_test'].tolist()
    #     domain_list = f['domain_list']
    # else:
    #     print('Building {0}'.format(seq_dom_split_cache_path))
    #     seq_dom_cache_path = '{0}-d{1}.npz'.format(os.path.splitext(
    #         os.path.splitext(regions_path)[0])[0], num_domain or 0)
    #     if os.path.exists(seq_dom_cache_path):
    #         print('Loading {0}'.format(seq_dom_cache_path))
    #         f = np.load(seq_dom_cache_path)
    #         domain_list = f['domain_list']
    #         domains = f['domains']
    #         sequences = f['sequences']
    #     else:
    #         print('Building {0}'.format(seq_dom_cache_path))

    print('Loading Protein Sequence Data from ', end="", flush=True)
    # seq_dict[seq_id] = sequence
    seq_dict_cache_path = '{0}.msgpack'.format(os.path.splitext(uniprot_path)[0])
    if os.path.exists(seq_dict_cache_path):
        print('CACHE: {0}'.format(seq_dict_cache_path), end="", flush=True)
        start = time.perf_counter()
        with open(seq_dict_cache_path, 'rb') as f:
            seq_dict = msgpack.load(f, raw=False)
        print(' - {:.0f}s'.format(time.perf_counter() - start))
        # Loading Protein Sequence Data from CACHE: D:/datasets2\uniprot.msgpack - 84s
    else:
        print('SOURCE: {0}'.format(uniprot_path))
        seq_dict = _fa_gz_to_dict(uniprot_path)
        # 30346375850/30346375850 [==============================] - 914s 0us/step
        with open(seq_dict_cache_path, 'wb') as f:
            msgpack.dump(seq_dict, f)
            # uniprot.msgpack 23.1 GB (24,875,252,089 位元組)
    # seq_dict stats
    seq_lengths = sorted([len(s) for s in seq_dict.values()])
    print('Loaded {:,} Sequences, Amino Acids Min {:,}, Median {:,}, Max {:,}, Total {:,}'.format(
        len(seq_lengths), seq_lengths[0], seq_lengths[len(seq_lengths)//2], seq_lengths[-1], sum(seq_lengths)))
    # Loaded 71,201,428 Sequences, (2, 267, 36,805, 23,867,549,122) = (min, median, max, total)
    
    print('Loading Domain Region Data from ', end="", flush=True)
    # domain_regions_dict[pfamA_acc] = [(uniprot_acc + '.' + seq_version, seq_start, seq_end), ...]
    domain_regions_dict_cache_path = '{0}.msgpack'.format(os.path.splitext(regions_path)[0])
    if os.path.exists(domain_regions_dict_cache_path):
        print('CACHE: {0}'.format(domain_regions_dict_cache_path), end="", flush=True)
        start = time.perf_counter()
        with open(domain_regions_dict_cache_path, 'rb') as f:
            domain_regions_dict = msgpack.load(f, raw=False)
        print(' - {:.0f}s'.format(time.perf_counter() - start))
        # Loading Domain Region Data from CACHE: D:/datasets2\Pfam-A.regions.uniprot.tsv.msgpack - 75s
    else:
        print('SOURCE: {0}'.format(regions_path))
        domain_regions_dict = _pfam_regions_tsv_gz_to_dict(regions_path)
        # 6758658323/6758658323 [==============================] - 557s 0us/step
        with open(domain_regions_dict_cache_path, 'wb') as f:
            msgpack.dump(domain_regions_dict, f)
            # Pfam-A.regions.uniprot.tsv.msgpack 1.32 GB (1,420,160,623 位元組)
    
    # domain_regions_dict stats
    domain_lengths = sorted([len(s) for s in domain_regions_dict.values()])
    print('Loaded {:,} Domains, Regions Min {:,}, Median {:,}, Max {:,}, Total {:,}'.format(
        len(domain_lengths), domain_lengths[0], domain_lengths[len(domain_lengths)//2], domain_lengths[-1], sum(domain_lengths)))
    # Loaded 16,712 Domains, Regions Min 2, Median 863, Max 1,078,482, Total 88,761,542

    # build domain_seq_dict

    # seq_regions_dict[seq_id] = [(pfamA_acc, seq_start, seq_end), ...]
    seq_regions_dict = defaultdict(list)
    # domain_seq_dict[pfamA_acc][seq_id] = [(pfamA_acc, seq_start, seq_end), ...]
    domain_seq_dict = defaultdict(dict)
    # domains with the most sequences first
    num_domains_wanted = num_domain or len(domain_regions_dict)
    print('Collecting Sequences Containing the Top {} Domains'.format(num_domains_wanted))
    prog = tf.keras.utils.Progbar(num_domains_wanted)
    for i, pfamA_acc in enumerate(sorted(domain_regions_dict, key=lambda k: (len(domain_regions_dict[k]), k), reverse=True)):
        if num_domain and i >= num_domain:
            break
        prog.update(i)
        for seq_id, seq_start, seq_end in domain_regions_dict[pfamA_acc]:
            seq_regions_dict[seq_id].append((pfamA_acc, seq_start, seq_end))
            if seq_id not in domain_seq_dict[pfamA_acc]:
                domain_seq_dict[pfamA_acc][seq_id] = seq_regions_dict[seq_id]
    prog.update(num_domains_wanted)
    print('Collected {:,} Sequences with {:,} Domains'.format(len(seq_regions_dict), len(domain_seq_dict)))
    # 16712/16712 [==============================] - 140s 8ms/step
    # Collected 54,223,493 Sequences with 16,712 Domains

    # create train test split for every domain
    domain_list = []
    train_set = set()
    test_set = set()
    np.random.seed(seed)
    status_text = 'Distributing Sequences to Training and Testing Sets with {:.0%} Split'.format(test_split)
    if max_seq_per_class_in_train:
        status_text += ' and Max {} Sequences per Domain in Training Set'.format(max_seq_per_class_in_train)
    if max_seq_per_class_in_test:
        status_text += ' and Max {} Sequences per Domain in Testing Set'.format(max_seq_per_class_in_test)
    print(status_text)
    prog = tf.keras.utils.Progbar(len(domain_seq_dict))
    # assign domains with fewer sequences first
    for i, pfamA_acc in enumerate(sorted(domain_seq_dict, key=lambda k: (len(domain_seq_dict[k]), k))):
        prog.update(i)
        seq_count = len(domain_seq_dict[pfamA_acc])
        if seq_count < 2: # need at least 2 sequences
            continue
        domain_list.append(pfamA_acc)
        # calculate expected train test counts
        if test_split < 0.5:
            test_count = math.ceil(seq_count * test_split)
        else:
            test_count = int(seq_count * test_split)
        train_count = seq_count - test_count
        # check sequences already in train_set and test_set
        all_seq_set = set(domain_seq_dict[pfamA_acc])
        test_count -= len(test_set & all_seq_set)
        train_count -= len(train_set & all_seq_set)
        seq_list = list(all_seq_set - test_set - train_set)
        # assert counts
        if test_count < 0 or train_count < 0:
            print('\r WARNING: {}={}, seq_count={}, test_count={}, train_count={}, already in test_set= {}, already in train_set= {}'.format(
                i, pfamA_acc, seq_count, test_count, train_count, len(test_set & all_seq_set), len(train_set & all_seq_set)
            ))
            test_count = max(0, test_count)
            train_count = max(0, train_count)
        np.random.shuffle(seq_list)
        # limit max_seq_per_class
        if max_seq_per_class_in_test:
            test_count = min(test_count, max_seq_per_class_in_test)
            if not max_seq_per_class_in_train:
                train_count = len(seq_list) - test_count
        if max_seq_per_class_in_train:
            train_count = min(train_count, max_seq_per_class_in_train)
            if not max_seq_per_class_in_test:
                test_count = len(seq_list) - train_count
        # add selected sequences to train and test sets
        test_set |= set(seq_list[len(seq_list)-test_count:])
        train_set |= set(seq_list[:train_count])
    prog.update(len(domain_seq_dict))
    print('Collected {:,} Training Sequences and {:,} Testing Sequences with {:,} Domains'.format(len(train_set), len(test_set), len(domain_list)))
    # Collected 16711 Training Sequences and 16711 Testing Sequences with 16711 Domains
    # Collected 33,399 Training Sequences and 33,317 Testing Sequences with 16,711 Domains
    # Distributing Sequences to Training and Testing Sets with 20% Split and Max 3 Sequences per Domain per Set
    # 16712/16712 [==============================] - 33s 2ms/step
    # Collected 50,066 Training Sequences and 49,799 Testing Sequences with 16,711 Domains
    # 16712/16712 [==============================] - 75s 5ms/step
    # Collected 43,373,043 Training Sequences and 10,850,449 Testing Sequences with 16,711 Domains

    domain_list.reverse() # popular domains first
    domain_list = ['PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'] + domain_list
    # build domain to id mapping
    domain_index = dict([(d, i) for i, d in enumerate(domain_list)])
    aa_index = dict(
        zip(aa_list, range(index_from, index_from + len(aa_list)))
        # msgpack.load produces byte literals
        #  + zip(str.encode(aa_list), range(index_from, index_from + len(aa_list)))
    )

    train_list = list(train_set)
    np.random.shuffle(train_list)
    test_list = list(test_set)
    np.random.shuffle(test_list)
    # build dataset
    dataset = {
        'train': {
            'seq_ids': train_list,
            'proteins': [],
            'domains': []
        },
        'test': {
            'seq_ids': test_list,
            'proteins': [],
            'domains': []
        }
    }
    # build metadata
    metadata = {
        'train': {
            'seq_count': {
                'total': len(train_list),
                'per_domain': {
                    'min': 0,
                    'median': 0,
                    'max': 0
                }
            },
            'seq_len': {
                'total': 0,
                'min': 0,
                'median': 0,
                'max': 0
            }
        },
        'test': {
            'seq_count': {
                'total': len(test_list),
                'per_domain': {
                    'min': 0,
                    'median': 0,
                    'max': 0
                }
            },
            'seq_len': {
                'total': 0,
                'min': 0,
                'median': 0,
                'max': 0
            }
        },
        'aa_list': aa_list,
        'aa_index': aa_index,
        'domain_list': domain_list,
        'domain_index': domain_index,
        'domain_count': len(domain_list)
    }

    # sequences = []
    # domains = []
    for k in dataset:
        print('Generating Domain Sequence Representation for {} Dataset'.format(k.upper()))
        prog = tf.keras.utils.Progbar(len(dataset[k]['seq_ids']))
        domain_seq_count = defaultdict(set)
        seq_len = []
        for i, seq_id in enumerate(dataset[k]['seq_ids']):
            prog.update(i)
            seq_len.append(len(seq_dict[seq_id]))
            try:
                dataset[k]['proteins'].append(np.array([aa_index[a] for a in seq_dict[seq_id]], dtype=np.uint8))
            except KeyError as e:
                print('{0} parsing {1}: {2}'.format(e, seq_id, seq_dict[seq_id]))
                raise e
            # initialize domain with 'NO_DOMAIN'
            domain = [domain_index['NO_DOMAIN']] * len(seq_dict[seq_id])
            for pfamA_acc, seq_start, seq_end in seq_regions_dict[seq_id]:
                domain_seq_count[pfamA_acc].add(seq_id)
                domain = domain[:seq_start-1] + [domain_index[pfamA_acc]] * (seq_end - seq_start + 1) + domain[seq_end:]
            dataset[k]['domains'].append(np.array(domain, dtype=np.uint16))
        prog.update(len(dataset[k]['seq_ids']))
        # record metadata
        seq_count_per_domain = sorted([len(s) for s in domain_seq_count.values()])
        metadata[k]['seq_count']['per_domain']['min'] = seq_count_per_domain[0]
        metadata[k]['seq_count']['per_domain']['median'] = seq_count_per_domain[len(seq_count_per_domain)//2]
        metadata[k]['seq_count']['per_domain']['max'] = seq_count_per_domain[-1]
        seq_len = sorted(seq_len)
        metadata[k]['seq_len']['total'] = sum(seq_len)
        metadata[k]['seq_len']['min'] = seq_len[0]
        metadata[k]['seq_len']['median'] = seq_len[len(seq_len)//2]
        metadata[k]['seq_len']['max'] = seq_len[-1]
        # Generating Domain Sequence Representation for TRAIN Dataset
        # 50066/50066 [==============================] - 14s 278us/step
        # Generating Domain Sequence Representation for TEST Dataset
        # 49799/49799 [==============================] - 12s 247us/step
        # 
        # Generating Domain Sequence Representation for TRAIN Dataset
        # 43375025/43375025 [==============================] - 3471s 80us/step
        # Generating Domain Sequence Representation for TEST Dataset
        # 10848467/10848467 [==============================] - 1125s 104us/step

    return dataset, metadata

            # save cache
            # print('Save sequence domain data...')
            # try:
            #     np.savez(seq_dom_cache_path, domain_list=domain_list,
            #             domains=domains, sequences=sequences)
            # except Error as e:
            #     # MemoryError
            #     # ValueError Zip64 Limit 48GB
            #     print(e)

        # print('Shuffle data...')
        # np.random.seed(seed)
        # np.random.shuffle(domains)
        # np.random.seed(seed)
        # np.random.shuffle(sequences)

        # print('Test split...')
        # x_train = np.array(sequences[:int(len(sequences) * (1 - test_split))])
        # y_train = np.array(domains[:int(len(sequences) * (1 - test_split))])

        # x_test = np.array(sequences[int(len(sequences) * (1 - test_split)):])
        # y_test = np.array(domains[int(len(sequences) * (1 - test_split)):])

        # print('Get max length...')
        # maxlen_train = max([len(x) for x in x_train])
        # maxlen_test = max([len(x) for x in x_test])

        # save cache
        # print('Save split data...')
        # try:
        #     np.savez(seq_dom_split_cache_path, x_train=x_train, y_train=y_train, maxlen_train=maxlen_train,
        #             x_test=x_test, y_test=y_test, maxlen_test=maxlen_test, domain_list=domain_list)
        # except Error as e:
        #     # MemoryError
        #     # ValueError Zip64 Limit 48GB
        #     print(e)

    # print(len(x_train), 'train sequences') # 3442895 train sequences
    # print(len(x_test), 'test sequences') # 860724 test sequences
    # # print(domain_list)
    # num_classes = len(domain_list)
    # print(num_classes, 'classes') # 13 classes
    # print('maxlen_train:', maxlen_train) # d10: 25572
    # print('maxlen_test:', maxlen_test) # d10: 22244

    # return (x_train, y_train, maxlen_train), (x_test, y_test, maxlen_test), domain_list

def save_to_tfrecord(data, path):
    """Saves data to tfrecord file.
    """
    print('Writing {}'.format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        prog = tf.keras.utils.Progbar(len(data['seq_ids']))
        for index in range(len(data['seq_ids'])):
            prog.update(index)
            protein = data['proteins'][index].astype(np.uint8)
            domains = data['domains'][index].astype(np.uint16)
            example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        'protein': tf.train.FeatureList(
                            feature=[tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[value.tostring()]))
                                        for value in protein]),
                        'domains': tf.train.FeatureList(
                            feature=[tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[value.tostring()]))
                                        for value in domains])
                    }
                )
            )
            writer.write(example.SerializeToString())
        prog.update(len(data['seq_ids']))
    # Writing D:/datasets2\pfam-regions-d0-s20-p1-train.tfrecords
    # 16711/16711 [==============================] - 105s 6ms/step
    # Writing D:/datasets2\pfam-regions-d0-s20-p1-test.tfrecords
    # 16711/16711 [==============================] - 105s 6ms/step
    # 
    # Writing D:/datasets2\pfam-regions-d0-s20-p2-train.tfrecords
    # 33399/33399 [==============================] - 186s 6ms/step
    # Writing D:/datasets2\pfam-regions-d0-s20-p2-test.tfrecords
    # 33317/33317 [==============================] - 182s 5ms/step
    #
    # Writing D:/datasets2\pfam-regions-d0-s20-p3-train.tfrecords
    # 50066/50066 [==============================] - 344s 7ms/step
    # Writing D:/datasets2\pfam-regions-d0-s20-p3-test.tfrecords
    # 49799/49799 [==============================] - 319s 6ms/step
    # 
    # Writing E:/datasets2\pfam-regions-d0-s20-train.tfrecords
    # 43375025/43375025 [==============================] - 269932s 6ms/step
    # Writing E:/datasets2\pfam-regions-d0-s20-test.tfrecords
    # 10848467/10848467 [==============================] - 65627s 6ms/step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Builds the pfam regions tfrecords dataset.')
    parser.add_argument('-n', '--num_classes', type=int, default=None,
        help='Include only the top N domain classes in the dataset file, include all domain classes if None.')
    parser.add_argument('-s', '--test_split', type=float, default=0.2,
        help='Fraction of the dataset to be used as test data, default is 0.2.')
    parser.add_argument('-t', '--max_seq_per_class_in_train', type=int, default=None,
        help='Maximum the number of sequences to include in the training datasets, default is no limit.')
    parser.add_argument('-e', '--max_seq_per_class_in_test', type=int, default=None,
        help='Maximum the number of sequences to include in the testing datasets, default is no limit.')
    parser.add_argument('-d', '--dataset_dir', type=str, default='~/datasets',
        help="Location to store dataset files, default is ~/datasets.")

    args, unparsed = parser.parse_known_args()
    # args.cache_root = 'D:\\'
    # print(args)
    
    file_prefix = 'pfam-regions-d{}-s{}'.format(args.num_classes or 0, int(args.test_split * 100))
    if args.max_seq_per_class_in_train:
        file_prefix = '{}-t{}'.format(file_prefix, args.max_seq_per_class_in_train)
    if args.max_seq_per_class_in_test:
        file_prefix = '{}-e{}'.format(file_prefix, args.max_seq_per_class_in_test)
    print('\nPreparing Dataset: {}\n'.format(file_prefix))

    # get absolute path to dataset directory
    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))

    # print('Loading data...')
    # (x_train, y_train_class, maxlen_train), (x_test, y_test_class, maxlen_test), domain_list = load_data(
    dataset, metadata = load_data(
        num_domain=args.num_classes, test_split=args.test_split, 
        max_seq_per_class_in_train=args.max_seq_per_class_in_train,
        max_seq_per_class_in_test=args.max_seq_per_class_in_test,
        cache_subdir=dataset_dir)
    
    meta_f = Path(dataset_dir, '{}-meta.json'.format(file_prefix))
    print('Writing Metadata: {} ... '.format(meta_f), end="", flush=True)
    meta_f.parent.mkdir(parents=True, exist_ok=True)
    meta_f.write_text(ujson.dumps(metadata, indent=2, sort_keys=False))
    print('DONE')

    for k in dataset:
        save_to_tfrecord(dataset[k], os.path.join(dataset_dir,
            '{}-{}.tfrecords'.format(file_prefix, k)))
    # train_prefix = '{}-{}'.format(file_prefix, 'train')
    # test_prefix = '{}-{}'.format(file_prefix, 'test')
    ## meta.json
    # sequence_count
    #   total
    #   max
    #   min
    #   median
    # sequence_length
    #   total
    #   max
    #   min
    #   median
    ## domain.csv
    # domain,sequence_count,sequence_length
    # train_meta_f = Path(args.cache_root, args.cache_dir, '{}.json'.format(train_prefix))
    # train_meta_f.parent.mkdir(parents=True, exist_ok=True)
    # train_meta_f.write_text(ujson.dumps({
    #     'number_of_records': len(y_train_class),
    #     'max_record_length': maxlen_train
    # }, indent=2, sort_keys=False))
    # test_meta_f = Path(args.cache_root, args.cache_dir, '{}.json'.format(test_prefix))
    # test_meta_f.write_text(ujson.dumps({
    #     'number_of_records': len(y_test_class),
    #     'max_record_length': maxlen_test
    # }, indent=2, sort_keys=False))

    # # convert to tfrecords
    # train_path = os.path.join(args.cache_root, args.cache_dir, 
    #     'pfam-regions-d{}-s{}-{}.tfrecords'.format(args.num_classes or 0, int(args.test_split * 100), 'train'))
    # save_to_tfrecord(x_train, y_train_class, train_path)
    # test_path = os.path.join(args.cache_root, args.cache_dir, 
    #     'pfam-regions-d{}-s{}-{}.tfrecords'.format(args.num_classes or 0, int(args.test_split * 100), 'test'))
    # save_to_tfrecord(x_test, y_test_class, test_path)
    
    # d0
    # 43,378,794 train sequences (??? bases)
    # 10,844,699 test sequences
    # 16,715 classes
    # maxlen_train: 36,507
    # maxlen_test: 34,350
    # pfam-regions-d0-s20-train.tfrecords
    ## MD5: 1E63566F7CCB5D9207DE8839ABC8A4C2
    # pfam-regions-d0-s20-test.tfrecords
    ## MD5: CB78507F2D2C3A3E7A3171901B2AC3CB
    # pfam-regions-d10-s20-train.tfrecords
    ## MD5: 009C907D5F4576B33C22C59F52531459
    # pfam-regions-d10-s20-test.tfrecords
    ## MD5: 99764A11E4BC86CD162242E08C968FE9

    # Make full dataset
    # python datasets/pfam_regions_build.py -d D:/datasets2
    # MemoryError on 64GB
    #
    # Make a toy dataset that can finish in 150 steps (60 sec)  - 1 sequence/class
    # python datasets/pfam_regions_build.py -t 1 -e 1 -d D:/datasets2
    # "train":{
    #     "seq_count":{
    #     "total":16711,
    #     "per_domain":{
    #         "min":1,
    #         "median":1,
    #         "max":67
    #     }
    #     },
    #     "seq_len":{
    #     "total":7263108,
    #     "min":8,
    #     "median":287,
    #     "max":15858
    #     }
    # },
    # "test":{
    #     "seq_count":{
    #     "total":16711,
    #     "per_domain":{
    #         "min":1,
    #         "median":1,
    #         "max":74
    #     }
    #     },
    #     "seq_len":{
    #     "total":7252482,
    #     "min":7,
    #     "median":283,
    #     "max":15143
    #     }
    # },
    # Build a toy dataset that can finish in 750 steps (300 sec) - 3 sequence/class
    # python datasets/pfam_regions_build.py -t 3 -e 3 -d ~/datasets2
    # Build a toy dataset that can finish in 300 steps (120 sec) - 2 sequence/class
    # python datasets/pfam_regions_build.py -t 2 -e 2 -d ~/datasets2
    # Build a toy dataset that can finish in 40 steps (20 sec)  - 1 sequence/class - 4000 classes
    # python datasets/pfam_regions_build.py -n 4000 -t 1 -e 1 -d ~/datasets2