
import os
import gzip
import json
import struct
import argparse
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
    print('Parsing {}'.format(os.path.basename(fa_path)))
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
    seq_dom_split_cache_path = '{0}-d{1}-s{2}.npz'.format(os.path.splitext(
        os.path.splitext(regions_path)[0])[0], num_domain or 0, int(test_split * 100))
    if os.path.exists(seq_dom_split_cache_path):
        print('Loading {0}'.format(seq_dom_split_cache_path))
        f = np.load(seq_dom_split_cache_path)
        x_train = f['x_train']
        y_train = f['y_train']
        maxlen_train = f['maxlen_train'].tolist()
        x_test = f['x_test']
        y_test = f['y_test']
        maxlen_test = f['maxlen_test'].tolist()
        domain_list = f['domain_list']
    else:
        print('Building {0}'.format(seq_dom_split_cache_path))
        seq_dom_cache_path = '{0}-d{1}.npz'.format(os.path.splitext(
            os.path.splitext(regions_path)[0])[0], num_domain or 0)
        if os.path.exists(seq_dom_cache_path):
            print('Loading {0}'.format(seq_dom_cache_path))
            f = np.load(seq_dom_cache_path)
            domain_list = f['domain_list']
            domains = f['domains']
            sequences = f['sequences']
        else:
            print('Building {0}'.format(seq_dom_cache_path))
            # seq_dict[seq_id] = sequence
            seq_dict_cache_path = '{0}.json'.format(os.path.splitext(uniprot_path)[0])
            if os.path.exists(seq_dict_cache_path):
                print('Loading {0}'.format(seq_dict_cache_path))
                with open(seq_dict_cache_path, 'r') as f:
                    seq_dict = json.load(f)
            else:
                print('Building {0}'.format(seq_dict_cache_path))
                seq_dict = _fa_gz_to_dict(uniprot_path)
                with open(seq_dict_cache_path, 'w') as f:
                    json.dump(seq_dict, f)

            domain_regions_dict_cache_path = '{0}.json'.format(os.path.splitext(regions_path)[0])
            if os.path.exists(domain_regions_dict_cache_path):
                print('Loading {0}'.format(domain_regions_dict_cache_path))
                with open(domain_regions_dict_cache_path, 'r') as f:
                    domain_regions_dict = json.load(f)
            else:
                print('Building {0}'.format(domain_regions_dict_cache_path))
                domain_regions_dict = _pfam_regions_tsv_gz_to_dict(regions_path)
                with open(domain_regions_dict_cache_path, 'w') as f:
                    json.dump(domain_regions_dict, f)
                
            print('seq_dict[{}]'.format(len(seq_dict))) # seq_dict[71201428]
            print('domain_regions_dict[{}]'.format(len(domain_regions_dict))) # domain_regions_dict[16712]

            domain_list = []
            # build seq_regions_dict[seq_id] = [(pfamA_acc, seq_start, seq_end), ...]
            seq_regions_dict = defaultdict(list)
            # domains with the most sequences first
            prog = tf.keras.utils.Progbar(num_domain or len(domain_regions_dict))
            for i, pfamA_acc in enumerate(sorted(domain_regions_dict, key=lambda k: (len(domain_regions_dict[k]), k), reverse=True)):
                prog.update(i)
                domain_list.append(pfamA_acc)
                for seq_id, seq_start, seq_end in domain_regions_dict[pfamA_acc]:
                    seq_regions_dict[seq_id].append((pfamA_acc, seq_start, seq_end))
                if num_domain and len(domain_list) >= num_domain:
                    break
            prog.update(num_domain or len(domain_regions_dict), force=True)

            domain_list = ['PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'] + domain_list
            # build domain to id mapping
            domain_id = dict([(d, i) for i, d in enumerate(domain_list)])

            sequences = []
            domains = []
            aa_index = dict(zip(aa_list, range(index_from, index_from + len(aa_list))))
            prog = tf.keras.utils.Progbar(len(seq_regions_dict))
            for i, seq_id in enumerate(seq_regions_dict):
                prog.update(i)
                try:
                    sequences.append(np.array([aa_index[a] for a in seq_dict[seq_id]], dtype=np.uint8))
                except KeyError as e:
                    print('{0} parsing {1}: {2}'.format(e, seq_id, seq_dict[seq_id]))
                    raise e
                # initialize domain with 'NO_DOMAIN'
                domain = [domain_id['NO_DOMAIN']] * len(seq_dict[seq_id])
                for pfamA_acc, seq_start, seq_end in seq_regions_dict[seq_id]:
                    domain = domain[:seq_start-1] + [domain_id[pfamA_acc]] * (seq_end - seq_start + 1) + domain[seq_end:]
                domains.append(np.array(domain, dtype=np.uint16))
            prog.update(len(seq_regions_dict), force=True)

            # save cache
            # print('Save sequence domain data...')
            # try:
            #     np.savez(seq_dom_cache_path, domain_list=domain_list,
            #             domains=domains, sequences=sequences)
            # except Error as e:
            #     # MemoryError
            #     # ValueError Zip64 Limit 48GB
            #     print(e)

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
        # print('Save split data...')
        # try:
        #     np.savez(seq_dom_split_cache_path, x_train=x_train, y_train=y_train, maxlen_train=maxlen_train,
        #             x_test=x_test, y_test=y_test, maxlen_test=maxlen_test, domain_list=domain_list)
        # except Error as e:
        #     # MemoryError
        #     # ValueError Zip64 Limit 48GB
        #     print(e)

    print(len(x_train), 'train sequences') # 3442895 train sequences
    print(len(x_test), 'test sequences') # 860724 test sequences
    # print(domain_list)
    num_classes = len(domain_list)
    print(num_classes, 'classes') # 13 classes
    print('maxlen_train:', maxlen_train) # d10: 25572
    print('maxlen_test:', maxlen_test) # d10: 22244

    return (x_train, y_train, maxlen_train), (x_test, y_test, maxlen_test), domain_list

def save_to_tfrecord(x, y, path):
    """Saves x and y to tfrecord file.
    """
    print('Writing {}'.format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        prog = tf.keras.utils.Progbar(len(x))
        for index in range(len(x)):
            prog.update(index)
            protein = x[index].astype(np.uint8)
            domains = y[index].astype(np.uint16)
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
        prog.update(len(x), force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepares files on disk for dataset API.')
    parser.add_argument('-n', '--num_classes', type=int, default=None,
        help='Include only the top N domain classes in the dataset file, include all domain classes if None.')
    parser.add_argument('-s', '--test_split', type=float, default=0.2,
        help='Fraction of the dataset to be used as test data.')
    parser.add_argument('-c', '--cache_root', type=str, default=os.path.join(os.path.expanduser('~'), '.keras'),
        help="Location to store cached files, defaults to '~/.keras'.")
    parser.add_argument('-d', '--cache_dir', type=str, default='datasets',
        help="Subdirectory under the cache_root dir where the file is saved.")

    args, unparsed = parser.parse_known_args()
    args.cache_root = 'D:\\'
    print(args)

    print('Loading data...')
    (x_train, y_train_class, maxlen_train), (x_test, y_test_class, maxlen_test), domain_list = load_data(
        num_domain=args.num_classes, test_split=args.test_split, 
        cache_dir=args.cache_root, cache_subdir=args.cache_dir)

    # convert to tfrecords
    train_path = os.path.join(args.cache_root, args.cache_dir, 
        'pfam-regions-d{}-s{}-{}.tfrecords'.format(args.num_classes or 0, int(args.test_split * 100), 'train'))
    save_to_tfrecord(x_train, y_train_class, train_path)
    test_path = os.path.join(args.cache_root, args.cache_dir, 
        'pfam-regions-d{}-s{}-{}.tfrecords'.format(args.num_classes or 0, int(args.test_split * 100), 'test'))
    save_to_tfrecord(x_test, y_test_class, test_path)
    