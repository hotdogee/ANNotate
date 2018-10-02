# ANNotate
### Protein sequence annotation using artificial neural networks

# Setup Environment

1. Follow official TensorFlow docs and make sure the required GPU drivers and libraries are installed and working.

   * https://www.tensorflow.org/install/gpu

2. Install Python 3.6

3. Install Python libraries

   * `pip install -r requirements.txt`

   * This installs the following libraries and its dependencies:
     ```
     tensorflow-gpu>=1.11
     colorama>=0.3.9
     coloredlogs>=10.0
     msgpack>=0.5.6
     tqdm>=4.26.0
     ujson>=1.35
     ```

# Download and Build the Pfam Regions Dataset

`python datasets/pfam_regions_build.py`

## Features

* Preprocess uniprot FASTA and regions TSV files into TFRECORDS files suitable for training:
  1. Training Set: `pfam-regions-train.tfrecords`
  2. Testing Set: `pfam-regions-test.tfrecords`
  3. Metadata: `pfam-regions-meta.json`

* Downloads uniprot FASTA and regions TSV files from the pfam FTP server into DATASET_DIR if those files don't already exist.

* Intermediate data are cached in MSGPACK files to speed up future builds with different arguments.

* Configure the fraction of sequences to be include in the testing set, default is 0.2 (20%).

* Option to Include only the top N domain classes in the dataset file, includes all domains by default.

* Option to limit the number of sequences to include in the training and/or testing sets, default is no limit.

## Arguments
```
optional arguments:
  -h, --help            show this help message and exit
  -n NUM_CLASSES, --num_classes NUM_CLASSES
                        Include only the top N domain classes in the dataset
                        file, include all domain classes if None.
  -s TEST_SPLIT, --test_split TEST_SPLIT
                        Fraction of the dataset to be used as test data,
                        default is 0.2.
  -t MAX_SEQ_PER_CLASS_IN_TRAIN, --max_seq_per_class_in_train MAX_SEQ_PER_CLASS_IN_TRAIN
                        Maximum the number of sequences to include in the
                        training datasets, default is no limit.
  -e MAX_SEQ_PER_CLASS_IN_TEST, --max_seq_per_class_in_test MAX_SEQ_PER_CLASS_IN_TEST
                        Maximum the number of sequences to include in the
                        testing datasets, default is no limit.
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        Location to store dataset files, default is ~/datasets.
```

## Examples
* Build full dataset with 20% test split, and store files at ~/datasets

  ```python datasets/pfam_regions_build.py```

* Build 10 class dataset with 20% test split, and store files at ~/datasets

  ```python datasets/pfam_regions_build.py --num_classes=10 --test_split=0.2 --dataset_dir=~/dataset```

* Build a toy dataset that can finish in 750 steps (300 sec) - 3 sequence/class

  ```python datasets/pfam_regions_build.py -t 3 -e 3 -d ~/datasets```

* Build a toy dataset that can finish in 300 steps (120 sec) - 2 sequence/class

  ```python datasets/pfam_regions_build.py -t 2 -e 2 -d ~/datasets```

* Build a toy dataset that can finish in 40 steps (20 sec) - 1 sequence/class - 4000 classes
  
  ```python datasets/pfam_regions_build.py -n 4000 -t 1 -e 1 -d ~/datasets```


# Train modal

`python models/pfam-regions/v5-BiRnn.py`

## Train with 10 class dataset, using batch size of 64
```python models/pfam-regions/v5-BiRnn.py --training_data=/home/user/datasets/pfam-regions-d10-s20-train.tfrecords --eval_data=/home/user/datasets/pfam-regions-d10-s20-test.tfrecords --model_dir=./checkpoints/d10-v1 --batch_size=64```

## Train with full dataset, using batch size of 4
```python models/pfam-regions/v5-BiRnn.py --training_data=/home/user/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/user/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0-v1 --num_classes=16715 --batch_size=4```