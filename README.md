# ANNotate
Genetic sequence annotation using artificial neural networks

# Download and Build TFRecords
## Build 10 class dataset with 20% test split, and store files at ~/datasets
```python util/make_dataset_pfam_regions.py -n 10 -s 0.2 -c ~/ -d datasets```
## Build full dataset with 20% test split, and store files at ~/datasets
```python util/make_dataset_pfam_regions.py -s 0.2 -c ~/ -d datasets```

# Train modal
## Train with 10 class dataset, using batch size of 64
```python main.py --training_data=/home/user/datasets/pfam-regions-d10-s20-train.tfrecords --eval_data=/home/user/datasets/pfam-regions-d10-s20-test.tfrecords --model_dir=./checkpoints/d10-v1 --batch_size=64```

## Train with full dataset, using batch size of 4
```python main.py --training_data=/home/user/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/user/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0-v1 --num_classes=16715 --batch_size=4```