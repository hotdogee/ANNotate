
import tensorflow as tf

def parser_fn(x):
  return tf.parse_single_example(x, features)

DATASET_BUFFER = 16 * 1024 * 1024 * 1024 # bytes
NUM_PARALLEL_READS = 1 
SHUFFLE_BUFFER = 16 * 1024 # examples
NUM_EPOCHS = -1 # Repeat the input indefinitely.
BATCH_SIZE = 32 # examples

def input_fn():
  # Creates a dataset that reads all of the examples from two files.
  files = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]

  # dataset = tf.data.TFRecordDataset(files, buffer_size=DATASET_BUFFER, num_parallel_reads=NUM_PARALLEL_READS)
  def tfrecord_dataset(filename):
      return tf.data.TFRecordDataset(files, buffer_size=DATASET_BUFFER)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tfrecord_dataset, cycle_length=NUM_PARALLEL_READS, sloppy=True))

  # dataset = dataset.shuffle(SHUFFLE_BUFFER)
  # dataset = dataset.repeat()  # Repeat the input indefinitely.
  dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_BUFFER, NUM_EPOCHS)

  # dataset = dataset.map(parser_fn)  # Parse the record into tensors.
  # dataset = dataset.batch(32)
  dataset = dataset.apply(tf.contrib.data.map_and_batch(parser_fn, BATCH_SIZE, num_parallel_batches=4))
  dataset = dataset.prefetch(4)

  return dataset