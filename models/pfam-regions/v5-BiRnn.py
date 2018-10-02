r"""Entry point for trianing a RNN-based classifier for pfam regions dataset.

python models/v5-BiRnn.py \
  --training_data train_data \
  --eval_data eval_data \
  --checkpoint_dir ./checkpoints/ \
  --cell_type cudnn_lstm
  
python models/v5-BiRnn.py train \
  --dataset pfam_regions \
  --model_spec rnn_v1 \

python models/v5-BiRnn.py predict \
  --trained_model rnn_v1 \
  --input predict_data

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import ast
import glob
import json
import logging
import argparse
import datetime
import functools
from pathlib import Path

import colorama
from colorama import Fore, Back, Style
colorama.init() # this needs to run before first run of tf_logging._get_logger()
import tensorflow as tf
from tensorflow.python.ops import variables, inplace_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.contrib.data.python.ops.iterator_ops import _Saveable, _CustomSaver
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training import training_util
from tensorflow.python.framework import meta_graph
from tensorflow.python.data.util import nest
from tensorflow.python.util.nest import is_sequence
from tensorflow.contrib.layers.python.layers import adaptive_clipping_fn
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import tf_logging
import numpy as np
import coloredlogs
from tqdm import tqdm


class TqdmFile(object):
    """ A file-like object that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

# Disable cpp warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Show debugging output, default: tf.logging.INFO

logger = None
FLAGS = None
_NEG_INF = -1e9
aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

def pad_to_multiples(features, labels, pad_to_mutiples_of=8, padding_values=0):
    """Nvidia Volta Tensor Cores are enabled when data shape is multiples of 8
    """
    max_len = tf.shape(labels)[1]
    target_len = tf.cast(tf.multiply(tf.ceil(tf.truediv(max_len, pad_to_mutiples_of)), pad_to_mutiples_of), tf.int32)
    paddings = [[0, 0], [0, target_len - max_len]]
    features['protein'] = tf.pad(tensor=features['protein'], paddings=paddings, constant_values=padding_values)
    return features, tf.pad(tensor=labels, paddings=paddings, constant_values=padding_values)


def bucket_by_sequence_length_and_pad_to_multiples(element_length_func,
                                                   bucket_boundaries,
                                                   bucket_batch_sizes,
                                                   padded_shapes=None,
                                                   padding_values=None,
                                                   pad_to_mutiples_of=None,
                                                   pad_to_bucket_boundary=False):
    """A transformation that buckets elements in a `Dataset` by length.

    Nvidia Volta Tensor Cores are enabled when data shape is multiples of 8

    Elements of the `Dataset` are grouped together by length and then are padded
    and batched.

    This is useful for sequence tasks in which the elements have variable length.
    Grouping together elements that have similar lengths reduces the total
    fraction of padding in a batch which increases training step efficiency.

    Args:
      element_length_func: function from element in `Dataset` to `tf.int32`,
        determines the length of the element, which will determine the bucket it
        goes into.
      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
        `len(bucket_boundaries) + 1`.
      padded_shapes: Nested structure of `tf.TensorShape` to pass to
        @{tf.data.Dataset.padded_batch}. If not provided, will use
        `dataset.output_shapes`, which will result in variable length dimensions
        being padded out to the maximum length in each batch.
      padding_values: Values to pad with, passed to
        @{tf.data.Dataset.padded_batch}. Defaults to padding with 0.
      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
        size to maximum length in batch. If `True`, will pad dimensions with
        unknown size to bucket boundary, and caller must ensure that the source
        `Dataset` does not contain any elements with length longer than
        `max(bucket_boundaries)`.

    Returns:
      A `Dataset` transformation function, which can be passed to
      @{tf.data.Dataset.apply}.

    Raises:
      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
    """
    with tf.name_scope("bucket_by_sequence_length_and_pad_to_multiples"):
        if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
            raise ValueError(
                "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1")

        batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

        def element_to_bucket_id(*args):
            """Return int64 id of the length bucket for this element."""
            seq_length = element_length_func(*args)

            boundaries = list(bucket_boundaries)
            buckets_min = [np.iinfo(np.int32).min] + boundaries
            buckets_max = boundaries + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less_equal(buckets_min, seq_length),
                tf.less(seq_length, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))

            return bucket_id

        def window_size_fn(bucket_id):
            # The window size is set to the batch size for this bucket
            window_size = batch_sizes[bucket_id]
            return window_size

        def make_padded_shapes(shapes, none_filler=None):
            padded = []
            # print('shapes', shapes)
            for shape in nest.flatten(shapes):
                # print('shape', shape)
                shape = tf.TensorShape(shape)
                # print(tf.TensorShape(None))
                shape = [
                    none_filler if d.value is None else d
                    for d in shape
                ]
                # print(shape)
                padded.append(shape)
            return nest.pack_sequence_as(shapes, padded)

        def batching_fn(bucket_id, grouped_dataset):
            """Batch elements in dataset."""
            # ({'protein': TensorShape(None), 'lengths': TensorShape([])}, TensorShape(None))
            print(grouped_dataset.output_shapes)
            batch_size = batch_sizes[bucket_id]
            none_filler = None
            if pad_to_bucket_boundary:
                err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                           "length <= max(bucket_boundaries).")
                check = tf.assert_less(
                    bucket_id,
                    tf.constant(len(bucket_batch_sizes) - 1,
                                dtype=tf.int64),
                    message=err_msg)
                with tf.control_dependencies([check]):
                    boundaries = tf.constant(bucket_boundaries,
                                             dtype=tf.int64)
                    bucket_boundary = boundaries[bucket_id]
                    none_filler = bucket_boundary
            # print(padded_shapes or grouped_dataset.output_shapes)
            shapes = make_padded_shapes(
                padded_shapes or grouped_dataset.output_shapes,
                none_filler=none_filler)
            return grouped_dataset.padded_batch(batch_size, shapes, padding_values)

        def _apply_fn(dataset):
            return dataset.apply(
                tf.contrib.data.group_by_window(element_to_bucket_id, batching_fn,
                                                window_size_func=window_size_fn))

        return _apply_fn


def input_fn(mode, params, config):
    """Estimator `input_fn`.
    Args:
      mode: Specifies if training, evaluation or
            prediction. tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
      params: `dict` of hyperparameters.  Will receive what
              is passed to Estimator in `params` parameter. This allows
              to configure Estimators from hyper parameter tuning.
      config: configuration object. Will receive what is passed
              to Estimator in `config` parameter, or the default `config`.
              Allows updating things in your `model_fn` based on
              configuration such as `num_ps_replicas`, or `model_dir`.
    Returns:
      A 'tf.data.Dataset' object
    """
    # the file names will be shuffled randomly during training
    dataset = tf.data.TFRecordDataset.list_files(
        file_pattern=params.tfrecord_pattern[mode],
        # A string or scalar string `tf.Tensor`, representing
        # the filename pattern that will be matched.
        shuffle=mode == tf.estimator.ModeKeys.TRAIN
        # If `True`, the file names will be shuffled randomly.
        # Defaults to `True`.
    )

    # Apply the interleave, prefetch, and shuffle first to reduce memory usage.

    # Preprocesses params.dataset_parallel_reads files concurrently and interleaves records from each file.
    def tfrecord_dataset(filename):
        return tf.data.TFRecordDataset(
            filenames=filename,
            # containing one or more filenames
            compression_type=None,
            # one of `""` (no compression), `"ZLIB"`, or `"GZIP"`.
            buffer_size=params.dataset_buffer * 1024 * 1024
            # the number of bytes in the read buffer. 0 means no buffering.
        )  # 256 MB

    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        map_func=tfrecord_dataset,
        # A function mapping a nested structure of tensors to a Dataset
        cycle_length=params.dataset_parallel_reads,
        # The number of input Datasets to interleave from in parallel.
        block_length=1,
        # The number of consecutive elements to pull from an input
        # `Dataset` before advancing to the next input `Dataset`.
        sloppy=True,
        # If false, elements are produced in deterministic order. Otherwise,
        # the implementation is allowed, for the sake of expediency, to produce
        # elements in a non-deterministic order.
        buffer_output_elements=None,
        # The number of elements each iterator being
        # interleaved should buffer (similar to the `.prefetch()` transformation for
        # each interleaved iterator).
        prefetch_input_elements=None
        # The number of input elements to transform to
        # iterators before they are needed for interleaving.
    ))

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=params.shuffle_buffer,
            # the maximum number elements that will be buffered when prefetching.
            count=params.repeat_count
            # the number of times the dataset should be repeated
        ))

    def parse_sequence_example(serialized, mode):
        """Parse a single record which is expected to be a tensorflow.SequenceExample."""
        context_features = {
            # 'length': tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            'protein': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'domains': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
        }
        context, sequence = tf.parse_single_sequence_example(
            serialized=serialized,
            # A scalar (0-D Tensor) of type string, a single binary
            # serialized `SequenceExample` proto.
            context_features=context_features,
            # A `dict` mapping feature keys to `FixedLenFeature` or
            # `VarLenFeature` values. These features are associated with a
            # `SequenceExample` as a whole.
            sequence_features=sequence_features,
            # A `dict` mapping feature keys to
            # `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
            # associated with data within the `FeatureList` section of the
            # `SequenceExample` proto.
            example_name=None,
            # A scalar (0-D Tensor) of strings (optional), the name of
            # the serialized proto.
            name=None
            # A name for this operation (optional).
        )
        sequence['protein'] = tf.decode_raw(
            bytes=sequence['protein'],
            out_type=tf.uint8,
            little_endian=True,
            name=None
        )
        # tf.Tensor: shape=(sequence_length, 1), dtype=uint8
        sequence['protein'] = tf.cast(
            x=sequence['protein'],
            dtype=tf.int32,
            name=None
        )
        # embedding_lookup expects int32 or int64
        # tf.Tensor: shape=(sequence_length, 1), dtype=int32
        sequence['protein'] = tf.squeeze(
            input=sequence['protein'],
            axis=[],
            # An optional list of `ints`. Defaults to `[]`.
            # If specified, only squeezes the dimensions listed. The dimension
            # index starts at 0. It is an error to squeeze a dimension that is not 1.
            # Must be in the range `[-rank(input), rank(input))`.
            name=None
        )
        # tf.Tensor: shape=(sequence_length, ), dtype=int32
        # protein = tf.one_hot(protein, params.vocab_size)
        sequence['lengths'] = tf.shape(
            input=sequence['protein'],
            name=None,
            out_type=tf.int32
        )[0]
        domains = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            domains = tf.decode_raw(sequence['domains'], out_type=tf.uint16)
            # tf.Tensor: shape=(sequence_length, 1), dtype=uint16
            domains = tf.cast(domains, tf.int32)
            # sparse_softmax_cross_entropy_with_logits expects int32 or int64
            # tf.Tensor: shape=(sequence_length, 1), dtype=int32
            domains = tf.squeeze(domains, axis=[])
            # tf.Tensor: shape=(sequence_length, ), dtype=int32
            del sequence['domains']
            # domains = tf.one_hot(domains, params.num_classes)
        return sequence, domains

    dataset = dataset.map(
        functools.partial(parse_sequence_example, mode=mode),
        num_parallel_calls=int(params.num_cpu_threads / 2)
    )

    # # Our inputs are variable length, so pad them.
    # if mode != tf.estimator.ModeKeys.PREDICT:
    #     dataset = dataset.padded_batch(
    #         batch_size=params.batch_size,
    #         # A `tf.int64` scalar `tf.Tensor`, representing the number of
    #         # consecutive elements of this dataset to combine in a single batch.
    #         padded_shapes=({'protein': [None], 'lengths': []}, [None])
    #         # A nested structure of `tf.TensorShape` or
    #         # `tf.int64` vector tensor-like objects representing the shape
    #         # to which the respective component of each input element should
    #         # be padded prior to batching. Any unknown dimensions
    #         # (e.g. `tf.Dimension(None)` in a `tf.TensorShape` or `-1` in a
    #         # tensor-like object) will be padded to the maximum size of that
    #         # dimension in each batch.
    #     )
    # else:
    #     dataset = dataset.padded_batch(
    #         batch_size=params.batch_size,
    #         padded_shapes={'protein': [None], 'lengths': []}
    #     )

    # Our inputs are variable length, so bucket, dynamic batch and pad them.
    if mode != tf.estimator.ModeKeys.PREDICT:
        padded_shapes = ({'protein': [None], 'lengths': []}, [None])
    else:
        padded_shapes = {'protein': [None], 'lengths': []}

    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
        element_length_func=lambda seq, dom: seq['lengths'],
        bucket_boundaries=[2 ** x for x in range(5, 15)], # 32 ~ 16384
        bucket_batch_sizes=[params.batch_size * 2 ** x for x in range(10, -1, -1)], # 1024 ~ 1
        padded_shapes=padded_shapes,
        padding_values=None, # Defaults to padding with 0.
        pad_to_bucket_boundary=False
    )).map(
        functools.partial(pad_to_multiples, pad_to_mutiples_of=8, padding_values=0),
        num_parallel_calls=int(params.num_cpu_threads / 2)
    )
    
    # dataset = dataset.apply(bucket_by_sequence_length_and_pad_to_multiples(
    #     element_length_func=lambda seq, dom: seq['lengths'],
    #     bucket_boundaries=[2 ** x for x in range(5, 15)],  # 32 ~ 16384
    #     bucket_batch_sizes=[params.batch_size * 2 **
    #                         x for x in range(10, -1, -1)],  # 1024 ~ 1
    #     padded_shapes=padded_shapes,
    #     padding_values=None,  # Defaults to padding with 0.
    #     pad_to_mutiples_of=8,
    #     pad_to_bucket_boundary=False
    # ))

    dataset = dataset.prefetch(
        buffer_size=params.prefetch_buffer  # 64 batches
        # A `tf.int64` scalar `tf.Tensor`, representing the
        # maximum number batches that will be buffered when prefetching.
    )
    return dataset

    # iterator = dataset.make_one_shot_iterator()

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     # Build the iterator SaveableObject.
    #     saveable_obj = tf.contrib.data.make_saveable_from_iterator(iterator)
    #     # Add the SaveableObject to the SAVEABLE_OBJECTS collection so
    #     # it can be automatically saved using Saver.
    #     tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
    # tf.logging.debug('input_fn called')
    # return iterator.get_next()

class EpochCheckpointInputPipelineHookSaver(tf.train.Saver):
  """`Saver` with a different default `latest_filename`.

  This is used in the `CheckpointInputPipelineHook` to avoid conflicts with
  the model ckpt saved by the `CheckpointSaverHook`.
  """

  def __init__(self, 
               var_list, 
               latest_filename,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               defer_build=False,
               save_relative_paths=True):
    super(EpochCheckpointInputPipelineHookSaver, self).__init__(
        var_list,
        sharded=sharded,
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        defer_build=defer_build,
        save_relative_paths=save_relative_paths
    )
    self._latest_filename = latest_filename

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False):
    return super(EpochCheckpointInputPipelineHookSaver, self).save(
        sess, save_path, global_step, latest_filename or self._latest_filename,
        meta_graph_suffix, write_meta_graph, write_state, strip_default_attrs)

class EpochCheckpointInputPipelineHook(tf.train.SessionRunHook):
    """Checkpoints input pipeline state every N steps or seconds.

    This hook saves the state of the iterators in the `Graph` so that when
    training is resumed the input pipeline continues from where it left off.
    This could potentially avoid overfitting in certain pipelines where the
    number of training steps per eval are small compared to the dataset
    size or if the training pipeline is pre-empted.

    Differences from `CheckpointSaverHook`:
    1. Saves only the input pipelines in the "iterators" collection and not the
       global variables or other saveable objects.
    2. Does not write the `GraphDef` and `MetaGraphDef` to the summary.

    Example of checkpointing the training pipeline:

    ```python
    est = tf.estimator.Estimator(model_fn)
    while True:
      est.train(
          train_input_fn,
          hooks=[tf.contrib.data.CheckpointInputPipelineHook(est)],
          steps=train_steps_per_eval)
      # Note: We do not pass the hook here.
      metrics = est.evaluate(eval_input_fn)
      if should_stop_the_training(metrics):
        break
    ```

    This hook should be used if the input pipeline state needs to be saved
    separate from the model checkpoint. Doing so may be useful for a few reasons:
    1. The input pipeline checkpoint may be large, if there are large shuffle
       or prefetch buffers for instance, and may bloat the checkpoint size.
    2. If the input pipeline is shared between training and validation, restoring
       the checkpoint during validation may override the validation input
       pipeline.

    For saving the input pipeline checkpoint alongside the model weights use
    @{tf.contrib.data.make_saveable_from_iterator} directly to create a
    `SaveableObject` and add to the `SAVEABLE_OBJECTS` collection. Note, however,
    that you will need to be careful not to restore the training iterator during
    eval. You can do that by not adding the iterator to the SAVEABLE_OBJECTS
    collector when building the eval graph.
    """

    def __init__(self,
                checkpoint_dir,
                config,
                save_timer=None,
                save_secs=None,
                save_steps=None,
                checkpoint_basename="input",
                listeners=None, 
                defer_build=False,
                save_relative_paths=True):
        """Initializes a `EpochCheckpointInputPipelineHook`.
        Creates a custom EpochCheckpointInputPipelineHookSaver

        Args:
            checkpoint_dir: `str`, base directory for the checkpoint files.
        save_timer: `SecondOrStepTimer`, timer to save checkpoints.
            save_secs: `int`, save every N secs.
            save_steps: `int`, save every N steps.
            checkpoint_basename: `str`, base name for the checkpoint files.
            listeners: List of `CheckpointSaverListener` subclass instances.
                Used for callbacks that run immediately before or after this hook saves
                the checkpoint.
            config: tf.estimator.RunConfig.

        Raises:
            ValueError: One of `save_steps` or `save_secs` should be set.
            ValueError: At most one of saver or scaffold should be set.
        """
        # `checkpoint_basename` is "input.ckpt" for non-distributed pipelines or
        # of the form "input_<task_type>_<task_id>.ckpt" for distributed pipelines.
        # Note: The default `checkpoint_basename` used by `CheckpointSaverHook` is
        # "model.ckpt". We intentionally choose the input pipeline checkpoint prefix
        # to be different to avoid conflicts with the model checkpoint.

        # pylint: disable=protected-access
        tf.logging.info("Create EpochCheckpointInputPipelineHook.")
        self._checkpoint_dir = checkpoint_dir
        self._config = config
        self._defer_build = defer_build
        self._save_relative_paths = save_relative_paths

        self._checkpoint_prefix = checkpoint_basename
        if self._config.num_worker_replicas > 1:
            # Distributed setting.
            suffix = "_{}_{}".format(self._config.task_type,
                                     self._config.task_id)
            self._checkpoint_prefix += suffix
        # pylint: enable=protected-access

        # We use a composition paradigm instead of inheriting from
        # `CheckpointSaverHook` because `Estimator` does an `isinstance` check
        # to check whether a `CheckpointSaverHook` is already present in the list
        # of hooks and if not, adds one. Inheriting from `CheckpointSaverHook`
        # would thwart this behavior. This hook checkpoints *only the iterators*
        # and not the graph variables.
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        if save_timer:
            self._timer = save_timer
        else:
            self._timer = tf.train.SecondOrStepTimer(
                every_secs=save_secs,
                every_steps=save_steps
            )
        self._listeners = listeners or []
        self._steps_per_run = 1

        # Name for the protocol buffer file that will contain the list of most
        # recent checkpoints stored as a `CheckpointState` protocol buffer.
        # This file, kept in the same directory as the checkpoint files, is
        # automatically managed by the `Saver` to keep track of recent checkpoints.
        # The default name used by the `Saver` for this file is "checkpoint". Here
        # we use the name "checkpoint_<checkpoint_prefix>" so that in case the
        # `checkpoint_dir` is the same as the model checkpoint directory, there are
        # no conflicts during restore.
        self._latest_filename = self._checkpoint_prefix + '.latest'
        self._first_run = True

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        # Build a Saver that saves all iterators in the `GLOBAL_ITERATORS`
        # collection 
        iterators = tf.get_collection(iterator_ops.GLOBAL_ITERATORS)
        saveables = [_Saveable(i) for i in iterators]
        self._saver = EpochCheckpointInputPipelineHookSaver(
            saveables,
            self._latest_filename,
            sharded=False,
            max_to_keep=self._config.keep_checkpoint_max,
            keep_checkpoint_every_n_hours=self._config.keep_checkpoint_every_n_hours,
            defer_build=self._defer_build,
            save_relative_paths=self._save_relative_paths
        )
            
        self._summary_writer = tf.summary.FileWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EpochCheckpointInputPipelineHook.")
        for l in self._listeners:
            l.begin()

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        global_step = session.run(self._global_step_tensor)
        self._timer.update_last_triggered_step(global_step)
        
    def _maybe_restore_input_ckpt(self, session):
        # Ideally this should be run in after_create_session but is not for the
        # following reason:
        # Currently there is no way of enforcing an order of running the
        # `SessionRunHooks`. Hence it is possible that the `_DatasetInitializerHook`
        # is run *after* this hook. That is troublesome because
        # 1. If a checkpoint exists and this hook restores it, the initializer hook
        #    will override it.
        # 2. If no checkpoint exists, this hook will try to save an initialized
        #    iterator which will result in an exception.
        #
        # As a temporary fix we enter the following implicit contract between this
        # hook and the _DatasetInitializerHook.
        # 1. The _DatasetInitializerHook initializes the iterator in the call to
        #    after_create_session.
        # 2. This hook saves the iterator on the first call to `before_run()`, which
        #    is guaranteed to happen after `after_create_session()` of all hooks
        #    have been run.

        # Check if there is an existing checkpoint. If so, restore from it.
        # pylint: disable=protected-access
        latest_checkpoint_path = tf.train.latest_checkpoint(
            self._checkpoint_dir,
            latest_filename=self._latest_filename)
        if latest_checkpoint_path:
            self._get_saver().restore(session, latest_checkpoint_path)

    def before_run(self, run_context):
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        if self._first_run:
            self._maybe_restore_input_ckpt(run_context.session)
            self._first_run = False
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
            
        # delete latest checkpoint file
        input_checkpoint_files = Path(self._checkpoint_dir).glob(self._checkpoint_prefix + '*')
        # print(input_checkpoint_files)
        for f in input_checkpoint_files:
            if f.exists():
                f.unlink()
                # print('DELETE: ', f)
        tf.logging.debug("Removed input checkpoints")

        last_step = session.run(self._global_step_tensor)
        for l in self._listeners:
            l.end(session, last_step)


    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf.logging.info("Saving\033[31m input\033[0m checkpoints for %d into %s.", step, self._save_path)

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step)
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
            step)

        should_stop = False
        for l in self._listeners:
            if l.after_save(session, step):
                tf.logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop

    def _get_saver(self):
        return self._saver

class EpochCheckpointSaverHook(tf.train.CheckpointSaverHook):
    """This checkpoint saver hook saves two types of checkpoints:

    1. step: 
    * Saves on save_secs or save_steps
    * Does not save on begin or end
    * Saves input pipeline state to continue training the remaining examples in the current epoch
    * Separately configurable garbage collection criteria from epoch
        * Defaults: max_to_keep=10, keep_checkpoint_every_n_hours=6
    * The default list of CheckpointSaverListener does not run on step checkpoint saves,
      you may configure a separate list of CheckpointSaverListeners by setting the step_listeners init arg
    * filename = step
    * latest_filename = step.latest
    
    2. epoch:
    * Does not save on save_secs or save_steps
    * Saves on epoch end
    * Does not save input pipeline
    * Separately configurable garbage collection criteria from step
        * Does not garbage collect by default
            * Defaults: max_to_keep=9999, keep_checkpoint_every_n_hours=999999
        * set epoch_saver to a custom tf.train.Saver to change defaults
    * The default list of CheckpointSaverListener only runs on epoch checkpoint saves,
      this includes the default _NewCheckpointListenerForEvaluate added by tf.estimator.train_and_evaluate
      which runs the eval loop after every new checkpoint
    * filename = epoch
    * latest_filename = epoch.latest
    
    Usage:
    * Added to the list of EstimatorSpec.training_chief_hooks in your model_fn.
      * This prevents the default CheckpointSaverHook from being added
    * The end of an "epoch" is defined as the input_fn raising the OutOfRangeError,
      don't repeat the dataset or set the repeat_count to 1 if you want to "expected" behavior of
      one "epoch" is one iteration over all of the training data.
    * estimator.train or tf.estimator.train_and_evaluate will exit after the OutOfRangeError,
      wrap it with a for loop to train a limited number of epochs or a while True loop to train forever.

    Fixes more than one graph event per run warning in Tensorboard
    """

    def __init__(self,
                checkpoint_dir,
                epoch_tensor=None,
                save_timer=None,
                save_secs=None,
                save_steps=None,
                saver=None,
                checkpoint_basename=None,
                scaffold=None,
                listeners=None,
                step_listeners=None,
                epoch_saver=None,
                epoch_basename='epoch',
                step_basename='step',
                epoch_latest_filename='epoch.latest',
                step_latest_filename='step.latest'):
        """Maintains compatibility with the `CheckpointSaverHook`.

        Args:
        checkpoint_dir: `str`, base directory for the checkpoint files.
        save_timer: `SecondOrStepTimer`, timer to save checkpoints.
        save_secs: `int`, save a step checkpoint every N secs.
        save_steps: `int`, save a step checkpoint every N steps.
        saver: `Saver` object, used for saving a final step checkpoint.
        checkpoint_basename: `str`, base name for the checkpoint files.
        scaffold: `Scaffold`, use to get saver object a final step checkpoint.
        listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            a epoch checkpoint.
        step_listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            a step checkpoint.
        epoch_saver: `Saver` object, used for saving a epoch checkpoint.
        step_basename: `str`, base name for the step checkpoint files.
        epoch_basename: `str`, base name for the epoch checkpoint files.

        Raises:
        ValueError: One of `save_steps` or `save_secs` should be set.
        ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EpochCheckpointSaverHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        checkpoint_basename = checkpoint_basename or ''
        epoch_basename = ''.join((checkpoint_basename, epoch_basename or 'step'))
        step_basename = ''.join((checkpoint_basename, step_basename or 'step'))
        self._epoch_save_path = os.path.join(checkpoint_dir, epoch_basename)
        self._step_save_path = os.path.join(checkpoint_dir, step_basename)
        self._epoch_latest_filename = epoch_latest_filename or 'epoch.latest'
        self._step_latest_filename = step_latest_filename or 'step.latest'
        self._scaffold = scaffold
        if save_timer:
            self._timer = save_timer
        else:
            self._timer = tf.train.SecondOrStepTimer(
                every_secs=save_secs,
                every_steps=save_steps
            )
        self._epoch_listeners = listeners or []
        # In _train_with_estimator_spec
        # saver_hooks[0]._listeners.extend(saving_listeners)
        self._listeners = self._epoch_listeners
        self._step_listeners = step_listeners or []
        self._epoch_saver = epoch_saver
        self._steps_per_run = 1
        self._epoch_tensor = epoch_tensor

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        self._summary_writer = tf.summary.FileWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EpochCheckpointSaverHook.")
                
        if self._epoch_saver is None:
            self._epoch_saver = tf.train.Saver(
                sharded=False,
                max_to_keep=9999,
                keep_checkpoint_every_n_hours=999999,
                defer_build=False,
                save_relative_paths=True
            )

        for l in self._epoch_listeners:
            l.begin()
        for l in self._step_listeners:
            l.begin()

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        global_step = session.run(self._global_step_tensor)
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        tf.train.write_graph(
            tf.get_default_graph().as_graph_def(add_shapes=True),
            self._checkpoint_dir,
            "graph.pbtxt"
        )
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        graph = tf.get_default_graph()
        meta_graph_def = meta_graph.create_meta_graph_def(
            graph_def=graph.as_graph_def(add_shapes=True),
            saver_def=saver_def)
        self._summary_writer.add_graph(graph, global_step=global_step)
        self._summary_writer.add_meta_graph(meta_graph_def, global_step=global_step)
        # The checkpoint saved here is the state at step "global_step".
        # do not save any checkpoints at start
        # self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A SessionRunValues object.
        """
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save_step(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
        # savables = tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        # savables_ref = tf.get_collection_ref(tf.GraphKeys.SAVEABLE_OBJECTS)
        # print('SAVEABLE_OBJECTS before', len(savables_ref), savables_ref)
        # # remove tensorflow.contrib.data.python.ops.iterator_ops._Saveable object
        # for v in savables:
        #     if isinstance(v, _Saveable):
        #         savables_ref.remove(v)
        # print('SAVEABLE_OBJECTS after', len(savables_ref), savables_ref)

        last_step = session.run(self._global_step_tensor)
        epoch = None
        if self._epoch_tensor is not None:
            epoch = session.run(self._epoch_tensor)

        if last_step != self._timer.last_triggered_step():
            self._save_step(session, last_step)
        
        self._save_epoch(session, last_step, epoch)
        
        for l in self._epoch_listeners:
            # _NewCheckpointListenerForEvaluate will run here at end
            l.end(session, last_step)
        
        for l in self._step_listeners:
            l.end(session, last_step)


    def _save_epoch(self, session, step, epoch):
        """Saves the latest checkpoint, returns should_stop."""
        if epoch:
            save_path = '{}-{}'.format(self._epoch_save_path, epoch)
        else:
            save_path = self._epoch_save_path
        tf.logging.info("Saving\033[1;31m epoch\033[0m checkpoints for %d into %s.", step, save_path)

        for l in self._epoch_listeners:
            l.before_save(session, step)

        self._get_epoch_saver().save(
            sess=session, 
            save_path=save_path, 
            global_step=step,
            latest_filename=self._epoch_latest_filename,
            meta_graph_suffix="meta",
            write_meta_graph=True,
            write_state=True,
            strip_default_attrs=False
        )

        should_stop = False
        for l in self._epoch_listeners:
            # _NewCheckpointListenerForEvaluate will not run here 
            # since _is_first_run == True, it will run at end
            if l.after_save(session, step):
                tf.logging.info(
                    "An Epoch CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop

    def _save_step(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf.logging.info("Saving\033[1;31m step\033[0m checkpoints for %d into %s.", step, self._step_save_path)

        for l in self._step_listeners:
            l.before_save(session, step)
        
        saver = self._get_step_saver()
        
        saver.save(
            sess=session, 
            save_path=self._step_save_path, 
            global_step=step,
            # latest_filename=self._step_latest_filename,
            latest_filename=None,
            meta_graph_suffix="meta",
            write_meta_graph=True,
            write_state=True,
            strip_default_attrs=False
        )
        self._summary_writer.add_session_log(
            SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=self._step_save_path),
            step
        )

        should_stop = False
        for l in self._step_listeners:
            if l.after_save(session, step):
                tf.logging.info(
                    "A Step CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
        return should_stop

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        return self._save_step(session, step)

    def _get_epoch_saver(self):
        return self._epoch_saver

    def _get_step_saver(self):
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver

        # Get saver from the SAVERS collection if present.
        collection_key = tf.GraphKeys.SAVERS
        savers = tf.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor.".
                format(collection_key))

        self._saver = savers[0]
        return savers[0]

    def _get_saver(self):
        return self._get_step_saver()

orig_stdout = sys.stdout
class EpochProgressBarHook(tf.train.SessionRunHook):
    def __init__(self,
                 total,
                 initial_tensor,
                 n_tensor,
                 postfix_tensors=None,
                 every_n_iter=None):
        self._total = total
        self._initial_tensor = initial_tensor
        self._n_tensor = n_tensor
        self._postfix_tensors = postfix_tensors
        self._every_n_iter = every_n_iter
    
    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        pass
        
    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        initial = session.run(self._initial_tensor)
        epoch = initial // self._total
        epoch_initial = initial % self._total
        # print('after_create_session', initial, epoch)
        # setup progressbar
        self.pbar = tqdm(
            total=self._total,
            unit='seq',
            desc='Epoch {}'.format(epoch),
            mininterval=0.1,
            maxinterval=10.0,
            miniters=None,
            file=orig_stdout,
            dynamic_ncols=True,
            smoothing=0,
            bar_format=None,
            initial=epoch_initial,
            postfix=None
        )
        
    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        return tf.train.SessionRunArgs(self._n_tensor)
        
    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A SessionRunValues object.
        """
        # print('run_values', run_values.results)
        # update progressbar
        self.pbar.update(run_values.results)

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
        self.pbar.close()

class ColoredLoggingTensorHook(tf.train.LoggingTensorHook):
  """Prints the given tensors every N local steps, every N seconds, or at end.

  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:

  ```python
    tf.logging.set_verbosity(tf.logging.INFO)
  ```

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.
  """
  def _log_tensors(self, tensor_values):
    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._formatter:
      if elapsed_secs is not None:
        tf.logging.info("%s (%.3f sec)", self._formatter(tensor_values), elapsed_secs)
      else:
        tf.logging.info(self._formatter(tensor_values))
    else:
      stats = []
      for tag in self._tag_order:
        stats.append("%s = %s" % (tag, tensor_values[tag]))
      if elapsed_secs is not None:
        tf.logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
      else:
        tf.logging.info("%s", ", ".join(stats))
    np.set_printoptions(**original)


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
        kwargs
    return variable


def softmax_mask(val, mask, inf=1e30):
    return -inf * (1 - tf.cast(mask, tf.float32)) + val


def dot_attention(inputs, memory, mask, units, 
    drop_inputs=0.0, drop_memory=0.0, drop_res=0.2, 
    is_train=None, scope="dot_attention"):
    """
    memory (key): A sequence of vectors also known as the memory. It is the contextual information that we want to look at. In traditional sequence-to-sequence learning they are usually the RNN encoder outputs.
    Values: A sequence of vectors from which we aggregate the output through a weighted linear combination. Often Keys serve as Values.
    Query: A single vector that we use to probe the Keys. By probing we mean the Query is independently combined with each key to arrive at a single probability. The type of attention determines how the combination is done. Usually Query is the decoder RNN state at a given time step in traditional sequence-to-sequence learning
    Output: A single vector which is derived from a linear combination of the Values using the probabilities from the previous step as weights.
    """
    with tf.variable_scope(scope):

        d_inputs = tf.layers.dropout(
            inputs=inputs,
            rate=drop_inputs,  # 0
            noise_shape=None,  # [batch_size, 1, embed_dim]
            training=is_train,
            name='drop_inputs'
        )
        d_memory = tf.layers.dropout(
            inputs=memory,
            rate=drop_memory,  # 0
            noise_shape=None,  # [batch_size, 1, embed_dim]
            training=is_train,
            name='drop_memory'
        )
        sequence_length = tf.shape(inputs)[1] # sequence_length

        inputs_ = tf.layers.dense(
            inputs=d_inputs,
            units=units,
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name='inputs',
            reuse=None
        )
        memory_ = tf.layers.dense(
            inputs=d_memory,
            units=units,
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name='memory',
            reuse=None
        )
        outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (units ** 0.5)
        # mask 
        #   shape=(batch_size, sequence_length), dtype=float32
        # mask.expand_dims 
        #   shape=(batch_size, 1, sequence_length), dtype=float32
        # mask.expand_dims.tile 
        #   shape=(batch_size, sequence_length, sequence_length), dtype=float32
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, sequence_length, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        outputs = tf.matmul(logits, memory)
        res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = tf.layers.dropout(
                inputs=res,
                rate=drop_res,  # 0
                noise_shape=None,  # [batch_size, 1, embed_dim]
                training=is_train,
                name='drop_res'
            )
            gate = tf.layers.dense(
                inputs=d_res,
                units=dim,
                activation=tf.nn.sigmoid,
                use_bias=False,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name='memory',
                reuse=None
            )
            return res * gate


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
    self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
    self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5
    # a = tf.constant()
    # Calculate dot product attention
    #logits = tf.matmul(q, k, transpose_b=True)
    #logits += bias
    #weights = tf.nn.softmax(logits, name="attention_weights")
    logits = tf.matmul(q, k, transpose_b=True)
    # [batch_size, heads, length, units] * [batch_size, heads, units, length]
    # logits shape = [batch_size, heads, length, length]
    dtype = logits.dtype
    if dtype != tf.float32:
      # upcast softmax inputs
      logits = tf.cast(x=logits, dtype=tf.float32)
      logits += bias
      weights = tf.nn.softmax(logits, name="attention_weights")
      # downcast softmax output
      weights = tf.cast(weights, dtype=dtype)
    else:
      logits += bias
      # bias shape = [batch_size, 1, 1, length]
      weights = tf.nn.softmax(logits, name="attention_weights")


    if self.train:
      weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


def get_padding(x, padding_value=0, dtype=tf.float32):
  """Return float tensor representing the padding values in x.
  Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: type of the output
  Returns:
    flaot tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype=dtype)


def get_padding_bias(x, res_rank=4, pad_sym=0):
  """Calculate bias tensor from padding values in tensor.
  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  Args:
    x: int tensor with shape [batch_size, length]
    res_rank: int indicates the rank of attention_bias.
    dtype: type of the output attention_bias
    pad_sym: int the symbol used for padding
  Returns:
    Attention bias tensor of shape
    [batch_size, 1, 1, length] if  res_rank = 4 - for Transformer
    or [batch_size, 1, length] if res_rank = 3 - for ConvS2S
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x, padding_value=pad_sym)
    attention_bias = padding * _NEG_INF
    if res_rank == 4:
      attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
    elif res_rank == 3:
      attention_bias = tf.expand_dims(attention_bias, axis=1)
    else:
      raise ValueError("res_rank should be 3 or 4 but got {}".format(res_rank))
  return attention_bias
  

class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)

# tensor2tensor start

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                       x.device, cast_x.device)
  return cast_x

def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.

  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".

  Returns:
    Tensor of the same shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, keep_prob, **kwargs)

def should_generate_summaries():
  """Is this an appropriate context to generate summaries.

  Returns:
    a boolean
  """
  name_scope = tf.contrib.framework.get_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True

def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])

def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.

  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
  attn = tf.cast(attn, tf.float32)
  num_heads = shape_list(attn)[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    if len(image_shapes) == 4:
      q_rows, q_cols, m_rows, m_cols = list(image_shapes)
      image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
      image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
      image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
    else:
      assert len(image_shapes) == 6
      q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
          image_shapes)
      image = tf.reshape(
          image,
          [-1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3])
      image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
      image = tf.reshape(
          image,
          [-1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3])
  tf.summary.image("attention", image, max_outputs=1)

def reshape_by_blocks(x, x_shape, memory_block_size):
  """Reshapes input by splitting its length over blocks of memory_block_size.

  Args:
    x: a Tensor with shape [batch, heads, length, depth]
    x_shape: tf.TensorShape of x.
    memory_block_size: Integer which divides length.

  Returns:
    Tensor with shape
    [batch, heads, length // memory_block_size, memory_block_size, depth].
  """
  x = tf.reshape(x, [
      x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
      memory_block_size, x_shape[3]
  ])
  return x

def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.

  We have hacked symbol_modality to return all-zero embeddings for padding.

  Args:
    emb: a Tensor with shape [..., depth].

  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
    embedding vector is all zero, and is 0 otherwise.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.0))

def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])

def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])
  
def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0):
  """Computes attention compoenent (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)

  Returns:
    c : [batch, length, depth] tensor
  """
  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.get_variable(
        name, [input_depth,
               vars_3d_num_heads,
               total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return tf.layers.dense(
        antecedent, total_depth, use_bias=False, name=name)
  else:
    return tf.nn.conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)

def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads)
  return q, k, v

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = cast_like(bias, logits)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)

def local_attention_1d(q, k, v, block_length=128, filter_width=100, name=None):
  """Strided block local self-attention.

  The sequence is divided into blocks of length block_length. Attention for a
  given query position can see all memory positions in the corresponding block
  and filter_width many positions to the left and right of the block.

  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    filter_width: an integer indicating how much to look left.
    name: an optional string

  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  with tf.variable_scope(
      name, default_name="local_self_attention_1d", values=[q, k, v]):
    batch_size, num_heads, original_length, _ = shape_list(q)
    depth_v = shape_list(v)[-1]

    # Pad query, key, value to ensure multiple of corresponding lengths.
    def pad_to_multiple(x, pad_length):
      x_length = shape_list(x)[2]
      return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

    def pad_l_and_r(x, pad_length):
      return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

    q = pad_to_multiple(q, block_length)
    k = pad_to_multiple(k, block_length)
    v = pad_to_multiple(v, block_length)

    # Set up query blocks.
    new_q_shape = shape_list(q)
    # q: [batch, heads, length // memory_block_size, memory_block_size, depth]
    q = reshape_by_blocks(q, new_q_shape, block_length)

    # Set up key and value blocks.
    # Get gather indices.
    k = pad_l_and_r(k, filter_width)
    v = pad_l_and_r(v, filter_width)
    length = shape_list(k)[2]
    full_filter_width = block_length + 2 * filter_width
    indices = tf.range(0, length, delta=1, name="index_range")
    indices = tf.reshape(indices, [1, -1, 1])  # [1, length, 1] for convs
    kernel = tf.expand_dims(tf.eye(full_filter_width), axis=1)
    gather_indices = tf.nn.conv1d(
        tf.cast(indices, tf.float32),
        kernel,
        block_length,
        padding="VALID",
        name="gather_conv")

    gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

    # Reshape keys and values to [length, batch, heads, dim] for gather. Then
    # reshape to [batch, heads, blocks, block_length + filter_width, dim].
    k_t = tf.transpose(k, [2, 0, 1, 3])
    k_new = tf.gather(k_t, gather_indices)
    k_new = tf.transpose(k_new, [2, 3, 0, 1, 4])

    attention_bias = tf.expand_dims(embedding_to_padding(k_new) * -1e9, axis=-2)

    v_t = tf.transpose(v, [2, 0, 1, 3])
    v_new = tf.gather(v_t, gather_indices)
    v_new = tf.transpose(v_new, [2, 3, 0, 1, 4])

    output = dot_product_attention(
        q, # [batch, heads, blocks, memory_block_size, depth_k]
        k_new, # [batch, heads, blocks, block_length + 2*filter_width, depth_k]
        v_new, # [batch, heads, blocks, block_length + 2*filter_width, depth_v]
        attention_bias,
        dropout_rate=0.,
        name="local_1d",
        make_image_summary=False)
    # output: [batch, heads, blocks, memory_block_size, depth_v]
    output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

    # Remove the padding if introduced.
    output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
    output.set_shape([None if isinstance(dim, tf.Tensor) else dim for dim in
                      (batch_size, num_heads, original_length, depth_v)])
    return output

def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        shared_rel=False,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        filter_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        max_length=None,
                        vars_3d=False,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    shared_rel: boolean to share relative embeddings
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    block_length: an integer - relevant for "local_mask_right"
    filter_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    max_length: an integer - needed by relative attention
    vars_3d: use 3-dimensional variables for input/output transformations
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0
  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads)
    if cache is not None:
      if attention_type != "dot_product":
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    # [batch, num_heads, length, channels / num_heads]
    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                save_weights_to=save_weights_to,
                                make_image_summary=make_image_summary,
                                dropout_broadcast_dims=dropout_broadcast_dims)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=filter_width)
    x = combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = tf.layers.dense(
          x, output_depth, use_bias=False, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * _NEG_INF
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

# tensor2tensor end

# tensor2tensor Normalization start
def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable("scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable("bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]

    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def group_norm(x, filters=None, num_groups=8, epsilon=1e-5, name=None):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""
  x_shape = shape_list(x)
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4
  assert filters % num_groups == 0
  # Prepare variables.
  with tf.variable_scope(name, default_name="group_norm", values=[x]):
    scale = tf.get_variable(
        "scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    # Reshape and compute group norm.
    x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
    # Calculate mean and variance on heights, width, channels (not groups).
    mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return tf.reshape(norm_x, x_shape) * scale + bias

def noam_norm(x, epsilon=1.0, name=None):
  """One version of layer normalization."""
  with tf.variable_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        tf.to_float(shape[-1])))

def l2_layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalization with l2 norm."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(name, default_name="l2_layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    l2norm = tf.reduce_sum(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(l2norm + epsilon)
    return norm_x * scale + bias

def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return tf.layers.batch_normalization(x, epsilon=epsilon)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "l2layer":
    return l2_layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")
# tensor2tensor Normalization end

# Correct Layer Norm LSTM start
def _norm(g, b, inp, scope, center=True):
  """layer normalize helper:
  Args:
    g: float, layer normalization scale (gamma) initial value.
    b: float, layer normalization offset (beta) initial value.
    scope: Optional scope for `variable_scope`.
    center: boolean. If True (default), add layer normalization offset (beta)
      after normalization.
  """
  shape = inp.get_shape()[-1:]
  gamma_init = tf.constant_initializer(g)
  beta_init = tf.constant_initializer(b)
  with tf.variable_scope(scope):
    tf.get_variable("gamma", shape=shape, initializer=gamma_init)
    if center:
      tf.get_variable("beta", shape=shape, initializer=beta_init)
  normalized = tf.contrib.layers.layer_norm(inp, center=center, reuse=True, scope=scope)
  return normalized

class LayerNormLSTMCellv2(tf.nn.rnn_cell.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.
  The default non-peephole implementation is based on:
    http://www.bioinf.jku.at/publications/older/2604.pdf
  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
  The peephole implementation is based on:
    https://research.google.com/pubs/archive/43905.pdf
  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.
  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  Layer normalization implementation is based on:
    https://arxiv.org/abs/1607.06450.
  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  and is applied before the internal nonlinearities.
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               forget_bias=1.0,
               activation=None,
               layer_norm=True,
               layer_norm_columns=False,
               norm_gain=1.0,
               norm_shift=0.0,
               reuse=None):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      activation: Activation function of the inner states.  Default: `tanh`.
      layer_norm: If `True`, layer normalization will be applied.
      layer_norm_columns: If `True`, layer normalization will be applied
        separately to columns of the linear transformation of the inputs and
        recurrent states. If `False`, normalization will be applied as
        described in the paper "Layer Normalization" by Ba et. al.
        `False` is the recommended setting. `True` is the default for backwards
        compatibility. If `layer_norm` is `False`, this argument is ignored.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(LayerNormLSTMCellv2, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    self._activation = activation or tf.tanh
    self._layer_norm = layer_norm
    self._layer_norm_columns = layer_norm_columns
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift

    if num_proj:
      self._state_size = (tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj))
      self._output_size = num_proj
    else:
      self._state_size = (tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units))
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _linear(self,
              args,
              output_size,
              bias,
              bias_initializer=None,
              kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a Variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] taking value
      sum_i(args[i] * W[i]), where each W[i] is a newly created Variable,
      and where args[i] * W[i] will be layer normalized if `layer_norm_columns`
      is False.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not is_sequence(args):
      args = [args]
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      kernel = tf.get_variable(
          "kernel", [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if self._layer_norm and not self._layer_norm_columns:
        axis_sizes = [shape[1].value for shape in shapes]
        weights = tf.split(kernel, axis_sizes)
        res = [tf.matmul(arg, w) for arg, w in zip(args, weights)]
        # Don't add layer norm offset because we add a bias.
        res = [_norm(self._norm_gain, self._norm_shift, out,
                     str(i), center=False) for i, out in enumerate(res)]
        res = sum(res)
      else:
        res = tf.matmul(tf.concat(args, 1), kernel)
      if not bias or (self._layer_norm and self._layer_norm_columns):
        return res
      with tf.variable_scope(outer_scope) as inner_scope:
        inner_scope.set_partitioner(None)
        if bias_initializer is None:
          bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
        biases = tf.get_variable(
            "bias", [output_size], dtype=dtype, initializer=bias_initializer)

    res = tf.nn.bias_add(res, biases)

    return res

  def call(self, inputs, state):
    """Run one step of LSTM.
    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: this must be a tuple of state Tensors,
       both `2-D`, with column sizes `c_state` and
        `m_state`.
    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.
    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = tf.sigmoid

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, initializer=self._initializer) as unit_scope:

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = self._linear(
          [inputs, m_prev],
          4 * self._num_units,
          bias=True,
          bias_initializer=None)
      i, j, f, o = tf.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      if self._layer_norm and self._layer_norm_columns:
        i = _norm(self._norm_gain, self._norm_shift, i, "input")
        j = _norm(self._norm_gain, self._norm_shift, j, "transform")
        f = _norm(self._norm_gain, self._norm_shift, f, "forget")
        o = _norm(self._norm_gain, self._norm_shift, o, "output")

      # Diagonal connections
      if self._use_peepholes:
        with tf.variable_scope(unit_scope):
          w_f_diag = tf.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = tf.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = tf.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (
            sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
            sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (
            sigmoid(f + self._forget_bias) * c_prev +
            sigmoid(i) * self._activation(j))

      if self._layer_norm:
        c = _norm(self._norm_gain, self._norm_shift, c, "state")

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with tf.variable_scope("projection"):
          m = self._linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = tf.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m))
    return m, new_state
# Correct Layer Norm LSTM end

def model_fn(features, labels, mode, params, config):
    # labels shape=(batch_size, sequence_length), dtype=int32
    is_train = mode == tf.estimator.ModeKeys.TRAIN

    protein = features['protein']
    # protein shape=(batch_size, sequence_length), dtype=int32
    lengths = features['lengths']
    # lengths shape=(batch_size, ), dtype=int32
    global_step = tf.train.get_global_step()
    # global_step is assign_add 1 in tf.train.Optimizer.apply_gradients
    batch_size = tf.shape(lengths)[0]
    # number of sequences per epoch
    seq_total = batch_size
    if mode == tf.estimator.ModeKeys.TRAIN:
        seq_total = params.metadata['train']['seq_count']['total']
    elif mode == tf.estimator.ModeKeys.EVAL:
        seq_total = params.metadata['test']['seq_count']['total']

    if params.use_tensor_ops:
        float_type = tf.float16
    else:
        float_type = tf.float32

    with tf.name_scope("padding"):
        padding_value = 0
        padding = tf.cast(tf.equal(labels, padding_value), dtype=tf.float32)
        mask = tf.cast(tf.sign(labels), dtype=tf.float32)  # 0 = 'PAD'
    # mask shape=(batch_size, sequence_length), dtype=float32

    # attention_layers = {
    #     'rnet': 1,
    #     'seq2seq': tf.contrib.cudnn_rnn.CudnnGRU
    # }
    # dot_attention(
    #             inputs=convolved,
    #             memory=convolved,
    #             mask=mask,
    #             units=params.conv_1_attention_units,
    #             drop_inputs=0.0,
    #             drop_memory=0.0,
    #             drop_res=params.conv_1_attention_drop_res, # 0.2
    #             is_train=is_train,
    #             scope="attention"
    #         )

    # Embedding layer
    with tf.variable_scope('embedding_1', values=[features]):
        embeddings = tf.contrib.framework.model_variable(
            name='embeddings',
            shape=[params.vocab_size, params.embed_dim],
            dtype=float_type,  # default: tf.float32
            # initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            initializer=tf.random_uniform_initializer(
                minval=-0.5,
                maxval=0.5,
                dtype=float_type
            ),
            trainable=True,
        )  # vocab_size * embed_dim = 28 * 32 = 896
        # tf.Variable 'embedding_matrix:0' shape=(vocab_size, embed_dim) dtype=float32
        embedded = tf.nn.embedding_lookup(
            params=embeddings,
            ids=protein,
            name='embedding_lookup'
        )
        # tf.Tensor: shape=(batch_size, sequence_length, embed_dim), dtype=float32
        dropped_embedded = tf.layers.dropout(
            inputs=embedded,
            rate=params.embedded_dropout,  # 0.2
            # noise_shape=None, # [batch_size, 1, embed_dim]
            noise_shape=[batch_size, 1, params.embed_dim],  # drop embedding
            # noise_shape=[params.batch_size, tf.shape(embedded)[1], 1], # drop word
            training=is_train,
            name='dropout'
        )

    # temporal convolution
    with tf.variable_scope('conv_1'):
        if params.use_conv_1_prenet:
            conv_1_prenet_sizes=[params.embed_dim, params.embed_dim // 2]
            for d, units in enumerate(conv_1_prenet_sizes):
                with tf.variable_scope('prenet_{}'.format(d)):
                    dense = tf.layers.dense(
                        inputs=dropped_embedded,
                        units=units,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name='dense',
                        reuse=None
                    )
                    dropped_embedded = tf.layers.dropout(
                        inputs=dense,
                        rate=params.conv_1_prenet_dropout,  # 0.2
                        # noise_shape=None, # [batch_size, 1, embed_dim]
                        noise_shape=[batch_size, 1, units],  # drop embedding
                        # noise_shape=[params.batch_size, tf.shape(embedded)[1], 1], # drop word
                        training=is_train,
                        name='dropout'
                    )
        if params.use_conv_1_bank:
            with tf.variable_scope('conv_bank'):
                conv_fn = lambda k: \
                    tf.layers.conv1d(
                        inputs=dropped_embedded,
                        filters=params.conv_1_filters,  # 32
                        kernel_size=k,  # 7
                        strides=params.conv_1_strides,  # 1
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=1,
                        activation=None,  # relu6, default: linear
                        use_bias=False,
                        # kernel_initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
                        kernel_initializer=tf.glorot_uniform_initializer(
                            seed=None, dtype=float_type),
                        bias_initializer=tf.zeros_initializer(dtype=float_type),
                        trainable=True,
                        name='conv1d_{}'.format(k),
                        reuse=None
                    )  # (kernel_size * conv_1_conv1d_filters + use_bias) * embed_dim = (7 * 32 + 1) * 32 = 7200
                convolved = tf.concat(
                    [conv_fn(k) for k in range(1, params.conv_1_bank_size + 1)], axis=-1
                )
        else:
            convolved = tf.layers.conv1d(
                inputs=dropped_embedded,
                filters=params.conv_1_filters,  # 32
                kernel_size=params.conv_1_kernel_size,  # 7
                strides=params.conv_1_strides,  # 1
                padding='same',
                data_format='channels_last',
                dilation_rate=1,
                activation=None,  # relu6, default: linear
                use_bias=False,
                # kernel_initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
                kernel_initializer=tf.glorot_uniform_initializer(
                    seed=None, dtype=float_type),
                bias_initializer=tf.zeros_initializer(dtype=float_type),
                trainable=True,
                name='conv1d',
                reuse=None
            )  # (kernel_size * conv_1_conv1d_filters + use_bias) * embed_dim = (7 * 32 + 1) * 32 = 7200
        # convolved_norm = tf.layers.batch_normalization(
        #     convolved,
        #     axis=-1,
        #     momentum=0.99,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer=tf.zeros_initializer(dtype=float_type),
        #     gamma_initializer=tf.ones_initializer(dtype=float_type),
        #     moving_mean_initializer=tf.zeros_initializer(dtype=float_type),
        #     moving_variance_initializer=tf.ones_initializer(dtype=float_type),
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     training=is_train,
        #     trainable=True,
        #     name='batch_norm',
        #     reuse=None,
        #     renorm=False,
        #     renorm_clipping=None,
        #     renorm_momentum=0.99,
        #     fused=True,
        #     virtual_batch_size=None,
        #     adjustment=None
        # )
        if params.use_conv_batch_norm:
            convolved = tf.contrib.layers.batch_norm(
                inputs=convolved,
                decay=0.99,
                center=True,
                scale=False,
                epsilon=0.001,
                activation_fn=None,
                param_initializers=None,
                param_regularizers=None,
                updates_collections=tf.GraphKeys.UPDATE_OPS,
                is_training=is_train,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                batch_weights=None,
                fused=True,
                data_format='NHWC',
                zero_debias_moving_mean=False,
                scope=None,
                renorm=params.use_batch_renorm,
                renorm_clipping=None,
                renorm_decay=0.99,
                adjustment=None
            )
        convolved = tf.nn.relu(
            convolved,
            name='relu'
        )
        convolved = tf.layers.dropout(
            inputs=convolved,
            rate=params.conv_1_dropout,  # 0.2
            noise_shape=None,  # [batch_size, 1, embed_dim]
            training=is_train,
            name='dropout'
        )

        # Residual connection
        if params.use_conv_1_residual:
            convolved = tf.concat(values=[convolved, dropped_embedded], axis=-1)
        
        # Highway network
        if params.use_conv_1_highway:
            # Handle dimensionality mismatch:
            if convolved.shape[2] != params.conv_1_highway_units:
                convolved = tf.layers.dense(convolved, params.conv_1_highway_units)
            for d in range(params.conv_1_highway_depth):
                with tf.variable_scope('highway_{}'.format(d)):
                    H = tf.layers.dense(
                        inputs=convolved,
                        units=params.conv_1_highway_units,
                        activation=tf.nn.relu,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name='H',
                        reuse=None
                    )
                    T = tf.layers.dense(
                        inputs=convolved,
                        units=params.conv_1_highway_units,
                        activation=tf.nn.sigmoid,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=tf.constant_initializer(-1.0),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name='T',
                        reuse=None
                    )
                    # convolved = tf.Print(convolved,
                    #     data=[tf.shape(convolved), tf.shape(H), tf.shape(T)],
                    #     message='## DEBUG Highway: ',
                    #     summarize=500
                    #     ) # [128 512 16][128 512 32][128 512 32]
                    convolved = H * T + convolved * (1.0 - T)
        # Self attention layer
        if params.use_conv_1_attention:
            # convolved = dot_attention(
            #     inputs=convolved,
            #     memory=convolved,
            #     mask=mask,
            #     units=params.conv_1_attention_units,
            #     drop_inputs=0.0,
            #     drop_memory=0.0,
            #     drop_res=params.conv_1_attention_drop_res, # 0.2
            #     is_train=is_train,
            #     scope="attention"
            # )
            convolved = multihead_attention(
                query_antecedent=convolved,
                memory_antecedent=None,
                bias=attention_bias_ignore_padding(padding),
                total_key_depth=params.attention_key_channels or params.attention_hidden_size, # 32
                total_value_depth=params.attention_value_channels or params.attention_hidden_size, # 32
                output_depth=params.attention_hidden_size, # 32
                num_heads=params.attention_num_heads, # 1
                dropout_rate=params.attention_dropout, # 0.0
                attention_type=params.attention_type, # local_unmasked or dot_product
                block_length=params.attention_block_length, # 128
                filter_width=params.attention_filter_width, # 64
                save_weights_to=None,
                make_image_summary=False,
                dropout_broadcast_dims=None,
                max_length=None,
                vars_3d=False)
            if params.use_conv_1_attention_batch_norm:
                convolved = tf.contrib.layers.batch_norm(
                    inputs=convolved,
                    decay=0.99,
                    center=True,
                    scale=False,
                    epsilon=0.001,
                    activation_fn=None,
                    param_initializers=None,
                    param_regularizers=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
                    is_training=is_train,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    batch_weights=None,
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=False,
                    scope=None,
                    renorm=params.use_batch_renorm,
                    renorm_clipping=None,
                    renorm_decay=0.99,
                    adjustment=None
                )

    # bidirectional gru
    with tf.variable_scope('bi_rnn_1'):
        # rnn_float_type = tf.float16
        rnn_float_type = float_type
        rnn_cell_type = params.rnn_cell_type.lower()
        rnn_cells = {
            'lstmcell': tf.nn.rnn_cell.LSTMCell,
            'grucell': tf.nn.rnn_cell.GRUCell,
            'grublockcellv2': tf.contrib.rnn.GRUBlockCellV2,
            'srucell': tf.contrib.rnn.SRUCell,
            'layernormlstmcell': functools.partial(tf.contrib.rnn.LayerNormBasicLSTMCell,
                layer_norm=True
            ),
            'layernormlstmcellv2': functools.partial(LayerNormLSTMCellv2,
                layer_norm=True,
                layer_norm_columns=False,
                use_peepholes=params.use_rnn_peepholes
            ),
            'lstmblockfusedcell': lstm_ops.LSTMBlockFusedCell,
            'cudnnlstm': tf.contrib.cudnn_rnn.CudnnLSTM,
            'cudnngru': tf.contrib.cudnn_rnn.CudnnGRU,
            'qrnn': 'qrnn'
        }
        cell = rnn_cells[rnn_cell_type]
        if rnn_cell_type.startswith('cudnn'):
            transposed_convolved = tf.transpose(
                convolved, [1, 0, 2], name='transpose_to_rnn')
            lstm = cell(
                num_layers=len(params.rnn_num_units),
                num_units=params.rnn_num_units[0],
                input_mode='linear_input', # can be 'linear_input', 'skip_input' or 'auto_select'
                direction="bidirectional", # Can be either 'unidirectional' or 'bidirectional'
                dropout=params.rnn_dropout,
                seed=0,
                dtype=rnn_float_type,
                kernel_initializer=None,
                bias_initializer=None, # default is all zeros
                name='CudnnGRU1'
            )
            # call(self, inputs, initial_state=None, training=True)
            outputs, output_h = lstm(tf.cast(transposed_convolved, rnn_float_type))
            # Convert back from time-major outputs to batch-major outputs.
            outputs = tf.transpose(
                outputs, [1, 0, 2], name='transpose_from_rnn')
            # outputs = tf.Print(outputs,
            #     data=[outputs, dropped_convolved,
            #         tf.shape(outputs), tf.shape(dropped_convolved)],
            #     message='## DEBUG RNN: ',
            #     summarize=500
            #     ) # [64 1016 256][64 1016 32]
        elif rnn_cell_type == 'lstmblockfusedcell':
            transposed_convolved = tf.transpose(
                convolved, [1, 0, 2], name='transpose_to_rnn')
            lstm_fw = cell(
                num_units=params.rnn_num_units[0],
                forget_bias=1.0,
                cell_clip=None,
                use_peephole=params.use_rnn_peepholes,
                reuse=None,
                name="lstm_fused_cell_fw"
            )
            output_fw, output_fw_states = lstm_fw(
                inputs=tf.cast(transposed_convolved, rnn_float_type),
                initial_state=None,
                dtype=rnn_float_type,
                sequence_length=lengths
            )
            reversed_convolved = tf.reverse_sequence(
                input=transposed_convolved,
                seq_lengths=lengths,
                seq_axis=0,
                batch_axis=1,
                name='reverse_sequence'
            )
            lstm_bw = cell(
                num_units=params.rnn_num_units[0],
                forget_bias=1.0,
                cell_clip=None,
                use_peephole=params.use_rnn_peepholes,
                reuse=None,
                name="lstm_fused_cell_bw"
            )
            output_bw, output_bw_states = lstm_bw(
                inputs=tf.cast(reversed_convolved, rnn_float_type),
                initial_state=None,
                dtype=rnn_float_type,
                sequence_length=lengths
            )
            outputs = tf.concat(
                values=[output_fw, output_bw],
                axis=2,
                name='concat'
            )
            # Convert back from time-major outputs to batch-major outputs.
            outputs = tf.transpose(
                outputs, [1, 0, 2], name='transpose_from_rnn')
        elif rnn_cell_type == 'qrnn':
            import qrnn
            output_fw, output_fw_states = qrnn.qrnn(
                inputs=convolved,
                num_outputs=params.rnn_num_units[0],
                window=2,
                output_gate=True,
                activation_fn=tf.tanh,
                gate_activation_fn=tf.nn.sigmoid,
                padding="SAME",
                initial_state=None,
                time_major=False,
                scope='qrnn_fw'
            )
            reversed_convolved = tf.reverse_sequence(
                input=convolved,
                seq_lengths=lengths,
                seq_axis=1,
                batch_axis=0,
                name='reverse_sequence'
            )
            output_bw, output_bw_states = qrnn.qrnn(
                inputs=reversed_convolved,
                num_outputs=params.rnn_num_units[0],
                window=2,
                output_gate=True,
                activation_fn=tf.tanh,
                gate_activation_fn=tf.nn.sigmoid,
                padding="SAME",
                initial_state=None,
                time_major=False,
                scope='qrnn_bw'
            )
            # output_fw shape [batch, length, rnn_num_units]
            outputs = tf.concat(
                values=[output_fw, output_bw],
                axis=2,
                name='concat'
            )
        else:
            # cell = tf.nn.rnn_cell.LSTMCell
            # cell = tf.nn.rnn_cell.GRUCell
            # cell.count_params()
            # (embed_dim + rnn_num_units + use_bias) * (4 * rnn_num_units) * bidirectional
            # = (32 + 32 + 1) * (4 * 32) * 2 = 16640
            # cell = tf.nn.rnn_cell.GRUCell
            # (embed_dim + rnn_num_units + use_bias) * (3 * rnn_num_units) * bidirectional
            # = (32 + 32 + 1) * (3 * 32) * 2 = 12480
            # outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #     cells_fw=[cell(params.rnn_num_units)], # 32
            #     cells_bw=[cell(params.rnn_num_units)],
            #     inputs=dropped_convolved,
            #     dtype=rnn_float_type,
            #     sequence_length=lengths, # An int32/int64 vector, size `[batch_size]`,
            #     # containing the actual lengths for each of the sequences.
            #     ## the network is fully unrolled for the given (passed in)
            #     # length(s) of the sequence(s) or completely unrolled if length(s) is not
            #     # given.
            #     ## If the sequence_length vector is provided, dynamic calculation is performed.
            #     # This method of calculation does not compute the RNN steps past the maximum
            #     # sequence length of the minibatch (thus saving computational time),
            #     # and properly propagates the state at an example's sequence length
            #     # to the final state output.
            #     parallel_iterations=None, # default: 32. The number of iterations to run in
            #     # parallel.  Those operations which do not have any temporal dependency
            #     # and can be run in parallel, will be.  This parameter trades off
            #     # time for space.  Values >> 1 use more memory but take less time,
            #     # while smaller values use less memory but computations take longer.
            #     time_major=False,
            #     scope='bi_rnn_1'
            # )
            # outputs shape=(batch_size, sequence_length, params.rnn_num_units * 2), dtype=float32
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell(params.rnn_num_units[0]), # 32
                cell_bw=cell(params.rnn_num_units[0]),
                inputs=convolved,
                dtype=rnn_float_type,
                sequence_length=lengths, # An int32/int64 vector, size `[batch_size]`,
                # containing the actual lengths for each of the sequences.
                ## the network is fully unrolled for the given (passed in)
                # length(s) of the sequence(s) or completely unrolled if length(s) is not
                # given.
                ## If the sequence_length vector is provided, dynamic calculation is performed.
                # This method of calculation does not compute the RNN steps past the maximum
                # sequence length of the minibatch (thus saving computational time),
                # and properly propagates the state at an example's sequence length
                # to the final state output.
                parallel_iterations=None, # default: 32. The number of iterations to run in
                # parallel.  Those operations which do not have any temporal dependency
                # and can be run in parallel, will be.  This parameter trades off
                # time for space.  Values >> 1 use more memory but take less time,
                # while smaller values use less memory but computations take longer.
                time_major=False,
                scope='bi_rnn_1'
            )
            outputs = tf.concat(
                values=outputs,
                axis=2,
                name='concat'
            )
            # outputs = tf.Print(outputs,
            #     data=[outputs, dropped_convolved,
            #         tf.shape(outputs), tf.shape(dropped_convolved)],
            #     message='## DEBUG RNN: ',
            #     summarize=500
            #     ) # [64 1016 128][64 1016 32]
        if params.use_rnn_batch_norm:
            # outputs = tf.layers.batch_normalization(
            #     outputs,
            #     axis=-1,
            #     momentum=0.99,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     beta_initializer=tf.zeros_initializer(dtype=float_type),
            #     gamma_initializer=tf.ones_initializer(dtype=float_type),
            #     moving_mean_initializer=tf.zeros_initializer(dtype=float_type),
            #     moving_variance_initializer=tf.ones_initializer(dtype=float_type),
            #     beta_regularizer=None,
            #     gamma_regularizer=None,
            #     beta_constraint=None,
            #     gamma_constraint=None,
            #     training=is_train,
            #     trainable=True,
            #     name='batch_norm',
            #     reuse=None,
            #     renorm=False,
            #     renorm_clipping=None,
            #     renorm_momentum=0.99,
            #     fused=True,
            #     virtual_batch_size=None,
            #     adjustment=None
            # )
            outputs = tf.contrib.layers.batch_norm(
                inputs=outputs,
                decay=0.99,
                center=True,
                scale=False,
                epsilon=0.001,
                activation_fn=None,
                param_initializers=None,
                param_regularizers=None,
                updates_collections=tf.GraphKeys.UPDATE_OPS,
                is_training=is_train,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                batch_weights=None,
                fused=True,
                data_format='NHWC',
                zero_debias_moving_mean=False,
                scope=None,
                renorm=params.use_batch_renorm,
                renorm_clipping=None,
                renorm_decay=0.99,
                adjustment=None
            )

    with tf.variable_scope('rnn_attention_1'):
        if params.use_rnn_attention:
            # outputs = dot_attention(
            #     inputs=outputs,
            #     memory=outputs,
            #     mask=mask,
            #     units=params.rnn_attention_units,
            #     drop_inputs=0.0,
            #     drop_memory=0.0,
            #     drop_res=params.rnn_attention_drop_res, # 0.2
            #     is_train=is_train,
            #     scope="attention"
            # )
            outputs = multihead_attention(
                query_antecedent=outputs,
                memory_antecedent=None,
                bias=attention_bias_ignore_padding(padding),
                total_key_depth=params.attention_key_channels or params.attention_hidden_size, # 32
                total_value_depth=params.attention_value_channels or params.attention_hidden_size, # 32
                output_depth=params.rnn_attention_hidden_size, # 128
                num_heads=params.attention_num_heads, # 1
                dropout_rate=params.attention_dropout, # 0.0
                attention_type=params.attention_type, # local_unmasked or dot_product
                block_length=params.attention_block_length, # 128
                filter_width=params.attention_filter_width, # 64
                save_weights_to=None,
                make_image_summary=False,
                dropout_broadcast_dims=None,
                max_length=None,
                vars_3d=False)
            if params.use_rnn_attention_batch_norm:
                outputs = tf.contrib.layers.batch_norm(
                    inputs=outputs,
                    decay=0.99,
                    center=True,
                    scale=False,
                    epsilon=0.001,
                    activation_fn=None,
                    param_initializers=None,
                    param_regularizers=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
                    is_training=is_train,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    batch_weights=None,
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=False,
                    scope=None,
                    renorm=params.use_batch_renorm,
                    renorm_clipping=None,
                    renorm_decay=0.99,
                    adjustment=None
                )

    # output layer
    with tf.variable_scope('output_1'):
        logits = tf.layers.dense(
            inputs=outputs,
            units=params.num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.glorot_uniform_initializer(
                seed=None, dtype=float_type),
            bias_initializer=tf.zeros_initializer(dtype=float_type),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name='dense',
            reuse=None
        )
        # logits shape=(batch_size, sequence_length, num_classes), dtype=float32

    # loss
    with tf.variable_scope('loss'):
        if params.use_crf:
            # crf layer
            with tf.variable_scope('crf_1'):
                # inputs shape=(batch_size, sequence_length, num_classes), dtype=float32
                # tag_indices shape=(batch_size, sequence_length), dtype=int32
                # sequence_lengths shape=(batch_size), dtype=int32
                # transition_params shape=(num_classes, num_classes), dtype=float32
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=labels,
                    sequence_lengths=lengths,
                    transition_params=None
                )
                loss = tf.reduce_mean(-log_likelihood)
                # log_likelihood shape=(batch_size, sequence_length, num_classes), dtype=float32
                # transition_params shape=(num_classes, num_classes), dtype=float32
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=tf.cast(logits, tf.float32)
            )
            # tf.summary.histogram('losses', losses)
            # losses shape=(batch_size, sequence_length), dtype=float32
            masked_losses = losses * mask
            # average across batch_size and sequence_length
            loss = tf.reduce_sum(masked_losses) / \
                tf.cast(tf.reduce_sum(lengths), dtype=tf.float32)
        # tf.summary.scalar('loss', loss)

    # predictions
    with tf.variable_scope('predictions'):
        predictions = {
            'logits': logits,
            # Add `softmax_tensor` to the graph.
            'probabilities': tf.nn.softmax(logits=logits, axis=-1, name='softmax_tensor')
        }
        # # Generate predictions (for PREDICT and EVAL mode)
        # if params.use_crf:
        #     # potentials shape=(batch_size, sequence_length, num_classes), dtype=float32
        #     # sequence_length shape=(batch_size), dtype=int32
        #     # transition_params shape=(num_classes, num_classes), dtype=float32
        #     decode_tags, best_score = tf.contrib.crf.crf_decode(
        #         potentials=logits,
        #         transition_params=transition_params,
        #         sequence_length=lengths
        #     )
        #     # decode_tags shape=(batch_size, sequence_length), dtype=int32
        #     # best_score shape=(batch_size), dtype=float32
        #     predictions['classes'] = decode_tags
        # else:
        #     predictions['classes'] = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        predictions['classes'] = tf.argmax(input=logits, axis=-1, output_type=tf.int32)

    with tf.variable_scope('metrics'):
        metrics = {
            # # true_positives / (true_positives + false_positives)
            # 'precision': tf.metrics.precision(
            #     labels=labels,
            #     predictions=predictions['classes'],
            #     weights=mask,
            #     name='precision'
            # ),
            # # true_positives / (true_positives + false_negatives)
            # 'recall': tf.metrics.recall(
            #     labels=labels,
            #     predictions=predictions['classes'],
            #     weights=mask,
            #     name='recall'
            # ),
            # matches / total
            'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions['classes'],
                weights=mask,
                name='accuracy'
            )
        }
        with tf.name_scope('batch_accuracy', values=[predictions['classes'], labels]):
            is_correct = tf.cast(
                tf.equal(predictions['classes'], labels), tf.float32)
            is_correct = tf.multiply(is_correct, mask)
            num_values = tf.multiply(mask, tf.ones_like(is_correct))
            batch_accuracy = tf.div(tf.reduce_sum(
                is_correct), tf.reduce_sum(num_values))
        tf.summary.scalar('accuracy', batch_accuracy)
        # tf.summary.scalar('accuracy', metrics['accuracy'][0])
        # currently only works for bool
        # tf.summary.scalar('precision', metrics['precision'][1])
        # tf.summary.scalar('recall', metrics['recall'][1])

    # optimizer list
    optimizers = {
        'adagrad': tf.train.AdagradOptimizer,
        'adam': lambda lr: tf.train.AdamOptimizer(lr, epsilon=params.adam_epsilon),
        'nadam': lambda lr: tf.contrib.opt.NadamOptimizer(lr, epsilon=params.adam_epsilon),
        'ftrl': tf.train.FtrlOptimizer,
        'momentum': lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9),
        'rmsprop': tf.train.RMSPropOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
    }

    # optimizer
    with tf.variable_scope('optimizer'):
        # clip_gradients = params.gradient_clipping_norm
        clip_gradients = adaptive_clipping_fn(
            std_factor=params.clip_gradients_std_factor,  # 2.
            decay=params.clip_gradients_decay,  # 0.95
            static_max_norm=params.clip_gradients_static_max_norm,  # 6.
            global_step=global_step,
            report_summary=True,
            epsilon=np.float32(1e-7),
            name=None
        )

        
        def learning_rate_warmup(global_step, warmup_steps, repeat_steps=0, start=0.01, warmup_schedule='exp'):
            """Learning rate warmup multiplier."""
            local_step = global_step
            if repeat_steps > 0:
                local_step = global_step % repeat_steps
            if not warmup_steps:
                return tf.constant(1.)

            tf.logging.info('Applying %s learning rate warmup for %d steps',
                            warmup_schedule, warmup_steps)

            local_step = tf.to_float(local_step)
            warmup_steps = tf.to_float(warmup_steps)
            start = tf.to_float(start)
            warmup = tf.constant(1.)
            if warmup_schedule == 'exp':
                warmup = tf.exp(tf.log(start) / warmup_steps)**(warmup_steps - local_step)
            else:
                assert warmup_schedule == 'linear'
                warmup = ((tf.constant(1.) - start) / warmup_steps) * local_step + start
            return tf.where(local_step < warmup_steps, warmup, tf.constant(1.))

        decay = params.learning_rate_decay_fn.lower()
        lr = params.learning_rate # if not decay or decay == 'none':
        if decay == 'noisy_linear_cosine_decay':
            lr = tf.train.noisy_linear_cosine_decay(
                params.learning_rate,
                global_step,
                decay_steps=params.learning_rate_decay_steps,  # 27000000
                initial_variance=1.0,
                variance_decay=0.55,
                num_periods=0.5,
                alpha=0.0,
                beta=0.001,
                name=None
            )
        if decay == 'exponential_decay':
            lr = tf.train.exponential_decay(
                params.learning_rate,
                global_step,
                decay_steps=params.learning_rate_decay_steps, # 27000000
                decay_rate=params.learning_rate_decay_rate, # 0.95
                staircase=False,
                name=None
            )
        if params.warmup_steps > 0:
            warmup = learning_rate_warmup(
                global_step, 
                warmup_steps=params.warmup_steps, # 35000
                repeat_steps=params.warmup_repeat_steps, # 0
                start=params.warmup_start_lr, # 0.001,
                warmup_schedule=params.warmup_schedule, # 'exp'
            )
            lr = lr * warmup
        # Add learning rate to summary
        tf.summary.scalar("learning_rate", lr)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=lr,  # 0.001
            optimizer=optimizers[params.optimizer.lower()],
            gradient_noise_scale=None,
            gradient_multipliers=None,
            # some gradient clipping stabilizes training in the beginning.
            # clip_gradients=clip_gradients,
            # clip_gradients=6.,
            # clip_gradients=None,
            # learning_rate_decay_fn=learning_rate_decay_fn,
            update_ops=None,
            variables=None,
            name=None,
            summaries=[
                # 'gradients',
                # 'gradient_norm',
                'loss',
                # 'learning_rate' # only added if learning_rate_decay_fn is not None
            ],
            colocate_gradients_with_ops=True,
            increment_global_step=True
        )

    group_inputs = [train_op]

    # runtime numerical checks
    if params.check_nans:
        checks = tf.add_check_numerics_ops()
        group_inputs = [checks]

    # update accuracy
    # group_inputs.append(metrics['accuracy'][1])

    # record total number of examples processed
    examples_processed = tf.get_variable(
        name='examples_processed',
        initializer=tf.cast(0, tf.int64),
        trainable=False,
        dtype=tf.int64,
        aggregation=tf.VariableAggregation.SUM
    )
    # print('examples_processed', examples_processed)
    group_inputs.append(tf.assign_add(examples_processed,
                                      tf.cast(batch_size, tf.int64), name='update_examples_processed'))
    epoch = examples_processed // seq_total
    group_inputs.append(epoch)
    progress = examples_processed / seq_total - tf.cast(epoch, tf.float64)
    group_inputs.append(progress)
    
    train_op = tf.group(*group_inputs)

    if params.debug:
        train_op = tf.cond(
            pred=tf.logical_or(
                tf.is_nan(tf.reduce_max(embeddings)),
                tf.equal(global_step, 193000)
            ),
            false_fn=lambda: train_op,
            true_fn=lambda: tf.Print(train_op,
                                     # data=[global_step, metrics['accuracy'][0], lengths, loss, losses, predictions['classes'], labels, mask, protein, embeddings],
                                     data=[global_step, batch_accuracy,
                                           lengths, loss, embeddings],
                                     message='## DEBUG LOSS: ',
                                     summarize=50000
                                     )
        )

    training_hooks = []
    # INFO:tensorflow:global_step/sec: 2.07549
    training_hooks.append(tf.train.StepCounterHook(
        output_dir=params.model_dir,
        every_n_steps=params.log_step_count_steps
    ))
    # INFO:tensorflow:accuracy = 0.16705106, examples = 15000, loss = 9.688441, step = 150 (24.091 sec)
    def logging_formatter(v):
        return 'accuracy:\033[1;32m {:9.5%}\033[0m, loss:\033[1;32m {:8.5f}\033[0m, lr:\033[1;32m {:8.5f}\033[0m, step:\033[1;32m {:7,d}\033[0m'.format(v['accuracy'], v['loss'], v['learning_rate'], v['step'])

    tensors = {
        'accuracy': batch_accuracy,
        'loss': loss,
        'step': global_step,
        'learning_rate': lr
        # 'input_size': tf.shape(protein),
        # 'examples': examples_processed
    }
    # if is_train:
    #     tensors['epoch'] = epoch
    #     tensors['progress'] = progress
        
    training_hooks.append(ColoredLoggingTensorHook(
        tensors=tensors,
        every_n_iter=params.log_step_count_steps,
        at_end=False, formatter=logging_formatter
    ))
    training_hooks.append(EpochProgressBarHook(
        total=seq_total,
        initial_tensor=examples_processed,
        n_tensor=batch_size,
        postfix_tensors=None,
        every_n_iter=params.log_step_count_steps
    ))
    if params.trace:
        training_hooks.append(tf.train.ProfilerHook(
            save_steps=params.save_summary_steps,
            output_dir=params.model_dir,
            show_dataflow=True,
            show_memory=True
        ))
    training_hooks.append(EpochCheckpointInputPipelineHook(
        checkpoint_dir=params.model_dir,
        config=config,
        save_secs=params.save_checkpoints_secs, # 10m
        save_steps=None,
    ))

    # default saver is added in estimator._train_with_estimator_spec
    # training.Saver(
    #   sharded=True,
    #   max_to_keep=self._config.keep_checkpoint_max,
    #   keep_checkpoint_every_n_hours=(
    #       self._config.keep_checkpoint_every_n_hours),
    #   defer_build=True,
    #   save_relative_paths=True)
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(
        sharded=False,
        max_to_keep=config.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=(
            config.keep_checkpoint_every_n_hours),
        defer_build=True,
        save_relative_paths=True))

    training_chief_hooks = []
    # # saving_listeners like _NewCheckpointListenerForEvaluate 
    # # will be called on the first CheckpointSaverHook
    # training_chief_hooks.append(tf.train.CheckpointSaverHook(
    #     checkpoint_dir=params.model_dir,
    #     # effectively only save on start and end of MonitoredTrainingSession
    #     save_secs=30 * 24 * 60 * 60,
    #     save_steps=None,
    #     checkpoint_basename="model.epoch",
    #     saver=tf.train.Saver(
    #         sharded=False,
    #         max_to_keep=0,
    #         defer_build=False,
    #         save_relative_paths=True
    #     )
    # ))
    # # Add a second CheckpointSaverHook to save every save_checkpoints_secs
    # training_chief_hooks.append(tf.train.CheckpointSaverHook(
    #     checkpoint_dir=params.model_dir,
    #     save_secs=params.save_checkpoints_secs, # 10m
    #     save_steps=None,
    #     checkpoint_basename="model.step",
    #     scaffold=scaffold
    # ))
    training_chief_hooks.append(EpochCheckpointSaverHook(
        checkpoint_dir=params.model_dir,
        epoch_tensor=epoch,
        save_secs=params.save_checkpoints_secs, # 10m
        save_steps=None,
        scaffold=scaffold
    ))

    # local training:
    # all_hooks=[
    # EpochCheckpointSaverHook, Added into training_chief_hooks in this model_fn
    # SummarySaverHook, # Added into chief_hooks in MonitoredTrainingSession()
    # _DatasetInitializerHook, # Added into worker_hooks in Estimator._train_model_default
    # NanTensorHook, # Added into worker_hooks in Estimator._train_with_estimator_spec
    # StepCounterHook, # Added into training_hooks in this model_fn
    # LoggingTensorHook,  # Added into training_hooks in this model_fn
    # EpochCheckpointInputPipelineHook # Added into training_hooks in this model_fn
    # ]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions)
        },
        scaffold=scaffold,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks,
        evaluation_hooks=None,
        prediction_hooks=None
    )

# https://github.com/tensorflow/models/blob/69cf6fca2106c41946a3c395126bdd6994d36e6b/tutorials/rnn/quickdraw/train_model.py


def create_estimator_and_specs(run_config):
    """Creates an Estimator, TrainSpec and EvalSpec."""
    # parse metadata
    metadata = None
    with open(FLAGS.metadata_path) as f:
        metadata = json.load(f)
    
    # build hyperparameters
    model_params = tf.contrib.training.HParams(
        command=FLAGS.command,
        model_dir=FLAGS.model_dir,
        tfrecord_pattern={
            tf.estimator.ModeKeys.TRAIN: FLAGS.training_data,
            tf.estimator.ModeKeys.EVAL: FLAGS.eval_data,
        },
        metadata_path=FLAGS.metadata_path,
        experiment_name=FLAGS.experiment_name,
        host_script_name=FLAGS.host_script_name,
        job=FLAGS.job,
        num_gpus=FLAGS.num_gpus,
        num_cpu_threads=FLAGS.num_cpu_threads,
        random_seed=FLAGS.random_seed,
        use_jit_xla=FLAGS.use_jit_xla,
        use_tensor_ops=FLAGS.use_tensor_ops,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        log_step_count_steps=FLAGS.log_step_count_steps,
        eval_delay_secs=FLAGS.eval_delay_secs,
        eval_throttle_secs=FLAGS.eval_throttle_secs,
        steps=FLAGS.steps,
        eval_steps=FLAGS.eval_steps,
        dataset_buffer=FLAGS.dataset_buffer,  # 256 MB
        dataset_parallel_reads=FLAGS.dataset_parallel_reads,  # 1
        shuffle_buffer=FLAGS.shuffle_buffer,  # 16 * 1024 examples
        repeat_count=FLAGS.repeat_count,  # -1 = Repeat the input indefinitely.
        batch_size=FLAGS.batch_size,
        prefetch_buffer=FLAGS.prefetch_buffer,  # batches

        vocab_size=FLAGS.vocab_size,  # 28
        embed_dim=FLAGS.embed_dim,  # 32
        embedded_dropout=FLAGS.embedded_dropout,  # 0.2

        use_conv_1_prenet=FLAGS.use_conv_1_prenet,  # 32
        conv_1_prenet_dropout=FLAGS.conv_1_prenet_dropout,  # 32
        use_conv_1_bank=FLAGS.use_conv_1_bank,  # 32
        conv_1_bank_size=FLAGS.conv_1_bank_size,  # 32
        conv_1_filters=FLAGS.conv_1_filters,  # 32
        conv_1_kernel_size=FLAGS.conv_1_kernel_size,  # 7
        conv_1_strides=FLAGS.conv_1_strides,  # 1
        conv_1_dropout=FLAGS.conv_1_dropout,  # 0.2
        use_conv_batch_norm=FLAGS.use_conv_batch_norm,
        use_conv_1_residual=FLAGS.use_conv_1_residual,
        use_conv_1_highway=FLAGS.use_conv_1_highway,
        conv_1_highway_depth=FLAGS.conv_1_highway_depth,
        conv_1_highway_units=FLAGS.conv_1_highway_units,
        use_conv_1_attention=FLAGS.use_conv_1_attention,
        attention_key_channels=FLAGS.attention_key_channels,
        attention_value_channels=FLAGS.attention_value_channels,
        attention_hidden_size=FLAGS.attention_hidden_size,
        attention_num_heads=FLAGS.attention_num_heads,
        attention_dropout=FLAGS.attention_dropout,
        attention_type=FLAGS.attention_type,
        attention_block_length=FLAGS.attention_block_length,
        attention_filter_width=FLAGS.attention_filter_width,
        use_conv_1_attention_batch_norm=FLAGS.use_conv_1_attention_batch_norm,

        rnn_cell_type=FLAGS.rnn_cell_type,
        rnn_num_units=FLAGS.rnn_num_units, # list
        rnn_dropout=FLAGS.rnn_dropout,
        use_rnn_batch_norm=FLAGS.use_rnn_batch_norm,
        use_rnn_attention=FLAGS.use_rnn_attention,
        rnn_attention_hidden_size=FLAGS.rnn_attention_hidden_size,
        use_rnn_attention_batch_norm=FLAGS.use_rnn_attention_batch_norm,
        use_rnn_peepholes=FLAGS.use_rnn_peepholes,

        use_crf=FLAGS.use_crf, # True
        use_batch_renorm=FLAGS.use_batch_renorm,

        num_classes=FLAGS.num_classes,

        clip_gradients_std_factor=FLAGS.clip_gradients_std_factor,  # 2.
        clip_gradients_decay=FLAGS.clip_gradients_decay,  # 0.95
        # 6.
        clip_gradients_static_max_norm=FLAGS.clip_gradients_static_max_norm,

        learning_rate_decay_fn=FLAGS.learning_rate_decay_fn,
        learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,  # 10000
        learning_rate_decay_rate=FLAGS.learning_rate_decay_rate,  # 0.9
        learning_rate=FLAGS.learning_rate,  # 0.001
        warmup_steps=FLAGS.warmup_steps,  # 35000 (10% epoch)
        warmup_repeat_steps=FLAGS.warmup_repeat_steps,  # 0
        warmup_start_lr=FLAGS.warmup_start_lr,  # 0.001
        warmup_schedule=FLAGS.warmup_schedule,  # exp
        optimizer=FLAGS.optimizer,
        adam_epsilon=FLAGS.adam_epsilon,

        check_nans=FLAGS.check_nans,
        trace=FLAGS.trace,
        debug=FLAGS.debug,
        # num_layers=FLAGS.num_layers,
        # num_conv=ast.literal_eval(FLAGS.num_conv),
        # conv_len=ast.literal_eval(FLAGS.conv_len),
        # gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        # cell_type=FLAGS.cell_type,
        # batch_norm=FLAGS.batch_norm
        metadata=metadata
    )

    # hook = tf_debug.LocalCLIDebugHook()

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    # save model_params to model_dir/hparams.json
    hparams_path = Path(estimator.model_dir,
                     'hparams-{:%Y-%m-%d-%H-%M-%S}.json'.format(datetime.datetime.now()))
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    hparams_path.write_text(model_params.to_json(indent=2, sort_keys=False))

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        # A function that provides input data for training as minibatches.
        # max_steps=FLAGS.steps or None,  # 0
        max_steps=None,
        # Positive number of total steps for which to train model. If None, train forever.
        hooks=None 
        # passed into estimator.train(hooks)
        # and then into _train_with_estimator_spec(hooks)
        # Iterable of `tf.train.SessionRunHook` objects to run
        # on all workers (including chief) during training.
        # CheckpointSaverHook? Not here, need only to run on cchief, put in 
        # estimator_spec.training_chief_hooks
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        # A function that constructs the input data for evaluation.
        steps=FLAGS.eval_steps,  # 10
        # Positive number of steps for which to evaluate model. If
        # `None`, evaluates until `input_fn` raises an end-of-input exception.
        name=None,
        # Name of the evaluation if user needs to run multiple
        # evaluations on different data sets. Metrics for different evaluations
        # are saved in separate folders, and appear separately in tensorboard.
        hooks=None,
        # Iterable of `tf.train.SessionRunHook` objects to run
        # during evaluation.
        exporters=None,
        # Iterable of `Exporter`s, or a single one, or `None`.
        # `exporters` will be invoked after each evaluation.
        start_delay_secs=FLAGS.eval_delay_secs,  # 30 * 24 * 60 * 60
        # used for distributed training continuous evaluator only
        # Int. Start evaluating after waiting for this many seconds.
        throttle_secs=FLAGS.eval_throttle_secs  # 30 * 24 * 60 * 60
        # full dataset at batch=4 currently needs 15 days
        # adds a StopAtSecsHook(eval_spec.throttle_secs)
        # Do not re-evaluate unless the last evaluation was
        # started at least this many seconds ago. Of course, evaluation does not
        # occur if no new checkpoints are available, hence, this is the minimum.
    )

    return estimator, train_spec, eval_spec


def main(unused_args):
    # Parse experiment_name and host_script_name if needed
    # Example: --model_dir=${MODELDIR}/Attention_lr0.5_ws35000_${CARDTYPE}_${HOSTSCRIPT}.${NUMGPU}
    # experiment_name = Attention
    # host_script_name = ${HOSTSCRIPT}.${NUMGPU}
    model_dir_parts = Path(FLAGS.model_dir).name.split('_')
    if len(model_dir_parts) > 1:
        if FLAGS.experiment_name == 'PARSE': # Default: Exp
            FLAGS.experiment_name = model_dir_parts[0]
        if FLAGS.host_script_name == 'PARSE': # Default: tensorflow
            FLAGS.host_script_name = model_dir_parts[-1]
    # setup colored logger
    coloredlogs.DEFAULT_FIELD_STYLES = dict(
        asctime=dict(color='green'),
        hostname=dict(color='magenta', bold=True),
        levelname=dict(color='black', bold=True),
        programname=dict(color='cyan', bold=True),
        name=dict(color='blue'))
    coloredlogs.DEFAULT_LEVEL_STYLES = dict(
        spam=dict(color='green', faint=True),
        debug=dict(color='green'),
        verbose=dict(color='blue'),
        info=dict(),
        notice=dict(color='magenta'),
        warning=dict(color='yellow'),
        success=dict(color='green', bold=True),
        error=dict(color='red'),
        critical=dict(color='red', bold=True))
    # '\x1b[32m%(asctime)s,%(msecs)03d\x1b[0m \x1b[1;35m%(hostname)s\x1b[0m \x1b[34m%(name)s[%(process)d]\x1b[0m \x1b[1;30m%(levelname)s\x1b[0m %(message)s'
    coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s \x1b[1;35m{}\x1b[0m \x1b[34m{}[%(process)d]\x1b[0m %(levelname)s %(message)s'.format(
        FLAGS.experiment_name, FLAGS.host_script_name
    )
    logger = tf_logging._get_logger()
    coloredlogs.install(
        level='DEBUG', 
        logger=logger, 
        milliseconds=True,
        stream=logger.handlers[0].stream
    )
    # print(Fore.RED + 'some red text' + Style.RESET_ALL, file=logger.handlers[0].stream)

    # set logger.handler.stream to output to our TqdmFile
    for h in logger.handlers:
        # <StreamHandler <stderr> (NOTSET)>
        # <StandardErrorHandler <stderr> (DEBUG)>
        # print(h)
        h.acquire()
        try:
            h.flush()
            orig_stdout = h.stream
            h.stream = TqdmFile(file=h.stream)
        finally:
            h.release()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    # tf.logging.debug('test')

    # check tfrecords data exists
    if len(glob.glob(FLAGS.training_data)) == 0:
        msg = 'No training data files found for pattern: {}'.format(FLAGS.training_data)
        tf.logging.fatal(msg)
        raise IOError(msg)
    if len(glob.glob(FLAGS.eval_data)) == 0:
        msg = 'No evaluation data files found for pattern: {}'.format(FLAGS.eval_data)
        tf.logging.fatal(msg)
        raise IOError(msg)
    if len(glob.glob(FLAGS.metadata_path)) == 0:
        msg = 'No metadata file found for pattern: {}'.format(FLAGS.metadata_path)
        tf.logging.fatal(msg)
        raise IOError(msg)

    # Hardware info
    FLAGS.num_gpus = FLAGS.num_gpus or tf.contrib.eager.num_gpus()
    FLAGS.num_cpu_threads = FLAGS.num_cpu_threads or os.cpu_count()

    # multi gpu distribution strategy
    distribution = None
    if FLAGS.num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
        tf.logging.info('MirroredStrategy num_gpus: {}'.format(FLAGS.num_gpus))

    # Set the seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    # Use JIT XLA
    # session_config = tf.ConfigProto(log_device_placement=True)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    # default session config when init Estimator
    session_config.graph_options.rewrite_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    if FLAGS.use_jit_xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1  # pylint: disable=no-member

    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            train_distribute=distribution,
            model_dir=FLAGS.model_dir,
            # Directory to save model parameters, graph and etc. This can
            # also be used to load checkpoints from the directory into a estimator to
            # continue training a previously saved model. If `PathLike` object, the
            # path will be resolved. If `None`, the model_dir in `config` will be used
            # if set. If both are set, they must be same. If both are `None`, a
            # temporary directory will be used.
            tf_random_seed=FLAGS.random_seed,  # 33
            # Random seed for TensorFlow initializers.
            # Setting this value allows consistency between reruns.
            save_summary_steps=FLAGS.save_summary_steps,  # 10
            # if not None, a SummarySaverHook will be added in MonitoredTrainingSession()
            # The frequency, in number of global steps, that the
            # summaries are written to disk using a default SummarySaverHook. If both
            # `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
            # the default summary saver isn't used. Default 100.
            save_checkpoints_steps=None, # 100
            # Save checkpoints every this many steps.
            # save_checkpoints_secs=None,
            # We will define our own CheckpointSaverHook in EstimatorSpec.training_chief_hooks
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,  # 10m
            # if not None, a CheckpointSaverHook will be added in MonitoredTrainingSession()
            # Save checkpoints every this many seconds with
            # CheckpointSaverHook. Can not be specified with `save_checkpoints_steps`.
            # Defaults to 600 seconds if both `save_checkpoints_steps` and
            # `save_checkpoints_secs` are not set in constructor.
            # If both `save_checkpoints_steps` and `save_checkpoints_secs` are None,
            # then checkpoints are disabled.
            keep_checkpoint_max=FLAGS.keep_checkpoint_max,  # 5
            # Maximum number of checkpoints to keep.  As new checkpoints
            # are created, old ones are deleted.  If None or 0, no checkpoints are
            # deleted from the filesystem but only the last one is kept in the
            # `checkpoint` file.  Presently the number is only roughly enforced.  For
            # example in case of restarts more than max_to_keep checkpoints may be
            # kept.
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,  # 6
            # keep an additional checkpoint
            # every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
            # kept for every 0.5 hours of training, this is in addition to the
            # keep_checkpoint_max checkpoints.
            # Defaults to 10,000 hours.
            log_step_count_steps=None, # Customized LoggingTensorHook defined in model_fn
            # if not None, a StepCounterHook will be added in MonitoredTrainingSession()
            # log_step_count_steps=FLAGS.log_step_count_steps,  # 10
            # The frequency, in number of global steps, that the
            # global step/sec will be logged during training.
            session_config=session_config))
    
    while True:
        eval_result_metrics, export_results = tf.estimator.train_and_evaluate(
            estimator, train_spec, eval_spec)
    # eval_result = _EvalResult(
    #   status=_EvalStatus.EVALUATED,
    #   metrics=metrics,
    #   checkpoint_path=latest_ckpt_path)
    # export_results = [eval_spec.exporters.export(), ...
    # eval_spec.exporters.export() = The string path to the exported directory.
        
    # _TrainingExecutor.run()
    # _TrainingExecutor.run_local()
    # estimator.train(input_fn, max_steps)
    # loss = estimator._train_model(input_fn, hooks, saving_listeners)
    # estimator._train_model_default(input_fn, hooks, saving_listeners)
    # features, labels, input_hooks = (estimator._get_features_and_labels_from_input_fn(input_fn, model_fn_lib.ModeKeys.TRAIN))
    # estimator_spec = estimator._call_model_fn(features, labels, model_fn_lib.ModeKeys.TRAIN, estimator.config)
    # estimator._train_with_estimator_spec(estimator_spec, worker_hooks, hooks, global_step_tensor, saving_listeners)
    # _, loss = MonitoredTrainingSession.run([estimator_spec.train_op, estimator_spec.loss])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.register('type', 'list', lambda v: ast.literal_eval(v))

    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Exp',
        help='Experiment name for logging purposes, if "PARSE", split model_dir by "_" and take first item')
    parser.add_argument(
        '--host_script_name',
        type=str,
        default='tensorflow',
        help='Host script name for logging purposes (8086K1-1.2), if "PARSE", split model_dir by "_" and take last item')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./checkpoints/Exp_tensorflow',
        help='Path for saving model checkpoints during training')
    parser.add_argument(
        '--job',
        type=str,
        choices=['train', 'eval', 'predict', 'prep_dataset'],
        default='train',
        help='Set job type to run')
    parser.add_argument(
        '--training_data',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-train.tfrecords',
        default='D:/datasets/pfam-regions-d10-s20-train.tfrecords',
        help='Path to training data (tf.Example in TFRecord format)')
    parser.add_argument(
        '--eval_data',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-test.tfrecords',
        default='D:/datasets/pfam-regions-d10-s20-test.tfrecords',
        help='Path to evaluation data (tf.Example in TFRecord format)')
    parser.add_argument(
        '--metadata_path',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-test.tfrecords',
        default='',
        help='Path to metadata.json generated by prep_dataset')
    parser.add_argument(
        '--num_classes',
        type=int,
        # default=16712 + 3, # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
        default=10 + 3,  # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
        help='Number of domain classes.')
    parser.add_argument(
        '--classes_file',
        type=str,
        default='',
        help='Path to a file with the classes - one class per line')

    parser.add_argument(
        '--num_gpus',
        type=int,
        default=0,
        help='Number of GPUs to use, defaults to total number of gpus available.')
    parser.add_argument(
        '--num_cpu_threads',
        type=int,
        default=0,
        help='Number of CPU threads to use, defaults to half the number of hardware threads.')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=33,
        help='The random seed.')
    parser.add_argument(
        '--use_jit_xla',
        type='bool',
        default='False',
        help='Whether to enable JIT XLA.')
    parser.add_argument(
        '--use_tensor_ops',
        type='bool',
        default='False',
        help='Whether to use tensorcores or not.')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=100,
        help='Save summaries every this many steps.')
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help='Save checkpoints every this many steps.')
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=10 * 60,
        help='Save checkpoints every this many seconds.')
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.')
    parser.add_argument(
        '--keep_checkpoint_every_n_hours',
        type=float,
        default=6,
        help='Keep an additional checkpoint every `N` hours.')
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec will be logged during training.')
    parser.add_argument(
        '--eval_delay_secs',
        type=int,
        default=30 * 24 * 60 * 60,
        help='Start distributed continuous evaluation after waiting for this many seconds. Not used in local training.')
    parser.add_argument(
        '--eval_throttle_secs',
        type=int,
        default=30 * 24 * 60 * 60,
        help='Stop training and start evaluation after this many seconds.')

    parser.add_argument(
        '--steps',
        type=int,
        default=0,  # 100000,
        help='Number of training steps, if 0 train forever.')
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=100,  # 100000,
        help='Number of evaluation steps, if 0, evaluates until end-of-input.')

    parser.add_argument(
        '--dataset_buffer',
        type=int,
        default=256,
        help='Number of MB in the read buffer.')
    parser.add_argument(
        '--dataset_parallel_reads',
        type=int,
        default=1,
        help='Number of input Datasets to interleave from in parallel.')
    parser.add_argument(
        '--shuffle_buffer',
        type=int,
        default=16 * 1024,
        help='Maximum number elements that will be buffered when shuffling input.')
    parser.add_argument(
        '--repeat_count',
        type=int,
        default=1,
        help='Number of times the dataset should be repeated.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size to use for longest sequence for training/evaluation. 1 if GPU Memory <= 6GB, 2 if <= 12GB')
    parser.add_argument(
        '--prefetch_buffer',
        type=int,
        default=64,
        help='Maximum number of batches that will be buffered when prefetching.')

    parser.add_argument(
        '--vocab_size',
        type=int,
        default=len(aa_list) + 1,  # 'PAD'
        help='Vocabulary size.')
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=32,
        help='Embedding dimensions.')
    parser.add_argument(
        '--embedded_dropout',
        type=float,
        default=0.2,
        help='Dropout rate used for embedding layer outputs.')

    parser.add_argument(
        '--use_conv_1_prenet',
        type='bool',
        default='True',
        help='Add prenet before convolution layer 1.')
    parser.add_argument(
        '--conv_1_prenet_dropout',
        type=float,
        default=0.2,
        help='Dropout rate used for prenet layer outputs.')
    parser.add_argument(
        '--use_conv_1_bank',
        type='bool',
        default='True',
        help='Use convolution bank.')
    parser.add_argument(
        '--conv_1_bank_size',
        type=int,
        default=16,
        help='Convolution bank kernal sizes 1 to bank_size.')
    parser.add_argument(
        '--conv_1_filters',
        type=int,
        default=32,
        help='Number of convolution filters.')
    parser.add_argument(
        '--conv_1_kernel_size',
        type=int,
        default=7,
        help='Length of the convolution filters.')
    parser.add_argument(
        '--conv_1_strides',
        type=int,
        default=1,
        help='The number of entries by which the filter is moved right at each step..')
    parser.add_argument(
        '--conv_1_dropout',
        type=float,
        default=0.2,
        help='Dropout rate used for convolution layer outputs.')
    parser.add_argument(
        '--use_conv_batch_norm',
        type='bool',
        default='True',
        help='Apply batch normalization after convolution layers.')
    parser.add_argument(
        '--use_conv_1_residual',
        type='bool',
        default='True',
        help='Add residual connection after convolution layer 1.')
    parser.add_argument(
        '--use_conv_1_highway',
        type='bool',
        default='True',
        help='Add a highway network after convolution layer 1.')
    parser.add_argument(
        '--conv_1_highway_depth',
        type=int,
        default=3,
        help='Number of layers of highway network.')
    parser.add_argument(
        '--conv_1_highway_units',
        type=int,
        default=32,
        help='Number of units per layer of highway network.')
    parser.add_argument(
        '--use_conv_1_attention',
        type='bool',
        default='True',
        help='Add an attention layer after convolution layer 1.')
    parser.add_argument(
        '--attention_key_channels',
        type=int,
        default=32,
        help='Number of attention key units.')
    parser.add_argument(
        '--attention_value_channels',
        type=int,
        default=32,
        help='Number of attention value units.')
    parser.add_argument(
        '--attention_hidden_size',
        type=int,
        default=32,
        help='Number of attention units.')
    parser.add_argument(
        '--attention_num_heads',
        type=int,
        default=1,
        help='Number of attention heads.')
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0.0,
        help='Dropout rate used for attention.')
    parser.add_argument(
        '--attention_type',
        type=str,
        choices=['local_unmasked', 'dot_product'],
        default='local_unmasked',
        help='Attention Type')
    parser.add_argument(
        '--attention_block_length',
        type=int,
        default=128,
        help='The sequence is divided into blocks of length block_length.')
    parser.add_argument(
        '--attention_filter_width',
        type=int,
        default=64,
        help='Attention for a given query position can see all memory positions in the corresponding block and filter_width many positions to the left and right of the block.')
    parser.add_argument(
        '--use_conv_1_attention_batch_norm',
        type='bool',
        default='True',
        help='Apply batch normalization after convolution attention layers.')

    parser.add_argument(
        '--rnn_cell_type',
        type=str,
        choices=['CudnnLSTM', 'CudnnGRU', 'LSTMCell', 'GRUCell', 'LayerNormLSTMCell', 'LayerNormLSTMCellv2', 'LSTMBlockFusedCell', 'GRUBlockCellV2', 'SRUCell', 'qrnn'],
        default='CudnnGRU',
        help='RNN Cell Type')
    parser.add_argument(
        '--rnn_num_units',
        type='list',
        default='[128]',
        help='Number of node per recurrent network layer.')
    parser.add_argument(
        '--rnn_dropout',
        type=float,
        default=0.0,
        help='Dropout rate used between rnn layers.')
    parser.add_argument(
        '--use_rnn_batch_norm',
        type='bool',
        default='True',
        help='Apply batch normalization after recurrent layers.')
    parser.add_argument(
        '--use_rnn_attention',
        type='bool',
        default='True',
        help='Add an attention layer after recurrent layers.')
    parser.add_argument(
        '--rnn_attention_hidden_size',
        type=int,
        default=128,
        help='Number of recurrent network layer attention output nodes.')
    parser.add_argument(
        '--use_rnn_attention_batch_norm',
        type='bool',
        default='True',
        help='Apply batch normalization after rnn attention layers.')
    parser.add_argument(
        '--use_rnn_peepholes',
        type='bool',
        default='False',
        help='Enable diagonal/peephole connections in LayerNormLSTMCellv2 and LSTMBlockFusedCell cells.')

    parser.add_argument(
        '--use_crf',
        type='bool',
        default='False',
        help='Calculate loss using linear chain CRF instead of Softmax.')
    parser.add_argument(
        '--use_batch_renorm',
        type='bool',
        default='True',
        help='Use Batch Renormalization.')

    parser.add_argument(
        '--clip_gradients_std_factor',
        type=float,
        default=2.,  # num_batches_per_epoch * num_epochs_per_decay(8)
        help='If the norm exceeds `exp(mean(log(norm)) + std_factor*std(log(norm)))` then all gradients will be rescaled such that the global norm becomes `exp(mean)`.')
    parser.add_argument(
        '--clip_gradients_decay',
        type=float,
        default=0.95,
        help='The smoothing factor of the moving averages.')
    parser.add_argument(
        '--clip_gradients_static_max_norm',
        type=float,
        default=6.,
        help='If provided, will threshold the norm to this value as an extra safety.')

    parser.add_argument(
        '--learning_rate_decay_fn',
        type=str,
        default='None',
        help='Learning rate decay function. One of "none", "noisy_linear_cosine_decay", "exponential_decay"')
    parser.add_argument(
        '--learning_rate_decay_steps',
        type=int,
        default=27000000,  # num_batches_per_epoch * num_epochs_per_decay(8)
        help='Decay learning_rate by decay_rate every decay_steps.')
    parser.add_argument(
        '--learning_rate_decay_rate',
        type=float,
        default=0.95,
        help='Learning rate decay rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate used for training.')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=35000,  # 10% epoch
        help='Learning rate warmup steps needed to reach specified learning_rate.')
    parser.add_argument(
        '--warmup_repeat_steps',
        type=int,
        default=0,  # 0 to disable repeat warmup
        help='Restart warmup every this many steps.')
    parser.add_argument(
        '--warmup_start_lr',
        type=float,
        default=0.001,
        help='Learning rate warmup starting multiplier value.')
    parser.add_argument(
        '--warmup_schedule',
        type=str,
        default='exp',
        help='Learning rate warmup schedule. One of "exp", "linear", "none"')
    # learning rate defaults
    # Adagrad: 0.01
    # Adam: 0.001
    # RMSProp: 0.001
    # :
    # Nadam: 0.002
    # SGD: 0.01
    # Adamax: 0.002
    # Adadelta: 1.0
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='Optimizer to use. One of "Adagrad", "Adam", "Ftrl", "Momentum", "RMSProp", "SGD"')
    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=0.1,
        help='A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.')

    parser.add_argument(
        '--check_nans',
        type='bool',
        default='False',
        help='Add runtime checks to spot when NaNs or other symptoms of numerical errors start occurring during training.')
    parser.add_argument(
        '--trace',
        type='bool',
        default='False',
        help='Captures CPU/GPU profiling information in "timeline-<step>.json", which are in Chrome Trace format.')
    parser.add_argument(
        '--debug',
        type='bool',
        default='False',
        help='Run debugging ops.')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.command = ' '.join(sys.argv)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
