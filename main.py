r"""Entry point for trianing a RNN-based classifier for the pfam data.

python train.py \
  --training_data train_data \
  --eval_data eval_data \
  --checkpoint_dir ./checkpoints/ \
  --cell_type cudnn_lstm
  
python main.py train \
  --dataset pfam_regions \
  --model_spec rnn_v1 \

python main.py predict \
  --trained_model rnn_v1 \
  --input predict_data

When running on GPUs using --cell_type cudnn_lstm is much faster.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import ast
import glob
import argparse
import datetime
import functools
from pathlib import Path

import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.data.ops import iterator_ops
from tensorflow.contrib.data.python.ops.iterator_ops import _Saveable, _CustomSaver
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training import training_util
from tensorflow.python.framework import meta_graph
from tensorflow.python.data.util import nest
from tensorflow.contrib.layers.python.layers import adaptive_clipping_fn
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import tf_logging as logging
import numpy as np
from colorama import init, Fore, Back, Style
import coloredlogs

# Disable cpp warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Show debugging output, default: tf.logging.INFO
logger = logging._get_logger()
coloredlogs.DEFAULT_FIELD_STYLES = dict(
    asctime=dict(color='green'),
    hostname=dict(color='magenta'),
    levelname=dict(color='black', bold=True),
    programname=dict(color='cyan'),
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
coloredlogs.install(level='DEBUG', logger=logger, milliseconds=True)
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = None

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
        self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
                                        every_steps=save_steps)
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
        tf.logging.info("Saving\033[33m input\033[0m checkpoints for %d into %s.", step, self._save_path)

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
        if self._epoch_tensor is None:
            self._epoch_tensor = tf.get_variable(
                name='epoch',
                shape=[],
                dtype=tf.int64,
                initializer=tf.zeros_initializer(),
                trainable=False)
                
        if self._epoch_saver is None:
            self._epoch_saver = tf.train.Saver(
                sharded=False,
                max_to_keep=9999,
                keep_checkpoint_every_n_hours=999999,
                defer_build=False,
                save_relative_paths=True
            )

        self._increment_epoch = tf.assign_add(self._epoch_tensor, 1, 
            use_locking=True, name='increment_epoch')

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
        epoch = session.run(self._increment_epoch)

        with tf.control_dependencies([self._increment_epoch]):
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
        save_path = '{}-{}'.format(self._epoch_save_path, epoch)
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
        tf.logging.info("Saving\033[1;32m step\033[0m checkpoints for %d into %s.", step, self._step_save_path)

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

# num_domain = 10
# test_split = 0.2
# validation_split = 0.1
# batch_size = 64
# epochs = 300

# embedding_dims = 32
# embedding_dropout = 0.2
# lstm_output_size = 32
# filters = 32
# kernel_size = 7
# conv_dropout = 0.2
# dense_size = 32
# dense_dropout = 0.2
# print('Building model...')
# model = Sequential()
# # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dims)
# model.add(Embedding(len(pfam_regions.aa_list) + 1,
#                     embedding_dims,
#                     input_length=None))
# model.add(Dropout(embedding_dropout))
# model.add(Conv1D(filters,
#                 kernel_size,
#                 padding='same',
#                 activation='relu',
#                 strides=1))
# model.add(TimeDistributed(Dropout(conv_dropout)))
# # Expected input batch shape: (batch_size, timesteps, data_dim)
# # returns a sequence of vectors of dimension lstm_output_size
# model.add(Bidirectional(CuDNNGRU(lstm_output_size, return_sequences=True)))
# # model.add(Bidirectional(LSTM(lstm_output_size, dropout=0.0, recurrent_dropout=0.0, return_sequences=True)))

# # model.add(TimeDistributed(Dense(dense_size)))
# # model.add(TimeDistributed(Activation('relu')))
# # model.add(TimeDistributed(Dropout(dense_dropout)))
# model.add(TimeDistributed(Dense(3 + num_domain, activation='softmax')))
# model.summary()
# epoch_start = 0
# pfam_regions.sparse_train(model, model_name, __file__, num_domain, device='/gpu:0',
#     epoch_start=epoch_start, batch_size=batch_size, epochs=epochs,
#     predicted_dir='C:/Users/Hotdogee/Documents/Annotate/predicted')

# # _________________________________________________________________
# # Layer (type)                 Output Shape              Param #
# # =================================================================
# # embedding_1 (Embedding)      (None, None, 32)          896
# # _________________________________________________________________
# # dropout_1 (Dropout)          (None, None, 32)          0
# # _________________________________________________________________
# # conv1d_1 (Conv1D)            (None, None, 32)          7200
# # _________________________________________________________________
# # time_distributed_1 (TimeDist (None, None, 32)          0
# # _________________________________________________________________
# # bidirectional_1 (Bidirection (None, None, 64)          12672
# # _________________________________________________________________
# # time_distributed_2 (TimeDist (None, None, 13)          845
# # =================================================================
# # Total params: 21,613
# # Trainable params: 21,613
# # Non-trainable params: 0
# # _________________________________________________________________

# # 13000/53795 [======>.......................] - ETA: 7:24:26 - loss: 0.0776 - acc: 0.9745(64, 3163) (64, 3163, 1) 3163 3163


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
    return variable


def model_fn(features, labels, mode, params, config):
    # labels shape=(batch_size, sequence_length), dtype=int32
    is_train = mode == tf.estimator.ModeKeys.TRAIN

    protein = features['protein']
    # protein shape=(batch_size, sequence_length), dtype=int32
    lengths = features['lengths']
    # lengths shape=(batch_size, ), dtype=int32
    global_step = tf.train.get_global_step()
    # global_step is assign_add 1 in tf.train.Optimizer.apply_gradients
    epoch = tf.get_variable(
        name='epoch',
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False)
    batch_size = tf.shape(lengths)[0]

    if params.use_tensor_ops:
        float_type = tf.float16
    else:
        float_type = tf.float32

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
        convolved = tf.layers.conv1d(
            inputs=dropped_embedded,
            filters=params.conv_1_filters,  # 32
            kernel_size=params.conv_1_kernel_size,  # 7
            strides=params.conv_1_strides,  # 1
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
            activation=tf.nn.relu,  # relu6, default: linear
            use_bias=True,
            # kernel_initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            kernel_initializer=tf.glorot_uniform_initializer(
                seed=None, dtype=float_type),
            bias_initializer=tf.zeros_initializer(dtype=float_type),
            trainable=True,
            name='conv1d',
            reuse=None
        )  # (kernel_size * conv_1_conv1d_filters + use_bias) * embed_dim = (7 * 32 + 1) * 32 = 7200
        dropped_convolved = tf.layers.dropout(
            inputs=convolved,
            rate=params.conv_1_dropout,  # 0.2
            noise_shape=None,  # [batch_size, 1, embed_dim]
            training=is_train,
            name='dropout'
        )

    # bidirectional gru
    with tf.variable_scope('bi_rnn_1'):
        # cell = tf.nn.rnn_cell.BasicLSTMCell
        # # cell.count_params()
        # # (embed_dim + rnn_num_units + use_bias) * (4 * rnn_num_units) * bidirectional
        # # = (32 + 32 + 1) * (4 * 32) * 2 = 16640
        # # cell = tf.nn.rnn_cell.GRUCell
        # # (embed_dim + rnn_num_units + use_bias) * (3 * rnn_num_units) * bidirectional
        # # = (32 + 32 + 1) * (3 * 32) * 2 = 12480
        # outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        #     cells_fw=[cell(params.rnn_num_units)], # 32
        #     cells_bw=[cell(params.rnn_num_units)],
        #     inputs=dropped_convolved,
        #     dtype=tf.float32,
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
        # # outputs shape=(batch_size, sequence_length, params.rnn_num_units * 2), dtype=float32
        ###############
        # rnn_float_type = tf.float16
        rnn_float_type = float_type
        transposed_convolved = tf.transpose(
            dropped_convolved, [1, 0, 2], name='transpose_to_rnn')
        lstm = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1,
            num_units=params.rnn_num_units,
            direction="bidirectional",
            name='CudnnGRU1',
            dtype=rnn_float_type,
            dropout=0.,
            seed=0
        )
        outputs, output_h = lstm(tf.cast(transposed_convolved, rnn_float_type))
        # Convert back from time-major outputs to batch-major outputs.
        transposed_outputs = tf.transpose(
            outputs, [1, 0, 2], name='transpose_from_rnn')

    # output layer
    with tf.variable_scope('output_1'):
        logits = tf.layers.dense(
            inputs=transposed_outputs,
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

    # predictions
    with tf.variable_scope('predictions'):
        predictions = {
            'logits': logits,
            # Add `softmax_tensor` to the graph.
            'probabilities': tf.nn.softmax(logits=logits, axis=-1, name='softmax_tensor'),
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        }

    # loss
    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=tf.cast(logits, tf.float32)
        )
        # tf.summary.histogram('losses', losses)
        # losses shape=(batch_size, sequence_length), dtype=float32
        mask = tf.cast(tf.sign(labels), dtype=tf.float32)  # 0 = 'PAD'
        masked_losses = losses * mask
        # average across batch_size and sequence_length
        loss = tf.reduce_sum(masked_losses) / \
            tf.cast(tf.reduce_sum(lengths), dtype=tf.float32)
        # tf.summary.scalar('loss', loss)

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

        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.noisy_linear_cosine_decay(
                learning_rate,
                global_step,
                decay_steps=params.learning_rate_decay_steps,  # 27000000
                initial_variance=1.0,
                variance_decay=0.55,
                num_periods=0.5,
                alpha=0.0,
                beta=0.001,
                name=None
            )
            # return tf.train.exponential_decay(
            #     learning_rate,
            #     global_step,
            #     decay_steps=params.learning_rate_decay_steps, # 27000000
            #     decay_rate=params.learning_rate_decay_rate, # 0.95
            #     staircase=False,
            #     name=None
            # )
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=params.learning_rate,  # 0.001
            optimizer=optimizers[params.optimizer.lower()],
            gradient_noise_scale=None,
            gradient_multipliers=None,
            # some gradient clipping stabilizes training in the beginning.
            # clip_gradients=clip_gradients,
            # clip_gradients=6.,
            # clip_gradients=None,
            learning_rate_decay_fn=learning_rate_decay_fn,
            update_ops=None,
            variables=None,
            name=None,
            summaries=[
                # 'gradients',
                # 'gradient_norm',
                'loss',
                'learning_rate',
            ],
            colocate_gradients_with_ops=False,
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
    examples_processed = tf.Variable(
        0, trainable=False, name='examples_processed')
    group_inputs.append(tf.assign_add(examples_processed,
                                      batch_size, name='update_examples_processed'))

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
    def logging_formatter(tensor_values):
        pass
    training_hooks.append(tf.train.LoggingTensorHook(
        tensors={
            'epoch': epoch,
            'accuracy': batch_accuracy,
            'loss': loss,
            'step': global_step,
            'input_size': tf.shape(protein),
            'examples': examples_processed
        },
        every_n_iter=params.log_step_count_steps,
        at_end=False, formatter=None
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
    model_params = tf.contrib.training.HParams(
        job=FLAGS.job,
        model_dir=FLAGS.model_dir,
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

        tfrecord_pattern={
            tf.estimator.ModeKeys.TRAIN: FLAGS.training_data,
            tf.estimator.ModeKeys.EVAL: FLAGS.eval_data,
        },
        dataset_buffer=FLAGS.dataset_buffer,  # 256 MB
        dataset_parallel_reads=FLAGS.dataset_parallel_reads,  # 1
        shuffle_buffer=FLAGS.shuffle_buffer,  # 16 * 1024 examples
        repeat_count=FLAGS.repeat_count,  # -1 = Repeat the input indefinitely.
        batch_size=FLAGS.batch_size,
        prefetch_buffer=FLAGS.prefetch_buffer,  # batches

        vocab_size=FLAGS.vocab_size,  # 28
        embed_dim=FLAGS.embed_dim,  # 32
        embedded_dropout=FLAGS.embedded_dropout,  # 0.2

        conv_1_filters=FLAGS.conv_1_filters,  # 32
        conv_1_kernel_size=FLAGS.conv_1_kernel_size,  # 7
        conv_1_strides=FLAGS.conv_1_strides,  # 1
        conv_1_dropout=FLAGS.conv_1_dropout,  # 0.2

        rnn_num_units=FLAGS.rnn_num_units,

        num_classes=FLAGS.num_classes,

        clip_gradients_std_factor=FLAGS.clip_gradients_std_factor,  # 2.
        clip_gradients_decay=FLAGS.clip_gradients_decay,  # 0.95
        # 6.
        clip_gradients_static_max_norm=FLAGS.clip_gradients_static_max_norm,

        learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,  # 10000
        learning_rate_decay_rate=FLAGS.learning_rate_decay_rate,  # 0.9
        learning_rate=FLAGS.learning_rate,  # 0.001
        learning_rate_decay_fn='noisy_linear_cosine_decay',
        optimizer=FLAGS.optimizer,
        adam_epsilon=FLAGS.adam_epsilon,

        check_nans=FLAGS.check_nans,
        trace=FLAGS.trace,
        debug=FLAGS.debug
        # num_layers=FLAGS.num_layers,
        # num_conv=ast.literal_eval(FLAGS.num_conv),
        # conv_len=ast.literal_eval(FLAGS.conv_len),
        # gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        # cell_type=FLAGS.cell_type,
        # batch_norm=FLAGS.batch_norm
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
    # check tfrecords data exists
    if len(glob.glob(FLAGS.training_data)) == 0:
        msg = 'No training data files found for pattern: {}'.format(FLAGS.training_data)
        tf.logging.fatal(msg)
        raise IOError(msg)
    if len(glob.glob(FLAGS.eval_data)) == 0:
        msg = 'No evaluation data files found for pattern: {}'.format(FLAGS.eval_data)
        tf.logging.fatal(msg)
        raise IOError(msg)
    
    # Hardware info
    FLAGS.num_gpus = FLAGS.num_gpus or tf.contrib.eager.num_gpus()
    FLAGS.num_cpu_threads = FLAGS.num_cpu_threads or os.cpu_count()

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

# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d10-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d10-s20-test.tfrecords --model_dir=./checkpoints/cent-d10-v1 --batch_size=64
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0-v1
# full dataset, batchsize=8 NaN loss, batchsize=4 works
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b4 --num_classes=16715 --batch_size=4 --save_summary_steps=10 --log_step_count_steps=100
# python main.py --model_dir=./checkpoints/win-d10
# python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:/checkpoints/win-d0b4 --num_classes=16715 --batch_size=4 --save_summary_steps=10 --log_step_count_steps=10
# python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:/checkpoints/win-d0b4-6 --num_classes=16715 --batch_size=4 --save_summary_steps=100 --log_step_count_steps=100 --learning_rate=0.001
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-2 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=27000000 --decay_rate=0.95 --learning_rate=0.001
# NaN at 844800 step
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-3 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=13500000 --decay_rate=0.95 --learning_rate=0.001
# NaN at 490000 step
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-4 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.1
# NaN at
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-5 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01
# NaN at 400
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-6 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.005
# NaN at 20000
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-7 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001
# NaN at 553400
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-8 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0008
# NaN at 1368600
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-9 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0007
# NaN at 1259000
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-10 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0006
# NaN at 1268600
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-11 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0005
# NaN at 2584200
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-12 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0004
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-13 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0008
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-14 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0008
# Stuck at accuracy 0.33
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-15 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01
# Stuck at accuracy 0.33 @ 45400 (5,567,542) avg_batch_size = 122
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-16 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.0001
# Stuck at accuracy (not stuck? 54% @ 7400, 60% @ 13600, 63.65% @ 24200 (2,966,834))
# NaN loss at 24200
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-17 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.001
## 26, 35, 37, 38, 41, 40, 41, 41, 41, 41
# NaN loss at 69800, 67.68% @ 69800 (8,559,218)
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-18 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.005
# NaN loss at accuracy = 0.7103102, examples = 32475754, loss = 1.290427, step = 264800 (77.593 sec)
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/cent-d0b2-19 --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.01
# 77.2% @ 1 epoch (353700 step), 81.5% @ 2 epoch (86,757,674 examples, 707400 step)
# NaN loss at accuracy = 0.7378826, examples = 132973318, loss = 1.0418819, step = 1084200 (77.894 sec)
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=./checkpoints/cent-d0b2-25 --use_tensor_ops=true
# Compiled tf with cuda9.2, cudnn7.1.4, nvidia driver 396.26, float32: 75.832 sec
# Compiled tf with cuda9.2, cudnn7.1.4, nvidia driver 396.26, float16: 95.004 sec
### after pad_to_multiples 8
# TF_DEBUG_CUDNN_RNN=1, TF_DEBUG_CUDNN_RNN_ALGO=1, TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=1
# Compiled tf with cuda9.0, cudnn7.1.4, nvidia driver 396.26, float32: 71.739 sec
# Compiled tf with cuda9.0, cudnn7.1.4, nvidia driver 396.26, float16: 77.675 sec
# Compiled tf with cuda9.2, cudnn7.1.4, nvidia driver 396.26, float32: 71.799 sec
# Compiled tf with cuda9.2, cudnn7.1.4, nvidia driver 396.26, float16: 77.719 sec
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=./checkpoints/cent-d0b2-63

# docker
# python main.py --training_data=/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=./checkpoints/cent-d0b2-26 --use_tensor_ops=true


# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b2-13-5930k --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0003
# OOM at 351600
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b1-1-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0003
# DataLossError at 570200
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b1-2-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.0003
# DataLossError at
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b1-3-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001
# DataLossError at
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b1-4-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001
# DataLossError at 13800, corrupted record at 4876677767
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-5-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.9
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-6-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.1 --adam_epsilon=0.001
# NaN at 0
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-6-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.1 --adam_epsilon=0.01
# Stuck at accuracy 31%
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-7-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.03 --adam_epsilon=0.001
# Stuck at accuracy 21, 24, 25
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-8-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.03 --adam_epsilon=0.0001
# Stuck at accuracy 21, 24
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-9-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=20 --log_step_count_steps=20 --decay_steps=1000000 --learning_rate=0.02 --adam_epsilon=0.0001
# Stuck at accuracy 13, 25, 25, 26, 27, 28, 29, 31, 30, 31, 33, 33
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-10-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=20 --log_step_count_steps=20 --decay_steps=1000000 --learning_rate=0.025 --adam_epsilon=0.0001
# Stuck at accuracy 13, 25, 25, 26, 27, 28, 29, 31, 30, 31, 33, 33
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-11-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=20 --log_step_count_steps=20 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.0001
# Stuck at accuracy
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-12-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=20 --log_step_count_steps=20 --decay_steps=1000000 --learning_rate=0.001 --adam_epsilon=0.0001
# Stuck at accuracy
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-13-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.05
# Stuck at accuracy 21, 24, 25, 26, 24, 24, 26, 27, 29, 30, 31, 30
# python main.py --training_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=C:\Users\Hotdogee\Documents\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=C:\Users\Hotdogee\Documents/checkpoints/d0b1-14-5930k --num_classes=16715 --batch_size=1 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 
# With CUDA 9.2 CUDNN 7.1.4:
# INFO:tensorflow:loss = 9.671407, step = 200 (80.576 sec)
# INFO:tensorflow:loss = 9.610422, step = 400 (78.203 sec)
# With CUDA 9.2 CUDNN 7.1.4 MKL:
# INFO:tensorflow:loss = 9.671407, step = 200 (79.954 sec)
# INFO:tensorflow:loss = 9.610422, step = 400 (78.346 sec)

# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b2nan-1-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01
# NaN at 400
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-2-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001 --check_nans=False --optimizer=adam
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-2-1950x-xla --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001 --check_nans=False --optimizer=adam --use_jit_xla=true
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-3-1950x-trace --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001 --check_nans=False --optimizer=adam --trace=true
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-4-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001 --check_nans=False --optimizer=adam
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-5-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.001 --check_nans=False --optimizer=adam
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-6-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.1
# python main.py --training_data=D:\datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:\datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=D:\checkpoints/d0b2-7-1950x --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=1000000 --learning_rate=0.01 --adam_epsilon=0.0001
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-8-1950x
# INFO:tensorflow:loss = 4.5400634, step = 200 (115.787 sec)
# $Env:CUDA_VISIBLE_DEVICES=1; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-9-1950x
# INFO:tensorflow:loss = 4.5400634, step = 200 (110.805 sec)
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-10-1950x-TITANV --use_tensor_ops=true
# INFO:tensorflow:loss = 4.526874, step = 200 (123.831 sec)
# $Env:CUDA_VISIBLE_DEVICES=1; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-11-1950x-1080Ti --use_tensor_ops=true
# INFO:tensorflow:loss = 4.526843, step = 200 (137.962 sec)
####################
## Compiled using CUDA 9.2 CUDNN 7.1.4
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-15-1950x-TITANV
# INFO:tensorflow:loss = 4.5400643, step = 200 (85.344 sec)
# $Env:TF_AUTOTUNE_THRESHOLD=1
# $Env:TF_DEBUG_CUDNN_RNN=1
# $Env:TF_DEBUG_CUDNN_RNN_ALGO=1
# $Env:TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=0
# $Env:TF_CUDNN_USE_AUTOTUNE=1
# $Env:TF_CUDNN_RNN_USE_AUTOTUNE=1
# $Env:TF_ENABLE_TENSOR_OP_MATH=1
# $Env:TF_ENABLE_TENSOR_OP_MATH_FP32=1
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-16-1950x-TITANV
# INFO:tensorflow:loss = 4.5400643, step = 200 (96.615 sec)
# $Env:TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=1
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.01 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-17-1950x-TITANV
# ERROR:tensorflow:Model diverged with loss = NaN.
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-18-1950x-TITANV
# INFO:tensorflow:loss = 9.653481, step = 200 (78.383 sec)
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-19-1950x-TITANV --use_tensor_ops=true
# INFO:tensorflow:loss = 9.655021, step = 200 (83.497 sec)
# $Env:TF_DEBUG_CUDNN_RNN_ALGO=2
# $Env:TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=0
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=D:/datasets/pfam-regions-d0-s20-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=200 --log_step_count_steps=200 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 --model_dir=D:/checkpoints/d0b2-20-1950x-TITANV
# INFO:tensorflow:loss = 9.6519, step = 200 (116.726 sec)
# $Env:TF_DEBUG_CUDNN_RNN_ALGO=0
# $Env:TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=1
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets2/pfam-regions-d0-s20-p1-train.tfrecords --eval_data=D:/datasets2/pfam-regions-d0-s20-p1-test.tfrecords --num_classes=16715 --batch_size=2 --save_summary_steps=50 --log_step_count_steps=50 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 --model_dir=D:/checkpoints/pfam-regions-d0-s20-p1/1950x-TITANV/d0b2-1
# 
# $Env:CUDA_VISIBLE_DEVICES=0; python main.py --training_data=D:/datasets2/pfam-regions-d4000-s20-t1-e1-train.tfrecords --eval_data=D:/datasets2/pfam-regions-d4000-s20-t1-e1-test.tfrecords --num_classes=4003 --batch_size=2 --save_summary_steps=50 --log_step_count_steps=5 --decay_steps=10000 --learning_rate=0.0001 --adam_epsilon=0.02 --save_checkpoints_secs=3 --keep_checkpoint_max=2 --model_dir=D:/checkpoints/pfam-regions-d4000-s20-t1-e1/1950x-TITANV/b2-lr0.0001-2
#


# sequence count = 54,223,493, train = 43,378,794, test = 10,844,699
# class count = 16715, batch size = 4, batch count = 13,555,873, batch per sec = 11, time per epoch = 1,232,352 sec = 14 days
# 1950X: 13.5 step/sec, 8107 step@10min, tf1.9.0, win10
# titan: 12.8 step/sec, 7675 step@10min, tf1.9.0, centos
if __name__ == '__main__':
    # $Env:TF_AUTOTUNE_THRESHOLD=1
    # $Env:TF_DEBUG_CUDNN_RNN=1
    # $Env:TF_DEBUG_CUDNN_RNN_ALGO=1
    # $Env:TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=0
    # $Env:TF_CUDNN_USE_AUTOTUNE=1
    # $Env:TF_CUDNN_RNN_USE_AUTOTUNE=1

    # $Env:TF_ENABLE_TENSOR_OP_MATH=1
    # $Env:TF_ENABLE_TENSOR_OP_MATH_FP32=1
    # $Env:TF_CUDNN_RNN_USE_V2=1

    # export TF_ENABLE_TENSOR_OP_MATH=1
    # export TF_ENABLE_TENSOR_OP_MATH_FP32=1
    # export TF_CUDNN_USE_AUTOTUNE=1
    # export TF_CUDNN_RNN_USE_AUTOTUNE=1
    # export TF_CUDNN_RNN_USE_V2=1

    # export TF_ENABLE_TENSOR_OP_MATH=1
    # export TF_ENABLE_TENSOR_OP_MATH_FP32=1

    # export TF_DEBUG_CUDNN_RNN=1
    # export TF_DEBUG_CUDNN_RNN_ALGO=1
    # export TF_DEBUG_CUDNN_RNN_USE_TENSOR_OPS=1

    # export TF_CPP_MIN_LOG_LEVEL=0
    # export TF_CPP_MIN_VLOG_LEVEL=0
    # export TF_USE_CUDNN=1
    # export CUDNN_VERSION=7.1.4
    # export TF_AUTOTUNE_THRESHOLD=2
    # export TF_NEED_CUDA=1
    # export TF_CUDNN_VERSION=7
    # export TF_CUDA_VERSION=9.2
    # export TF_ENABLE_XLA=1
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')

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
        '--meta_data',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-test.tfrecords',
        default='',
        help='Path to metadata.json generated by prep_dataset)')
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
        help='Whether to enable batch normalization or not.')
    parser.add_argument(
        '--use_tensor_ops',
        type='bool',
        default='False',
        help='Whether to use tensorcores or not.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./checkpoints/v2',
        help='Path for saving model checkpoints during training')
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
        '--rnn_num_units',
        type=int,
        default=128,
        help='Number of node per recurrent network layer.')

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
    # parser.add_argument(
    #     '--num_layers',
    #     type=int,
    #     default=3,
    #     help='Number of recurrent neural network layers.')
    # parser.add_argument(
    #     '--num_conv',
    #     type=str,
    #     default='[48, 64, 96]',
    #     help='Number of conv layers along with number of filters per layer.')
    # parser.add_argument(
    #     '--conv_len',
    #     type=str,
    #     default='[5, 5, 3]',
    #     help='Length of the convolution filters.')
    # parser.add_argument(
    #     '--gradient_clipping_norm',
    #     type=float,
    #     default=9.0,
    #     help='Gradient clipping norm used during training.')
    # parser.add_argument(
    #     '--cell_type',
    #     type=str,
    #     default='lstm',
    #     help='Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.')
    # parser.add_argument(
    #     '--batch_norm',
    #     type='bool',
    #     default='False',
    #     help='Whether to enable batch normalization or not.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# # Config
# FLAGS = tf.app.flags.FLAGS

# # Data parameters
# tf.app.flags.DEFINE_string('train_data_dir', '/tmp/output_records/train',
#                            'The path containing training TFRecords.')

# tf.app.flags.DEFINE_string('eval_data_dir', '/tmp/output_records/valid',
#                            'The path containing evaluation TFRecords.')

# tf.app.flags.DEFINE_string('model_dir', '/tmp/model/my_first_model',
#                            'The path to write the model to.')

# tf.app.flags.DEFINE_boolean('clean_model_dir', True,
#                             'Whether to start from fresh.')

# # Hyperparameters
# tf.app.flags.DEFINE_float('learning_rate', 1.e-2,
#                           'The learning rate.')

# tf.app.flags.DEFINE_integer('batch_size', 1024,
#                             'The batch size.')

# tf.app.flags.DEFINE_integer('epochs', 1024,
#                             'Number of epochs to train for.')

# tf.app.flags.DEFINE_integer('shuffle', True,
#                             'Whether to shuffle dataset.')


# # Evaluation
# tf.app.flags.DEFINE_integer('min_eval_frequency', 1024,
#                             'Frequency to do evaluation run.')


# # Globals
# tf.app.flags.DEFINE_integer('random_seed', 1234,
#                             'The extremely random seed.')

# tf.app.flags.DEFINE_boolean('use_jit_xla', False,
#                             'Whether to use XLA compilation..')

# # Hyperparameters
# tf.app.flags.DEFINE_string(
#   'hyperparameters_path',
#   'alignment/models/configurations/single_layer.json',
#   'The path to the hyperparameters.')


# def run_experiment(unused_argv):
#   """Run the training experiment."""
#   hyperparameters_dict = FLAGS.__flags

#   # Build the hyperparameters object
#   params = HParams(**hyperparameters_dict)

#   # Set the seeds
#   np.random.seed(params.random_seed)
#   tf.set_random_seed(params.random_seed)

#   # Initialise the run config
#   run_config = tf.contrib.learn.RunConfig()

#   # Use JIT XLA
#   session_config = tf.ConfigProto()
#   if params.use_jit_xla:
#     session_config.graph_options.optimizer_options.global_jit_level = (
#       tf.OptimizerOptions.ON_1)

#   # Clean the model directory
#   if os.path.exists(params.model_dir) and params.clean_model_dir:
#     shutil.rmtree(params.model_dir)

#   # Update the run config
#   run_config = run_config.replace(tf_random_seed=params.random_seed)
#   run_config = run_config.replace(model_dir=params.model_dir)
#   run_config = run_config.replace(session_config=session_config)
#   run_config = run_config.replace(
#     save_checkpoints_steps=params.min_eval_frequency)

#   # Output relevant info for inference
#   ex.save_dict_json(d=params.values(),
#                     path=os.path.join(params.model_dir, 'params.dict'),
#                     verbose=True)
#   ex.save_obj(obj=params,
#               path=os.path.join(params.model_dir, 'params.pkl'), verbose=True)

#   estimator = learn_runner.run(
#     experiment_fn=ex.experiment_fn,
#     run_config=run_config,
#     schedule='train_and_evaluate',
#     hparams=params)


# if __name__ == '__main__':
#   tf.app.run(main=run_experiment)

# 24, 48: BasicLSTM
# 339, 685: cudnn_gru (14x speedup)
###########################################################################
# INFO:tensorflow:Running local_init_op.
# INFO:tensorflow:Done running local_init_op.
# INFO:tensorflow:Saving checkpoints for 1 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:loss = 2.572331, step = 0
# INFO:tensorflow:Saving checkpoints for 11 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:global_step/sec: 0.429643
# INFO:tensorflow:loss = 1.8736705, step = 10 (23.276 sec)
# INFO:tensorflow:Saving checkpoints for 21 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:global_step/sec: 0.433559
# INFO:tensorflow:loss = 1.573603, step = 20 (23.065 sec)
# INFO:tensorflow:Saving checkpoints for 24 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:Loss for final step: 1.5242364.
# INFO:tensorflow:Calling model_fn.
# INFO:tensorflow:Done calling model_fn.
# INFO:tensorflow:Starting evaluation at 2018-04-29-23:45:20
# INFO:tensorflow:Graph was finalized.
# 2018-04-30 07:45:20.898384: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
# 2018-04-30 07:45:20.898447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-04-30 07:45:20.898457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
# 2018-04-30 07:45:20.898464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
# 2018-04-30 07:45:20.898660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10693 MB memory) -> physical GPU (device: 0, name: TITAN V, pci bus id: 0000:02:00.0, compute capability: 7.0)
# INFO:tensorflow:Restoring parameters from ./checkpoints/v4/model.ckpt-24
# INFO:tensorflow:Running local_init_op.
# INFO:tensorflow:Done running local_init_op.
# INFO:tensorflow:Evaluation [1/10]
# INFO:tensorflow:Evaluation [2/10]
# INFO:tensorflow:Evaluation [3/10]
# INFO:tensorflow:Evaluation [4/10]
# INFO:tensorflow:Evaluation [5/10]
# INFO:tensorflow:Evaluation [6/10]
# INFO:tensorflow:Evaluation [7/10]
# INFO:tensorflow:Evaluation [8/10]
# INFO:tensorflow:Evaluation [9/10]
# INFO:tensorflow:Evaluation [10/10]
# INFO:tensorflow:Finished evaluation at 2018-04-29-23:45:30
# INFO:tensorflow:Saving dict for global step 24: accuracy = 0.5237901, global_step = 24, loss = 1.7229433, precision = 1.0, recall = 1.0
# INFO:tensorflow:Calling model_fn.
# INFO:tensorflow:Done calling model_fn.
# INFO:tensorflow:Create CheckpointSaverHook.
# INFO:tensorflow:Graph was finalized.
# 2018-04-30 07:45:32.799050: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
# 2018-04-30 07:45:32.799118: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2018-04-30 07:45:32.799127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
# 2018-04-30 07:45:32.799136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
# 2018-04-30 07:45:32.799328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10693 MB memory) -> physical GPU (device: 0, name: TITAN V, pci bus id: 0000:02:00.0, compute capability: 7.0)
# INFO:tensorflow:Restoring parameters from ./checkpoints/v4/model.ckpt-24
# INFO:tensorflow:Running local_init_op.
# INFO:tensorflow:Done running local_init_op.
# INFO:tensorflow:Saving checkpoints for 25 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:loss = 1.9905627, step = 24
# INFO:tensorflow:Saving checkpoints for 35 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:global_step/sec: 0.409827
# INFO:tensorflow:loss = 1.6847508, step = 34 (24.401 sec)
# INFO:tensorflow:Saving checkpoints for 45 into ./checkpoints/v4/model.ckpt.
# INFO:tensorflow:global_step/sec: 0.422631
# INFO:tensorflow:loss = 1.5193172, step = 44 (23.661 sec)

########################################################################
