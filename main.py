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
import argparse
import ast
import functools
import sys
import os

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import adaptive_clipping_fn
import numpy as np

# Disable cpp warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Show debugging output, default: tf.logging.INFO
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = None

aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'

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
        ) # 256 MB

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

    # Our inputs are variable length, so pad them.
    if mode != tf.estimator.ModeKeys.PREDICT:
        dataset = dataset.padded_batch(
            batch_size=params.batch_size,
            # A `tf.int64` scalar `tf.Tensor`, representing the number of
            # consecutive elements of this dataset to combine in a single batch.
            padded_shapes=({'protein': [None], 'lengths': []}, [None])
            # A nested structure of `tf.TensorShape` or
            # `tf.int64` vector tensor-like objects representing the shape
            # to which the respective component of each input element should
            # be padded prior to batching. Any unknown dimensions
            # (e.g. `tf.Dimension(None)` in a `tf.TensorShape` or `-1` in a
            # tensor-like object) will be padded to the maximum size of that
            # dimension in each batch.
        )
    else:
        dataset = dataset.padded_batch(
            batch_size=params.batch_size,
            padded_shapes={'protein': [None], 'lengths': []}
        )
        
    dataset = dataset.prefetch(
        buffer_size=params.prefetch_buffer #  batches
        # A `tf.int64` scalar `tf.Tensor`, representing the
        # maximum number batches that will be buffered when prefetching.
    )

    return dataset

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

def model_fn(features, labels, mode, params, config):
    # labels shape=(batch_size, sequence_length), dtype=int32
    is_train = mode == tf.estimator.ModeKeys.TRAIN

    protein = features['protein']
    # protein shape=(batch_size, sequence_length), dtype=int32
    lengths = features['lengths']
    # lengths shape=(batch_size, ), dtype=int32
    # Embedding layer
    with tf.variable_scope('embedding_1', values=[features]):
        embeddings = tf.contrib.framework.model_variable(
            name='embeddings', 
            shape=[params.vocab_size, params.embed_dim],
            dtype=tf.float32, # default: tf.float32
            initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            trainable=True,
        ) # vocab_size * embed_dim = 28 * 32 = 896
        # tf.Variable 'embedding_matrix:0' shape=(vocab_size, embed_dim) dtype=float32
        embedded = tf.nn.embedding_lookup(
            params=embeddings, 
            ids=protein,
            name='embedding_lookup'
        )
        # tf.Tensor: shape=(batch_size, sequence_length, embed_dim), dtype=float32
        dropped_embedded = tf.layers.dropout(
            inputs=embedded, 
            rate=params.embedded_dropout, # 0.2
            noise_shape=None, # [batch_size, 1, embed_dim]
            training=is_train,
            name='dropout'
        )

    # temporal convolution
    with tf.variable_scope('conv_1'):
        convolved = tf.layers.conv1d(
            inputs=dropped_embedded,
            filters=params.conv_1_filters, # 32
            kernel_size=params.conv_1_kernel_size, # 7
            strides=params.conv_1_strides, # 1
            padding='same',
            data_format='channels_last',
            dilation_rate=1,
            activation=tf.nn.relu, # relu6, default: linear
            use_bias=True,
            kernel_initializer=None, # default: tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
            bias_initializer=tf.zeros_initializer(),
            trainable=True,
            name='conv1d',
            reuse=None
        ) # (kernel_size * conv_1_conv1d_filters + use_bias) * embed_dim = (7 * 32 + 1) * 32 = 7200
        dropped_convolved = tf.layers.dropout(
            inputs=convolved, 
            rate=params.conv_1_dropout, # 0.2
            noise_shape=None, # [batch_size, 1, embed_dim]
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
        convolved = tf.transpose(dropped_convolved, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1,
            num_units=params.rnn_num_units,
            direction="bidirectional",
            name='CudnnGRU1'
        )
        outputs, _ = lstm(convolved)
        # Convert back from time-major outputs to batch-major outputs.
        outputs = tf.transpose(outputs, [1, 0, 2])

    
    # output layer
    with tf.variable_scope('output_1'):
        logits = tf.layers.dense(
            inputs=outputs, 
            units=params.num_classes,
            activation=None,
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
        # logits shape=(batch_size, sequence_length, num_classes), dtype=float32

    # predictions
    with tf.variable_scope('predictions'):
        predictions = {
            'logits': logits, 
            # Add `softmax_tensor` to the graph. 
            'probabilities': tf.nn.softmax(logits=logits, axis=-1, name='softmax_tensor'),
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=-1)
        }

    # loss
    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, 
            logits=logits
        )
        # losses shape=(batch_size, sequence_length), dtype=float32
        mask = tf.to_float(tf.sign(labels)) # 0 = 'PAD'
        masked_losses = losses * mask
        # average across batch_size and sequence_length
        loss = tf.reduce_sum(masked_losses) / tf.to_float(tf.reduce_sum(lengths))

    # optimizer
    with tf.variable_scope('optimizer'):
        # clip_gradients = params.gradient_clipping_norm
        global_step=tf.train.get_global_step()
        clip_gradients = adaptive_clipping_fn(
            std_factor=2.,
            decay=0.95,
            static_max_norm=None,
            global_step=global_step,
            report_summary=True,
            epsilon=1e-8,
            name=None
        )
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, 
                global_step,
                decay_steps=params.decay_steps, # 100000
                decay_rate=params.decay_rate, # 0.96
                staircase=False,
                name=None
            )
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=params.learning_rate, # 0.1
            optimizer='Adam',
            gradient_noise_scale=None,
            gradient_multipliers=None,
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn,
            update_ops=None,
            variables=None,
            name=None,
            summaries=[
                'learning_rate', 
                'loss', 
                'gradients', 
                'gradient_norm'
            ],
            colocate_gradients_with_ops=False,
            increment_global_step=True
        )

    scaffold = tf.train.Scaffold(saver = tf.train.Saver(sharded = False, allow_empty = True))
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            # matches / total
            'accuracy': tf.metrics.accuracy(
                labels=labels, 
                predictions=predictions['classes'], 
                weights=mask
            ),
            # true_positives / (true_positives + false_positives)
            'precision': tf.metrics.precision(
                labels=labels, 
                predictions=predictions['classes'], 
                weights=mask
            ),
            # true_positives / (true_positives + false_negatives)
            'recall': tf.metrics.recall(
                labels=labels, 
                predictions=predictions['classes'], 
                weights=mask
            )
        },
        export_outputs={
            'predictions': tf.estimator.export.PredictOutput(predictions)
        },
        training_chief_hooks=None,
        training_hooks=None,
        scaffold=scaffold,
        evaluation_hooks=None,
        prediction_hooks=None
    )

# https://github.com/tensorflow/models/blob/69cf6fca2106c41946a3c395126bdd6994d36e6b/tutorials/rnn/quickdraw/train_model.py


def create_estimator_and_specs(run_config):
    """Creates an Estimator, TrainSpec and EvalSpec."""
    model_params = tf.contrib.training.HParams(
        num_gpus=FLAGS.num_gpus,
        num_cpu_threads=FLAGS.num_cpu_threads,

        tfrecord_pattern={
            tf.estimator.ModeKeys.TRAIN: FLAGS.training_data,
            tf.estimator.ModeKeys.EVAL: FLAGS.eval_data,
        },
        dataset_buffer=FLAGS.dataset_buffer, # 256 MB
        dataset_parallel_reads=FLAGS.dataset_parallel_reads, # 1
        shuffle_buffer=FLAGS.shuffle_buffer, # 16 * 1024 examples
        repeat_count=FLAGS.repeat_count, # -1 = Repeat the input indefinitely.
        batch_size=FLAGS.batch_size,
        prefetch_buffer=FLAGS.prefetch_buffer, #  batches
        
        vocab_size=FLAGS.vocab_size, # 28
        embed_dim=FLAGS.embed_dim, # 32
        embedded_dropout=FLAGS.embedded_dropout, # 0.2
        
        conv_1_filters=FLAGS.conv_1_filters, # 32
        conv_1_kernel_size=FLAGS.conv_1_kernel_size, # 7
        conv_1_strides=FLAGS.conv_1_strides, # 1
        conv_1_dropout=FLAGS.conv_1_dropout, # 0.2
        
        rnn_num_units=FLAGS.rnn_num_units,
        
        num_classes=FLAGS.num_classes,
        
        decay_steps=FLAGS.decay_steps,
        decay_rate=FLAGS.decay_rate,
        learning_rate=FLAGS.learning_rate

        # num_layers=FLAGS.num_layers,
        # num_conv=ast.literal_eval(FLAGS.num_conv),
        # conv_len=ast.literal_eval(FLAGS.conv_len),
        # gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        # cell_type=FLAGS.cell_type,
        # batch_norm=FLAGS.batch_norm
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn, 
        # A function that provides input data for training as minibatches.
        max_steps=FLAGS.steps or None, # 0
        # Positive number of total steps for which to train model. If None, train forever.
        hooks=None
        # Iterable of `tf.train.SessionRunHook` objects to run
        # on all workers (including chief) during training.
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        # A function that constructs the input data for evaluation.
        steps=10, # 100,
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
        start_delay_secs=120,
        # Int. Start evaluating after waiting for this many seconds.
        throttle_secs=30*60
        # Do not re-evaluate unless the last evaluation was
        # started at least this many seconds ago. Of course, evaluation does not
        # occur if no new checkpoints are available, hence, this is the minimum.
    )

    return estimator, train_spec, eval_spec

def main(unused_args):
    # Hardware info
    FLAGS.num_gpus = FLAGS.num_gpus or tf.contrib.eager.num_gpus()
    FLAGS.num_cpu_threads = FLAGS.num_cpu_threads or os.cpu_count()

    # Set the seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    # Use JIT XLA
    # session_config = tf.ConfigProto(log_device_placement=True)
    session_config = tf.ConfigProto(allow_soft_placement=True)
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
            tf_random_seed=FLAGS.random_seed, 
            # Random seed for TensorFlow initializers.
            # Setting this value allows consistency between reruns.
            save_summary_steps=FLAGS.save_summary_steps,  # 10
            # The frequency, in number of global steps, that the
            # summaries are written to disk using a default SummarySaverHook. If both
            # `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
            # the default summary saver isn't used. Default 100.
            # save_checkpoints_steps=FLAGS.save_checkpoints_steps, # 100
            # Save checkpoints every this many steps.
            save_checkpoints_secs=FLAGS.save_checkpoints_secs, # 10 * 60
            # Save checkpoints every this many seconds with 
            # CheckpointSaverHook. Can not be specified with `save_checkpoints_steps`. 
            # Defaults to 600 seconds if both `save_checkpoints_steps` and 
            # `save_checkpoints_secs` are not set in constructor.  
            # If both `save_checkpoints_steps` and `save_checkpoints_secs` are None, 
            # then checkpoints are disabled.
            keep_checkpoint_max=FLAGS.keep_checkpoint_max, # 10
            # Maximum number of checkpoints to keep.  As new checkpoints
            # are created, old ones are deleted.  If None or 0, no checkpoints are
            # deleted from the filesystem but only the last one is kept in the
            # `checkpoint` file.  Presently the number is only roughly enforced.  For
            # example in case of restarts more than max_to_keep checkpoints may be
            # kept.
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours, # 1
            # keep an additional checkpoint
            # every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
            # kept for every 0.5 hours of training; if `N` is 10, an additional
            # checkpoint is kept for every 10 hours of training.
            # Defaults to 10,000 hours.
            log_step_count_steps=FLAGS.log_step_count_steps, # 10
            # The frequency, in number of global steps, that the
            # global step/sec will be logged during training.
            session_config=session_config))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d10-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d10-s20-test.tfrecords --model_dir=./checkpoints/v3 --batch_size=64
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0-v1
# full dataset, batchsize=8 NaN loss, batchsize=4 works
# python main.py --training_data=/home/hotdogee/datasets/pfam-regions-d0-s20-train.tfrecords --eval_data=/home/hotdogee/datasets/pfam-regions-d0-s20-test.tfrecords --model_dir=./checkpoints/d0b4 --num_classes=16715 --batch_size=4
# python main.py --model_dir=./checkpoints/win-d10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')

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
        '--num_classes',
        type=int,
        # default=16712 + 3, # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
        default=10 + 3, # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
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
        '--model_dir',
        type=str,
        default='./checkpoints/v2',
        help='Path for saving model checkpoints during training')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
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
        default=10,
        help='The maximum number of recent checkpoint files to keep.')
    parser.add_argument(
        '--keep_checkpoint_every_n_hours',
        type=float,
        default=1,
        help='Keep an additional checkpoint every `N` hours.')
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=100,
        help='The frequency, in number of global steps, that the global step/sec will be logged during training.')
        
    parser.add_argument(
        '--steps',
        type=int,
        default=0, # 100000,
        help='Number of training steps, if 0 train forever.')
    
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
        default=-1,
        help='Number of times the dataset should be repeated.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size to use for training/evaluation.')
    parser.add_argument(
        '--prefetch_buffer',
        type=int,
        default=64,
        help='Maximum number of batches that will be buffered when prefetching.')

    parser.add_argument(
        '--vocab_size',
        type=int,
        default=len(aa_list) + 1, # 'PAD'
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
        '--decay_steps',
        type=int,
        default=100000, # num_batches_per_epoch * num_epochs_per_decay(8)
        help='Decay learning_rate by decay_rate every decay_steps.')
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.96,
        help='Learning rate decay rate.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate used for training.')

    parser.add_argument(
        '--num_layers',
        type=int,
        default=3,
        help='Number of recurrent neural network layers.')
    parser.add_argument(
        '--num_conv',
        type=str,
        default='[48, 64, 96]',
        help='Number of conv layers along with number of filters per layer.')
    parser.add_argument(
        '--conv_len',
        type=str,
        default='[5, 5, 3]',
        help='Length of the convolution filters.')
    parser.add_argument(
        '--gradient_clipping_norm',
        type=float,
        default=9.0,
        help='Gradient clipping norm used during training.')
    parser.add_argument(
        '--cell_type',
        type=str,
        default='lstm',
        help='Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.')
    parser.add_argument(
        '--batch_norm',
        type='bool',
        default='False',
        help='Whether to enable batch normalization or not.')

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
