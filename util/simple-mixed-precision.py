import tensorflow as tf
import numpy as np
import time

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

def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]

def create_simple_model(nbatch, nin, nout, dtype):
    """A simple softmax model."""
    data    = tf.placeholder(dtype, shape=(nbatch, nin))
    weights = tf.get_variable('weights', (nin, nout), dtype)
    biases  = tf.get_variable('biases',        nout,  dtype,
                              initializer=tf.zeros_initializer())
    logits  = tf.matmul(data, weights) + biases
    target  = tf.placeholder(tf.float32, shape=(nbatch, nout))
    # Note: The softmax should be computed in float32 precision
    loss    = tf.losses.softmax_cross_entropy(
        target, tf.cast(logits, tf.float32))
    return data, target, loss

if __name__ == '__main__':
    nbatch = 1024
    nin    = 2000
    nout   = 200
    learning_rate = 0.1
    momentum      = 0.9
    loss_scale    = 128
    # dtype         = tf.float32
    dtype         = tf.float16
    tf.set_random_seed(1234)
    np.random.seed(4321)

    # Create training graph
    with tf.device('/gpu:0'), \
         tf.variable_scope(
             # Note: This forces trainable variables to be stored as float32
             'fp32_storage', custom_getter=float32_variable_storage_getter):
        data, target, loss = create_simple_model(nbatch, nin, nout, dtype)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Note: Loss scaling can improve numerical stability for fp16 training
        grads = gradients_with_loss_scaling(loss, variables, loss_scale)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_step_op = optimizer.apply_gradients(zip(grads, variables))
        init_op = tf.global_variables_initializer()

    # Run training
    sess = tf.Session()
    sess.run(init_op)
    np_data   = np.random.normal(size=(nbatch, nin)).astype(np.float16)
    np_target = np.zeros((nbatch, nout), dtype=np.float32)
    np_target[:,0] = 1
    print('Step Loss')
    now = time.time()
    for step in range(50):
        np_loss, _ = sess.run([loss, training_step_op],
                              feed_dict={data: np_data, target: np_target})
        print('%4i %6f' % (step + 1, np_loss))
    print(time.time()-now)