import tensorflow as tf
import sonnet


def get_variable(name, shape):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    return var


def get_bias_variable(name, shape):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.zeros_initializer(dtype=tf.float32))
    return var


def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)


def normalize(inp, activation, is_training):
    bn = sonnet.BatchNorm(decay_rate=0.0, offset=False)
    out = bn(inp, is_training=is_training, test_local_stats=False)

    if activation:
        out = activation(out)

    return out
