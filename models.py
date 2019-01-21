import os
from abc import abstractmethod

import tensorflow as tf
import numpy as np

from meta_layers import dense, conv2d
from utils import get_variable, get_bias_variable, normalize, xent


class Model(object):
    def __init__(self, output_dimension, update_lr):
        self.output_dimension = output_dimension
        self.update_lr = update_lr
        self.weights = self.construct_weights()
        self.is_training = tf.placeholder(tf.bool)
        self.saver = tf.train.Saver()

    def define_loss(self, labels):
        self.loss = tf.reduce_mean(xent(self.out, labels))

    def define_update_op(self, labels, with_batch_norm_dependency=False):
        self.define_loss(labels)
        grads = tf.gradients(self.loss, list(self.weights.values()))
        gradients = dict(zip(self.weights.keys(), grads))

        assign_ops = list()
        for key in self.weights.keys():
            assign_ops.append(tf.assign(self.weights[key], self.weights[key] - self.update_lr * gradients[key]))

        if with_batch_norm_dependency:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.op = tf.group(assign_ops)
        else:
            self.op = tf.group(assign_ops)

    def save(self, sess, path, step=1):
        self.saver.save(sess, os.path.join(path, self.get_name()), global_step=step)

    def load(self, sess, address):
        last_model_file_name = sorted(
            [f for f in os.listdir(address) if f != 'checkpoint'],
            key=lambda x: int(x[len(self.get_name()) + 1: x.rindex('.')])
        )[-1]

        last_model_loading_name = last_model_file_name[:last_model_file_name.index('.')]

        self.saver.restore(sess, os.path.join(address, last_model_loading_name))

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def construct_weights(self):
        pass

    @abstractmethod
    def forward(self, inp, weights):
        pass


class MAMLMiniImagenetModel(Model):
    def construct_weights(self):
        weights = dict()
        weights['kc1'] = get_variable('kc1', (3, 3, 3, 64))
        weights['kc2'] = get_variable('kc2', (3, 3, 64, 64))
        weights['kc3'] = get_variable('kc3', (3, 3, 64, 64))
        weights['kc4'] = get_variable('kc4', (3, 3, 64, 64))
        weights['kd1'] = get_variable('kd1', (64 * 5 * 5, self.output_dimension))

        weights['bc1'] = get_bias_variable('bc1', (64,))
        weights['bc2'] = get_bias_variable('bc2', (64,))
        weights['bc3'] = get_bias_variable('bc3', (64,))
        weights['bc4'] = get_bias_variable('bc4', (64,))
        weights['bd1'] = get_bias_variable('bd1', (self.output_dimension,))

        return weights

    def forward(self, inp, weights=None):
        if weights is None:
            weights = self.weights

        inp = tf.reshape(inp, self.get_input_shape())
        conv1 = conv2d(inp, weights['kc1'], weights['bc1'], padding='SAME', name='c1')
        conv1 = normalize(conv1, activation=tf.nn.relu, is_training=self.is_training)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

        conv2 = conv2d(conv1, weights['kc2'], weights['bc2'], padding='SAME', name='c2')
        conv2 = normalize(conv2, activation=tf.nn.relu, is_training=self.is_training)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

        conv3 = conv2d(conv2, weights['kc3'], weights['bc3'], padding='SAME', name='c3')
        conv3 = normalize(conv3, activation=tf.nn.relu, is_training=self.is_training)
        conv3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2))

        conv4 = conv2d(conv3, weights['kc4'], weights['bc4'], padding='SAME', name='c4')
        conv4 = normalize(conv4, activation=tf.nn.relu, is_training=self.is_training)
        conv4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2))

        conv4 = tf.reshape(conv4, (-1, np.prod([int(dim) for dim in conv4.get_shape()[1:]])))

        self.out = dense(conv4, weights['kd1'], weights['bd1'], activation=None, name='out')

        return self.out

    def get_name(self):
        return 'MAMLMiniImagenetModel'

    def get_input_shape(self):
        return -1, 84, 84, 3

    def define_accuracy(self, labels):
        self.accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.out), 1), tf.argmax(labels, 1))


class SimpleModel(Model):
    def construct_weights(self):
        weights = dict()

        weights['kc1'] = get_variable('kc1', (3, 3, 1, 64))
        weights['kc2'] = get_variable('kc2', (3, 3, 64, 64))
        weights['kc3'] = get_variable('kc3', (3, 3, 64, 64))
        weights['kc4'] = get_variable('kc4', (3, 3, 64, 64))
        weights['kd1'] = get_variable('kd1', (64, self.output_dimension))

        weights['bc1'] = get_bias_variable('bc1', (64,))
        weights['bc2'] = get_bias_variable('bc2', (64,))
        weights['bc3'] = get_bias_variable('bc3', (64,))
        weights['bc4'] = get_bias_variable('bc4', (64,))
        weights['bd1'] = get_bias_variable('bd1', (self.output_dimension,))

        return weights

    def forward(self, inp, weights=None):
        if weights is None:
            weights = self.weights

        inp = tf.reshape(inp, self.get_input_shape())
        conv1 = conv2d(inp, weights['kc1'], weights['bc1'], padding='SAME', name='c1')
        conv1 = normalize(conv1, activation=tf.nn.relu, is_training=self.is_training)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

        conv2 = conv2d(conv1, weights['kc2'], weights['bc2'], padding='SAME', name='c2')
        conv2 = normalize(conv2, activation=tf.nn.relu, is_training=self.is_training)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

        conv3 = conv2d(conv2, weights['kc3'], weights['bc3'], padding='SAME', name='c3')
        conv3 = normalize(conv3, activation=tf.nn.relu, is_training=self.is_training)
        conv3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2))

        conv4 = conv2d(conv3, weights['kc4'], weights['bc4'], padding='SAME', name='c4')
        conv4 = normalize(conv4, activation=tf.nn.relu, is_training=self.is_training)
        conv4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2))

        conv4 = tf.reduce_mean(conv4, [1, 2])

        self.out = dense(conv4, weights['kd1'], weights['bd1'], activation=None, name='out')

        return self.out

    def get_name(self):
        return 'SimpleModel'

    def get_input_shape(self):
        return -1, 28, 28, 1

    def define_accuracy(self, labels):
        self.accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.out), 1), tf.argmax(labels, 1))


if __name__ == '__main__':
    model = SimpleModel(output_dimension=5, update_lr=0.1)
    model.forward(tf.placeholder(tf.float32))
    model.define_update_op(tf.placeholder(tf.float32))
    print('\n'.join([str(item) for item in tf.global_variables()]))
