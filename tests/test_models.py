import unittest

import tensorflow as tf

from models import SimpleModel


class TestSimpleModel(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        inp = tf.random.normal(shape=(5, 256 * 256 * 3))
        self.out = self.model.forward(inp)


    def test_forward(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print(sess.run(self.out))
