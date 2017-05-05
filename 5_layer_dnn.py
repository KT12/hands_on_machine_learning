# 5 Layer DNN for digits 0 to 4 of MNIST

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

tf.set_random_seed(8)
np.random.seed(8)

n_inputs = 28 * 28
n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 100
n_hidden_5 = 100

# Use only digits 0 to 4
n_outputs = 5

he_init = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('DNN'):
    hidden_1 = fully_connected(X, n_hidden_1, scope='Hidden_1')
    hidden_2 = fully_connected(hidden_1, n_hidden_2, scope='Hidden_2')
    hidden_3 = fully_connected(hidden_2, n_hidden_3, scope='Hidden_3')
    hidden_4 = fully_connected(hidden_3, n_hidden_4, scope='Hidden_4')
    hidden_5 = fully_connected(hidden_4, n_hidden_5, scope='Hidden_5')
    