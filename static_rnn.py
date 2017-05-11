# Static unrolling through time
import tensorflow as tf
import numpy as np

tf.set_random_seed(765)
np.random.seed(765)

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
    dtype=tf.float32)

Y0, Y1 = output_seqs