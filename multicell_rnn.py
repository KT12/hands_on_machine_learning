# Simple Multicell RNN
import tensorflow as tf
import numpy as np

# Need to figure out why it's not working
# Might be b/c tf.contrib changes frequently
n_inputs = 2
n_neurons = 100
n_layers = 3
n_steps = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell =tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()

X_batch = np.random.rand(2, n_steps, n_inputs)

with tf.Session() as sess:
    init.run()
    output_val, stats_val = sess.run([outputs, states], feed_dict={X: X_batch})
    print(outputs_val.shape)