# MNIST RNN classifier

"""
150 recurrent neurons
fully connected layer containing 10 neurons
softmax layer
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(8765)
np.random.seed(8765)

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

# Use He initalization
he_init = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
    swap_memory=True)
logits = fully_connected(states, n_outputs, activation_fn=None,
    weights_initializer=he_init)
x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
    logits=logits)
loss = tf.reduce_mean(x_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for k in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_rnn = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_rnn, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_rnn, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, 'Train acc: ', acc_train, 'Test acc: ', acc_test)