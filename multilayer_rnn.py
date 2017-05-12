import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_steps = 28
n_inputs = 28
n_neurons1 = 150
n_neurons2 = 100
n_outputs = 10

learning_rate = 0.001

X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

hidden1 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons1, activation=tf.nn.relu)
hidden2 = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons2, activation=tf.nn.relu)

multi_layer_cell = tf.contrib.rnn.MultiRNNCell([hidden1, hidden2])

outputs, states_tuple = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32,
    swap_memory=True)

states = tf.concat(axis=1, values=states_tuple)
logits = tf.layers.dense(states, n_outputs)

x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss=tf.reduce_mean(x_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for k in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y:y_test})
        print(epoch, "Train acc: ", acc_train, "Test acc: ", acc_test)