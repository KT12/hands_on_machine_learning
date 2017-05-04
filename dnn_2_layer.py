# Train a DNN using TF

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs = 28 * 28
n_hidden_1 = 512
n_hidden_2 = 64
n_outputs = 10

tf.set_random_seed(5)

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z

# with tf.name_scope('DNN'):
#     hidden_1 = neuron_layer(X, n_hidden_1, 'Hidden_1', activation='relu')
#     hidden_2 = neuron_layer(hidden_1, n_hidden_2, 'Hidden_2', activation='relu')
#     logits = neuron_layer(hidden_2, n_outputs, 'Outputs')

# Use fully_connected instead
with tf.name_scope('DNN'):
    hidden_1 = fully_connected(X, n_hidden_1, scope='Hidden_1')
    hidden_2 = fully_connected(hidden_1, n_hidden_2, scope='Hidden_2')
    logits = fully_connected(hidden_2, n_outputs, scope='Outputs', activation_fn=None)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
# saver = tf.train.Saver()

# Execution phase

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 250
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        if epoch % 10 == 0:
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    # save_path = saver.save(sess, "./my_model_final.ckpt")

# Interestingly, this 2 layer DNN did better than the dnn_4_layer.py network.