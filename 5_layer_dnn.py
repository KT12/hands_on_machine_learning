# 5 Layer DNN for digits 0 to 4 of MNIST
# 100 neurons in each layer
# ADAM optimization and early stopping

# Import modules and data
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

# Set random seed
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

# Get data and separate digits 0-4 out
mnist = input_data.read_data_sets("/tmp/data/")
X_images, y_images = mnist.train.images, mnist.train.labels
X_images_test, y_images_test = mnist.test.images, mnist.test.labels

# Create 'index' and subset of MNIST
indices = [idx for idx in range(len(y_images)) if y_images[idx] < 5]
X_masked_train = X_images[indices]
y_masked_train = y_images[indices]

# Do same for test set
indices_test = [idx for idx in range(len(y_images_test)) if y_images_test[idx] < 5]
X_test = X_images_test[indices_test]
y_test = y_images[indices_test]

# Construct graph
# Use He initalization
he_init = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('DNN'):
    hidden_1 = fully_connected(X, n_hidden_1, activation_fn=tf.nn.elu, scope='Hidden_1')
    hidden_2 = fully_connected(hidden_1, n_hidden_2, activation_fn=tf.nn.elu, scope='Hidden_2')
    hidden_3 = fully_connected(hidden_2, n_hidden_3, activation_fn=tf.nn.elu, scope='Hidden_3')
    hidden_4 = fully_connected(hidden_3, n_hidden_4, activation_fn=tf.nn.elu, scope='Hidden_4')
    hidden_5 = fully_connected(hidden_4, n_hidden_5, activation_fn=tf.nn.elu, scope='Hidden_5')
    logits = fully_connected(hidden_5, n_outputs, activation_fn=None, scope='Outputs')

with tf.name_scope('Loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='Loss')

learning_rate = 0.01

with tf.name_scope('Train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('Eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# Execution

n_epochs = 100
batch_size = 50
batches = len(y_masked_train//batch_size)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for k in range(batches):
            X_batch = X_masked_train[k*batches:k*batches+batches]
            y_batch = y_masked_train[k*batches:k*batches+batches]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        if epoch % 10 == 0:
            print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)
            