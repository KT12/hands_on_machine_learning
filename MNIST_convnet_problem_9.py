# Attempting to get very high accuracy with MNIST convnet

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Get data
mnist = input_data.read_data_sets("/tmp/data")

h = 28
w = 28
channels = 1
n_inputs = h * w

# Architecture Conv 64, Max Pool, Conv 32, Average Pool, FC

conv1_filters = 64
conv1_k = 3
conv1_s = 1
conv1_pad = 'SAME'

pool1_filters = 64
pool1_k = 3
pool1_s = 1
pool1_pad = 'SAME'

conv2_filters = 32
conv2_k = 3
conv2_s = 2
conv2_pad = 'SAME'

pool2_filters = 32
pool2_k = 3
pool2_s = 2
pool2_pad = 'SAME'

n_fc1 = 32
n_outputs = 10
fc1_dropout = 0.5

# Construct graph
graph = tf.Graph()

with graph.as_default():
    with tf.device("/cpu:0"):
        is_training = tf.placeholder(tf.bool, shape=(), name='Is_Training')
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
            X_reshaped = tf.reshape(X, shape=[-1, h, w, channels])
            y = tf.placeholder(tf.int32, shape=[None], name='y')
        
        with tf.name_scope('conv_1'):
            conv_1 = tf.layers.conv2d(X_reshaped, filters=conv1_filters,
                kernel_size=conv1_k, strides=conv1_s, padding=conv1_pad,
                activation=tf.nn.elu, name='conv_1')
        
        with tf.name_scope('pool_1'):
            pool_1 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,1,1,1],
                padding=pool1_pad)
        
        with tf.name_scope('conv_2'):
            conv_2 = tf.layers.conv2d(pool_1, filters=conv2_filters,
                kernel_size=conv2_k, strides=conv2_s, padding=conv2_pad,
                activation=tf.nn.elu, name='conv_2')
        
        with tf.name_scope('pool_2'):
            pool_2 = tf.nn.avg_pool(conv_2, ksize=[1,3,3,1], strides=[1,2,2,1],
                padding=pool2_pad)
            # Have to flatten pool_2 for fully connected layer
            pool_2_flat = tf.reshape(pool_2, shape=[-1, 7 * 7 * pool2_filters])
        
        with tf.name_scope('fully_connected_1'):
            fc_1 = tf.layers.dense(pool_2_flat, n_fc1, activation=tf.nn.elu,
                name='fully_connected_1')
            fc_1_drop = tf.layers.dropout(fc_1, rate=fc1_dropout,
                training=is_training)
        
        with tf.name_scope('output'):
            logits = tf.layers.dense(fc_1_drop, n_outputs, name='output')
            # y_probs = tf.nn.softmax(logits, name='y_probs')
        
        with tf.name_scope('train'):
            x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(x_entropy)
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)
        
        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        init = tf.global_variables_initializer()

# Run calcs

n_epochs = 25
batch_size = 16

with tf.Session(graph=graph) as sess:
    #with tf.device("/cpu:0"):
        init.run()
        for epoch in range(n_epochs):
            for k in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={is_training: True,
                    X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={is_training: False,
                X: X_batch, y: y_batch})
            acc_test  = accuracy.eval(feed_dict={is_training: False,
                X: mnist.test.images, y: mnist.test.labels})
            print(epoch, "Train acc: ", acc_train, 'Test acc: ', acc_test)