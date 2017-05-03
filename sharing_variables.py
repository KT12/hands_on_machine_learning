# Sharing variables
# Example code to use a shared threshold variable for all ReLUs
# Create var and pass it to relu() function

import tensorflow as tf
import numpy as np

def relu(X, threshold):
    with tf.name_scope('relu'):
        #[...........]
        return tf.maximum(z, threshold, name='max')

threshold = tf.Variable(0.0, name='threshold')
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name='output')

# Another alternative
# Set the shared var as an attribute of relu() upon first call

def relu(X):
    with tf.name_scope('relu'):
        if not hasattr(relu, 'threshold'):
            relu.threshold = tf.Variable(0.0, name='threshold')
            # [.........]
        return tf.maximum(z, relu.threshold, name='max')

# Third option use get_variable() to create shared var if it does not exist
# or re-use if it does exist

with tf.variable_scope('relu'):
    threshold = tf.get_variable('thershold', shape=(),
        initializer=tf.constant_initializer(0.0))

# If var has been created by an earlier call to get_variable(),
# it will raise an exception.  If you want to re-use, need to be explicit:

with tf.variable_scope('relu', reuse=True):
    threshold = tf.get_variable('threshold')

# Alternatively:

with tf.variable_scope('relu') as scope:
    scope.reuse_variables()
    threshold = tf.get_variable('threshold')

# Pulling it all together

def relu(X):
    with tf.variable_scope('relu', reuse=True):
        threshold = tf.get_variable('threshold')
        # [......]
        return tf.maximum(z, threshold, name='max')

X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
with tf.variable_scope('relu'):
    threshold = tf.get_variable('threshold', shape=(),
                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add(relus, name='output')