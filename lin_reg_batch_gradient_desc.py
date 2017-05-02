# Implementing Gradient Descent using TensorFlow

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Get data
housing = fetch_california_housing()
m, n = housing.data.shape

# Learning parameters
n_epochs = 1000
learning_rate = 0.01
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

# Transform data into usable tensors, set up theta
scaled_X = StandardScaler().fit_transform(housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_X]

# X and Y must be placeholders for batch gradient descent
X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

# Construct graph
y_pred = tf.matmul(X, theta, name='y_pred')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Define function to get batches
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

init = tf.global_variables_initializer()

# Computation
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()

print(best_theta)