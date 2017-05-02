# Implementing Gradient Descent using TensorFlow

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Get data
housing = fetch_california_housing()
m, n = housing.data.shape

# Learning parameters
n_epochs = 2500
learning_rate = 0.025

# Transform data into usable tensors, set up theta
scaled_X = StandardScaler().fit_transform(housing.data)
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_X]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

# Construct graph
y_pred = tf.matmul(X, theta, name='y_pred')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
# gradients = (2/m) * tf.matmul(tf.transpose(X), error)
# Use autodiff instead
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Alternate optimization
optimizer2 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

# Computation
with tf.Session() as sess:
    sess.run(init)
    print("Learning rate: ", learning_rate)
    print(theta.eval())
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch ', epoch, 'MSE = ', mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()

print(best_theta)