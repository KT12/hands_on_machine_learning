# Implementing Gradient Descent using TensorFlow

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Give every run a different log directory name with timestamp
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Get data
X_moons, y_moons = make_moons(n_samples=1000, noise=0.1)
m, n = X_moons.shape

# Learning parameters
n_epochs = 1000
learning_rate = 0.005
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

# Transform data into usable tensors, set up theta
scaled_X = StandardScaler().fit_transform(X_moons)
# Add column of 1's for bias term
moons = np.c_[np.ones((m, 1)), scaled_X]

# X and Y must be placeholders for batch gradient descent
X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')

# Construct graph
def logistic_regression(X,y):
    # y_pred is result of sigmoid
    y_pred = 1/(1 + tf.exp(tf.matmul(X, theta)))
    error = y_pred - y
    return y_pred, error

y_pred, error = logistic_regression(X, y)
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Define function to get batches
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = moons[indices]
    y_batch = y_moons.reshape(-1,1)[indices]
    return X_batch, y_batch

init = tf.global_variables_initializer()
# Add saver to restore model later
saver = tf.train.Saver()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# Computation
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X:X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            save_path =saver.save(sess, '/tmp/my_model.ckpt')
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
print(best_theta)