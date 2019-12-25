# Uses code:
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877


import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os

# First let's make some sample data. Let's say we measure displacement and velocity of the particle we are trying to
# chase, and we want the output to be our velocity.

# a single input is a 1 x 2 matrix
# a single output is a scalar

# Import data
data = pd.read_csv('data.csv')

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data = data.values

data_train = data[np.arange(0, 6), :]
X_train = data_train[:, 0:2]  # this is a training set of of distance/velocity  1 x 2 inputs from the particle
y_train = data_train[:, 2]  # this is a corresponding list of velocities that go with each pair

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Model architecture parameters
n_stocks = 2
n_neurons_1 = 4
n_neurons_2 = 2
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_2, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))

mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(y_test)
# line2, = ax1.plot(y_test*0.5)
# plt.show()

# Number of epochs and batch size
epochs = 500
batch_size = 1

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(X_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch

        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        # if np.mod(i, 5) == 0:
        # Prediction
        #   pred = net.run(out, feed_dict={X: X_test})
        #  line2.set_ydata(pred)
        # plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
        # file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
        # plt.savefig(file_name)
        # plt.pause(0.01)

# This line now saves the trained model to disk.
saver = tf.train.Saver()
save_path = saver.save(net, "./trained-model.ckpt")
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_train, Y: y_train})
print(mse_final)
# Try calculating the output for an input of [1.0 1.0]:

pred = net.run(out, feed_dict={X: np.array([[-.1, 1.1]])})
print(pred[0][0])

# This is how you would load the trained model inside your program. This should happen once when the program loads.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./trained-model.ckpt")
    print("Model restored.")
    # Check the values of the variables
    pred = net.run(out, feed_dict={X: np.array([[1.0, 1.0]])})

    print(pred[0][0])