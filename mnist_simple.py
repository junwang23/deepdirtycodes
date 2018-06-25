import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# optimisation parameters
learning_rate = 0.5
epochs = 10
batch_size = 100

# training data placeholders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# weights connecting the hidden layer to the ouput layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W1')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# the hidden layer with a relu activation
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
# output layer with a softmax activation
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# cross entropy loss function
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(
    tf.reduce_mean(
        y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)

# initializer
init_op = tf.global_variables_initializer()

# define accuracy assessment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session
with tf.Session() as sess:
    # intializae variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={
          x: mnist.test.images, y: mnist.test.labels}))
