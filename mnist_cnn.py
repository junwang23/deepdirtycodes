import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data", one_hot=True)

# optimisation parameters
learning_rate = 0.0001
epochs = 10
batch_size = 50

# training data placeholders
x = tf.placeholder(tf.float32, [None, 784])  # input data
x_shaped = tf.reshape(x, [-1, 28, 28, 1])  # reshape input
# output
y = tf.placeholder(tf.float32, [None, 10])


def create_new_conv_layer(input_data, num_input_channels, num_filters,
                          filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    # [filter_height, filter_width, in_channels, out_channels]
    conv_filt_shape = [filter_shape[0], filter_shape[1],
                       num_input_channels, num_filters]

    # initialise weights and bias
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                          name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_shape[0], pool_shape[1], 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer


# create convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

# fully connected layers
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
# weights and bias
w1 = tf.Variable(tf.truncated_normal(
    [7 * 7 * 64, 1000], stddev=0.03), name='w1')
b1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='b1')
dense_layer1 = tf.nn.relu(tf.matmul(flattened, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='w2')
b1 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='b2')
dense_layer2 = tf.matmul(dense_layer1, w2) + b1
y_ = tf.nn.softmax(dense_layer2)

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2, labels=y))

# optimiser
optimiser = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)

# accuracy assessment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialization
init_op = tf.global_variables_initializer()

# session
with tf.Session() as sess:
    # initial variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={
                            x: mnist.test._images, y: mnist.test._labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
              "test accuracy: {: .3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={
          x: mnist.test._images, y: mnist.test._labels}))
