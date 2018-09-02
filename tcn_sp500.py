import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tcn

import matplotlib.pylab as plt


def plot_data(label, prediction, length):
    df = pd.DataFrame({
        'label': label,
        'prediction': prediction,
        'time': list(range(length))
    })
    plt.style.use('seaborn-darkgrid')
    plt.plot('time', 'label', data=df, color='blue', alpha=0.8, linewidth=1)
    plt.plot('time', 'prediction', data=df,
             color='red', alpha=0.8, linewidth=1)
    plt.legend()
    plt.show()


path = os.path.dirname(sys.argv[0])

# Training Parameters
learning_rate = 0.00075
training_steps = 5000

# Network Parameters
kernel_size = 2
n_layers = 6
n_filters = 32

df = pd.read_csv(path + '/data/SP_500.csv')
data_size = df.shape[0]
values = df['Price'].values
x = values[:-1].reshape(1, data_size - 1, 1)
y = values[1:].reshape(1, data_size - 1, 1)

with tf.Graph().as_default() as g:
    # placeholders (batch_size/n_series, length, channel)
    x_ = tf.placeholder(dtype=tf.float32,
                        shape=[1, data_size - 1, 1],
                        name='input')

    # 1d dilated convolution network
    model = tcn.TemporalConvNet(n_layers=n_layers,
                                n_filters=n_filters,
                                kernel_size=kernel_size,
                                name='temporal_conv_net')
    y_ = model(x_)

    # Loss function using mean absolute difference
    with tf.name_scope('mae_loss'):
        loss = tf.losses.absolute_difference(
            y, y_, reduction=tf.losses.Reduction.MEAN)
        summary_loss = tf.summary.scalar('mae_loss', loss)

    # optimiser
    optimiser = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # initialization
    init = tf.global_variables_initializer()

    # tensorboard summary ops
    train_summary_op = tf.summary.merge([summary_loss])


with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(
        '{}/logs/tcn/{}l_{}k_{}f'.format(path,
                                         n_layers,
                                         kernel_size, n_filters),
        sess.graph)
    # Run the initializer
    sess.run(init)

    for i in range(training_steps):
        _, mae, summary = sess.run(
            [optimiser, loss, train_summary_op], feed_dict={x_: x})
        if i % 500 == 0:
            print('training step', i, 'mae:', mae)

        file_writer.add_summary(summary, i)

    print('Final mae:', mae)

    prediction = sess.run(y_, feed_dict={x_: x})
    plot_data(y[0, :, 0], prediction[0, :, 0], data_size - 1)
