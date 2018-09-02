import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 10
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length
num_layers = 3  # number of network layers


def generateData():
    """Generate training data.
        The input is a random binary vector
        The output is the shifted echo of the input
    """
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)


# input and output
X = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
Y = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
# states placeholders
# cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
# hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
# init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)

# declare the cell states and hidden states in one tensor
init_state = tf.placeholder(
    tf.float32, [num_layers, 2, batch_size, state_size])
# unstack init_state to a tuple of LSTMTuples
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],
                                   state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

# weights and biases
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(X, truncated_backprop_length, 1)
labels_series = tf.unstack(Y, axis=1)

# Forward pass
# cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
cell_stack = [tf.contrib.rnn.BasicLSTMCell(
    state_size, state_is_tuple=True) for i in range(num_layers)]
cells = tf.contrib.rnn.MultiRNNCell(cell_stack, state_is_tuple=True)
states_series, current_state = tf.contrib.rnn.static_rnn(
    cells, inputs_series, initial_state=rnn_tuple_state)

# softmax corss entropy loss
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
    for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

# optimizer
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


# Visualization function

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, (batch_size + 1)//2, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[
            :, batch_series_idx, :]
        single_output_series = np.array(
            [(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, (batch_size + 1)//2, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :],
                width=1, color="blue", edgecolor='black')
        plt.bar(left_offset, batchY[batch_series_idx, :]
                * 0.5, width=1, color="red", edgecolor='black')
        plt.bar(left_offset, single_output_series * 0.3,
                width=1, color="green", edgecolor='black')

    plt.draw()
    plt.pause(0.0001)


# training session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = generateData()
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series =
            sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    X: batchX,
                    Y: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
