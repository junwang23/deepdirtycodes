import os
import urllib.request
import zipfile
import collections
import random
import math
import datetime as dt

import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists('data/' + filename):
        with urllib.request.urlopen(url + filename) as r, open(
                'data/' + filename, 'wb') as f:
            data = r.read()
            f.write(data)
    statinfo = os.stat('data/' + filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename)
    return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data('data/' + filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 10000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# count is the # occurrence of the most frequent words, ordered by frequency
# dictionary  = {word0: identifier, ... } words are ordered as in count
# reverse_dictionary is the reverse of key and value in dictionary
# data is the original text in which each word is replaced by its identifier
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


def generate_batch(data, batch_size, num_skips, skip_window):
    """Generate a training batch for the skip-gram model
        data: full length of the text (identifiers)
        batch_size: # of (input, context) pair in a batch
        num_skips: # of context words picked
        skip_window: radius of the moving window
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # input word array
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # context words
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # moving window
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
data_index = 0

# Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit
# the validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


graph = tf.Graph()

with graph.as_default():

    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # validation set
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # input embedding matrix
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # look up embeddings for inputs
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # # Construct the variables for the softmax
    # weights = tf.Variable(
    #     tf.truncated_normal([embedding_size, vocabulary_size],
    #                         stddev=1.0 / math.sqrt(embedding_size)))
    # biases = tf.Variable(tf.zeros([vocabulary_size]))
    # # the fully connected hidden layer without activation
    # hidden_out = tf.matmul(embed, weights) + biases

    # # The model in summary is Y = softmax(X.Embd.W + b)
    # # Embd connects the input to the hidden layer, which is then connected to
    # # the contexts output by W (weights), with bias b

    # # convert train_context to a one-hot format
    # train_one_hot = tf.one_hot(train_labels, vocabulary_size)
    # # softmax cross entropy loss
    # loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,
    #                                             labels=train_one_hot))

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # NCE loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # use the SGD optimizer.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    norm_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(norm_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, norm_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()


def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_context = generate_batch(data,
                                                         batch_size,
                                                         num_skips,
                                                         skip_window)
            feed_dict = {train_inputs: batch_inputs,
                         train_labels: batch_context}

            # We perform one update step by evaluating the optimizer op
            # including it in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss],
                                      feed_dict=feed_dict)
            average_loss += loss_val

            # The average loss is an estimate over the last 2000 batches.
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = norm_embeddings.eval()

        tsne = TSNE(perplexity=30, n_components=2, init='pca',
                    n_iter=1000, method='exact', verbose=2)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)


# visualize the embeddings
def plot_with_labels(low_dim_embs, labels, filename='results/skipgram.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

num_steps = 50001
nce_start_time = dt.datetime.now()
run(graph, num_steps)
nce_end_time = dt.datetime.now()
time_span = (nce_end_time - nce_start_time).total_seconds()
print('NCE method took ' + str(time_span) + ' seconds to run' +
      str(num_steps) + 'iterations')
