# deepdirtycodes

Experimental and exercising codes for deep learning with TensorFlow.

## MNIST on CNN

Tutorial codes for learning the basics of nueral networks and training a simple CNN for image classification.

The code is from the tutorial blogs of [Python TensorFlow Tutorial – Build a Neural Network](http://adventuresinmachinelearning.com/python-tensorflow-tutorial/) and of [Convolutional Neural Networks Tutorial in TensorFlow](http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/).

The official tutorial for building CNN with TensorFlow is available [here](https://www.tensorflow.org/tutorials/layers).

Related scripts

-   `mnist_simple.py` - A simple feed-forward neural network for MNIST image classification (%96-97 accuracy).
-   `mnist_cnn.py` - A CNN model with two convolution layers (about 99% accuracy).

## Word2Vec

Tutorial codes for training word2vec embeddings with the Skip-Gram model.

Run the script - `python word2vec.py`

The code is from the [Tensorflow Word2Vec Tutorial](https://www.tensorflow.org/tutorials/word2vec) ([Github repo](https://github.com/tensorflow/tensorflow/tree/r1.3/tensorflow/examples/tutorials/word2vec)), with minor adaptions.

A very detailed explanation of the code, also a more noob-friendly tutorial of word2vec word embedding, can be found [here](http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/).

## RNN and LSTM

A review of RNN models can be found in this [arXiv paper](https://arxiv.org/pdf/1506.00019.pdf). An introduction of LSTM networks can be found in this [blog artical](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), which is referenced in many tutorials.

A serie of detailed _technical_ tutorials of RNN and LSTM is found [here](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767), based on which I experimented the networks. The older tensorflow api used in those articals have been changed in my scripts.

Related scripts

-   `rnn_basic.py` - A simple recurrent neural network learning a time-series of numbers echoing with fixed time delay.
-   `rnn_lstm.py` - The same echo time-serie learned with single/multi-layer lstm network, based on tensorflow api.
