import tensorflow as tf


class Conv1D(tf.layers.Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.truncated_normal_initializer(
                     mean=0.0, stddev=0.03, seed=None),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Conv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        if padding:
            inputs = tf.pad(inputs, tf.constant(
                [(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(Conv1D, self).call(inputs)


class DCNBlock(tf.layers.Layer):
    def __init__(self, n_filters,
                 kernel_size,
                 dilation_rate,
                 strides=1,
                 trainable=True,
                 name=None,
                 dtype=None,
                 first_layer=False,
                 **kwargs):
        super(DCNBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            name=name, **kwargs
        )
        self.conv = Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            activation=tf.nn.relu,
            name='dilated_conv')
        self.first_layer = first_layer

    def call(self, inputs, training=True):
        if self.first_layer:
            self.skip = Conv1D(1, 1, name='skip')
            skip_out = self.skip(inputs)
        else:
            skip_out = inputs
        layer_out = self.conv(inputs)
        network_out = tf.add(skip_out, layer_out)
        if self.first_layer:
            network_out = tf.reduce_sum(network_out, axis=0, keepdims=True)
        return network_out


class TemporalConvNet(tf.layers.Layer):
    def __init__(self, n_layers,
                 n_filters,
                 kernel_size=2,
                 trainable=True,
                 name=None,
                 dtype=None,
                 activity_regularizer=None,
                 **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                DCNBlock(n_filters=n_filters,
                         kernel_size=kernel_size,
                         strides=1,
                         dilation_rate=2 ** i,
                         first_layer=(i == 0),
                         name='layer_{}'.format(i))
            )
        self.dense = Conv1D(1, 1, name='forecast')

    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        network_out = self.dense(outputs)
        return network_out
