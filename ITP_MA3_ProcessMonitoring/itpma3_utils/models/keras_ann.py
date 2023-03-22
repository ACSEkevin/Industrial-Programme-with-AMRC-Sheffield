"""
ITP Group MA3
multi-layer perceptron, developed by keras/TensorFlow
"""

from keras.layers import *
from keras.models import Sequential
import numpy as np


class FullyConnectedBlock(Layer):
    def __init__(self, units, batch_norm=True, activation='relu'):
        super(FullyConnectedBlock, self).__init__()
        self.use_bias = batch_norm
        self.multi_layer_perceptron = Dense(units, use_bias=self.use_bias,
                                            kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self.batch_norm = BatchNormalization(momentum=.99, epsilon=1e-5) \
            if batch_norm is True else Activation('linear')
        self.non_linearity = Activation(activation)

    def call(self, inputs, *args, **kwargs):
        return self.non_linearity(self.batch_norm(self.multi_layer_perceptron(inputs)))


class FullyConnectedNetwork(Layer):
    def __init__(self, layers: int or tuple, batch_norm=False, activation='relu', drop_rate=0.):
        super(FullyConnectedNetwork, self).__init__()
        self.iterable = hasattr(layers, "__iter__")
        if self.iterable:
            self.block_stack = [FullyConnectedBlock(
                units=layers[0],
                batch_norm=batch_norm,
                activation=activation)]

            self.block_stack += [FullyConnectedBlock(
                units=layers[index],
                batch_norm=batch_norm,
                activation=activation) for index in range(1, len(layers))]
        else:
            self.block_stack = [FullyConnectedBlock(
                units=layers,
                batch_norm=batch_norm,
                activation=activation)]

        self.drop_rate = drop_rate
        self.dropout_scheduler = np.linspace(0, drop_rate, len(self.block_stack))

    def call(self, x, *args, **kwargs):
        for index, block in enumerate(self.block_stack):
            x = block(x)
            x = Dropout(rate=self.dropout_scheduler[index])(x) if self.drop_rate > 0. else x

        return x


def MultiLayerPerceptron(input_shape,
                         num_classes,
                         layers=(32, 20),
                         batch_norm=False,
                         activation='relu',
                         drop_rate=0.05):
    """
    # structural programming of fully connected network
    :param activation:
    :param batch_norm:
    :param layers:
    :param input_shape:
    :param num_classes:
    :param drop_rate:
    :return:
    """
    # placeholder is needed
    # inputs = Input(input_shape)

    # fc layer
    fc_block = [FullyConnectedNetwork(layers=layers,
                                      batch_norm=batch_norm,
                                      activation=activation,
                                      drop_rate=drop_rate),
                Dense(1 if num_classes <= 2 else num_classes,
                      activation='sigmoid' if num_classes <= 2 else 'softmax')
                ]

    assert num_classes in [1, 2], "Binary classification in this project, can be modified id needed"
    return Sequential(fc_block, name='multi_layer_perceptron')
