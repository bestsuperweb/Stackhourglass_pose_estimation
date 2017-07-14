import tensorflow as tf
import numpy as np
# from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops import array_ops
import itertools



######## LAYERS ########
def dense(input_data, N, H, name):
    """NN fully connected layer."""
    with tf.variable_scope(name):  
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.xavier_initializer())   
        b = tf.get_variable("b", [H], initializer=tf.constant_initializer(0))
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes, name):
    """NN batch normalization layer."""
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

def dense_relu_batch(input_data, N, H, axes, name):
    """NN dense relu batch layer."""
    with tf.variable_scope(name):
        affine = dense(input_data, N, H, "dense")
        bn = batch_normalization(affine, axes, "batch")
        return tf.nn.relu(bn, "relu")

def dense_relu(input_data, N, H, name):
    """NN dense relu layer"""
    with tf.variable_scope(name):
        affine = dense(input_data, N, H, "dense")
        return tf.nn.relu(affine, "relu")

def multi_dense_relu_batch(input_data, N, Hs, axes, name):
    """NN multi dense relu batch layer."""
    with tf.variable_scope(name):
        output = input_data
        for i, H in enumerate(itertools.izip([N] + Hs, Hs)):
            output = dense_relu_batch(output, H[0], H[1], axes, "fc_" + str(i))
        return output

def conv2d(input_data, filter_size, stride, name):
    """NN 2D convolutional layer."""
    with tf.variable_scope(name):
        W = tf.get_variable("W", filter_size, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME", name="conv2d")
        biases = tf.get_variable("b", shape=filter_size[-1])
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        return bias

def maxpool2d(input_data, stride, name):
    """NN 2D max pooling layer."""
    with tf.variable_scope(name):
        filter_size = [1, stride, stride, 1]
        return tf.nn.max_pool(input_data, filter_size, filter_size, "SAME", name="max_pool")

def conv2d_relu_batch(input_data, filter_size, stride, axes, name):
    with tf.variable_scope(name):
        conv = conv2d(input_data, filter_size, stride, "conv2d")
        bn = batch_normalization(conv, axes, "batch")
        return tf.nn.relu(bn, "relu")

def conv2d_relu(input_data, filter_size, stride, name):
    with tf.variable_scope(name):
        conv = conv2d(input_data, filter_size, stride, "conv2d")
        return tf.nn.relu(conv, "relu")

### LSTM LAYERS

def lstm(state, input_data, num_steps, hidden_size, num_layers, name):
    # input_data : (B, H*W*D)
    B, dim = input_data.get_shape()
    with tf.variable_scope(name) as scope:
        multi_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size)] * num_layers)
        #state = multi_lstm.zero_state(B, tf.float32)
        outputs = []
        for t in range(num_steps):
            output, state = multi_lstm(input_data, state)
            output = batch_normalization(output, [0, 1], "batch")
            outputs.append(output)
            scope.reuse_variables()
    return outputs


def lstm_attention(state, annots, T, H, Hs, num_layers, name):
    """NN lstm attention model.
    state : (B, 2 * D) - concatenation of cell and hidden states
    annots : (B, N, D) - annotation vectors
    T: number of timesteps - in this case - number of joints
    H: hidden state size (of one lstm, i.e. H * num_layers = D)
    Hs: hidden state sizes of attention perceptron (a list: one per layer)
    num_layers : number of lstm layers

    Reference variables
    B: size of batch
    N: number of annotations for a given image
    D: dimension of each annotation vector
    """
    B, N, D = tf.shape(annots)

    with tf.variable_scope(name) as scope:
        multi_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(H)] * num_layers)
        outputs = []
        for t in range(T):
            c, h = array_ops.split(1, 2, state) 

            # CORRECTION TODO - should use perceptron over all annotation vectors - not just one

            perceptron_input = tf.concat(1, [annots[:, t, :], h])
            perceptron = multi_dense_relu_batch(perceptron_input, D * 2, Hs, [0], "fc")
            weights = dense(perceptron, Hs[-1], N, "weights")
            weights = tf.nn.softmax(weights, "softmax")
            weights = tf.expand_dims(weights, -1, "expand_dim_weights")
            weights = tf.tile(weights, [1, 1, D], "tiled_weights")
            weighted_annots = tf.reduce_sum(weights * annots, 1) 
            output, state = multi_lstm(weighted_annots, state)
            outputs.append(output)
            scope.reuse_variables()
        return outputs
