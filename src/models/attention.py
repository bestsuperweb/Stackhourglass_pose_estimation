import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops import array_ops
from layer_utils import *
import itertools


######## SAMPLERS ########
def epoch_sampler(x, y, batch_size): 
    """Does one pass over the data."""
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    M = int(len(y) / batch_size) * batch_size
    x_splits = np.array_split(x[:M], M / batch_size)
    y_splits = np.array_split(y[:M], M / batch_size)
    for batch_x, batch_y in zip(x_splits, y_splits):
        yield batch_x, batch_y
    yield x[M:], y[M:]
    
def infinite_epoch_sampler(x, y, batch_size):
    """Does infinite passes over the data."""
    while True:
        for batch_x, batch_y in epoch_sampler(x, y, batch_size):
            yield batch_x, batch_y


######## MODELS ########
class RegressionModel:
    def __init__(self, N, Hs, axes, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, N], name="x")
            self.y = tf.placeholder(tf.float32, [None], name="y")
            self.keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")
            self.lr = tf.placeholder(tf.float32, [], name="lr")

            out = multi_dense_relu_batch(self.x, N, Hs, axes, "fc")
            out = tf.nn.dropout(out, self.keep_prob, name="dropout")

            self.logits = tf.squeeze(dense(out, Hs[-1], 1, "logits"))
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.logits - self.y))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

class LogisticModel:
    def __init__(self, N, Hs, axes, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, N], name="x")
            self.y = tf.placeholder(tf.float32, [None], name="y")
            self.keep_prob = tf.placeholder(tf.float32, [], name="keep_prob")
            self.lr = tf.placeholder(tf.float32, [], name="lr")

            out = multi_dense_relu_batch(self.x, N, Hs, axes, "fc")
            out = tf.nn.dropout(out, self.keep_prob, name="dropout")

            self.logits = tf.squeeze(dense(out, Hs[-1], 1, "logits"))
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

class AttnModel(Model):
    def __init__(self, H, W, C, T, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None, H, W, C], name="x")
            self.y = tf.placeholder(tf.float32, [None, T, 2], name="y")

            c0_0 = conv2d_relu_batch(self.x, [3, 3, C, 64], 1, [0, 1, 2], "c0_0")
            c0_1 = conv2d_relu_batch(c0_0, [3, 3, 64, 64], 1, [0, 1, 2], "c0_1")
            p0 = maxpool2d(c0_1, 2, "p0")

            c1_0 = conv2d_relu_batch(p0, [3, 3, 64, 128], 1, [0, 1, 2], "c1_0")
            c1_1 = conv2d_relu_batch(c1_0, [3, 3, 128, 128], 1, [0, 1, 2], "c1_1")
            p1 = maxpool2d(c1_1, 2, "p1")

            c2_0 = conv2d_relu_batch(p1, [3, 3, 128, 256], 1, [0, 1, 2], "c2_0")
            c2_1 = conv2d_relu_batch(c2_0, [3, 3, 256, 256], 1, [0, 1, 2], "c2_1")
            c2_2 = conv2d_relu_batch(c2_1, [3, 3, 256, 256], 1, [0, 1, 2], "c2_2")
            p2 = maxpool2d(c2_2, 2, "p2")

            c3_0 = conv2d_relu_batch(p2, [3, 3, 256, 512], 1, [0, 1, 2], "c3_0")
            c3_1 = conv2d_relu_batch(c3_0, [3, 3, 512, 512], 1, [0, 1, 2], "c3_1")
            c3_2 = conv2d_relu_batch(c3_1, [3, 3, 512, 512], 1, [0, 1, 2], "c3_2")
            p3 = maxpool2d(c3_2, 2, "p3")

            c4_0 = conv2d_relu_batch(p3, [3, 3, 512, 512], 1, [0, 1, 2], "c4_0")
            c4_1 = conv2d_relu_batch(c4_0, [3, 3, 512, 512], 1, [0, 1, 2], "c4_1")
            c4_2 = conv2d_relu_batch(c4_1, [3, 3, 512, 512], 1, [0, 1, 2], "c4_2")

            annots_shape = tf.pack([tf.shape(c4_2)[0], -1, 512])
            annots = tf.reshape(c4_2, annots_shape)
            lstm_c = lstm_h = tf.reduce_mean(annots, 1)
            state = tf.concat(1, [lstm_c, lstm_h])

            #attention = lstm_attention(state, annots, 196, 512, T, 512, [128, 128], 1, "attention")
            attention = lstm_attention(state, annots, T, 512, [128, 128], 1, "attention")
            self.logits = [dense(v, 512, 2, str(k) + "dense") for k, v in enumerate(attention)]
            truth = [tf.squeeze(t) for t in tf.split(1, T, self.y)] 
            self.loss = tf.add_n([tf.nn.l2_loss(t - l) for t, l in zip(truth, self.logits)])
            self.rmse = tf.add_n([tf.sqrt(tf.reduce_sum(tf.squared_difference(t, l))) for t,l in zip(truth, self.logits)]) / 16
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

            self.saver = tf.train.Saver(tf.all_variables())
            self.sess = tf.Session()
            self.sess.run(tf.initialize_all_variables())



if __name__ == '__main__':
    with tf.Session() as sess:
        T = 16
        B = 10
        x = np.random.randn(B, 224, 224, 3)
        y = np.random.randn(B, T, 2)
        model = PoserModel(224, 224, 3, T, "model")

        sess.run(tf.initialize_all_variables())
        _, loss = sess.run([model.train_op, model.loss], feed_dict={
            model.x: x,
            model.y: y,
            model.lr: 1e-4
        })

        print loss

