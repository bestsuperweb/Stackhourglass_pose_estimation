from layer_utils import *
from model import Model
import numpy as np
import os
import tensorflow as tf


class BaseModel_small(Model):

    def __init__(self, opts):

        B = opts.BATCH_SIZE
        H = opts.IN_HEIGHT
        W = opts.IN_WIDTH
        C = opts.IN_CHANNELS
        T = opts.NUM_JOINTS

        self.opts = opts

        self.x = tf.placeholder(tf.float32, [B, H, W, C], name="x")
        self.y = tf.placeholder(tf.float32, [B, T, 2], name="y")
        self.global_step = tf.Variable(0,trainable=False)

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
        p4 = maxpool2d(c4_2, 2, "p4")


        reshape = tf.reshape(p4, [B, -1])
        dim = reshape.get_shape()[1]
        fc5 = dense_relu_batch(reshape, dim, 256, [0], 'fc5')
        fc6 = dense_relu_batch(fc5, 256,256, [0], 'fc6')
        self.logits = tf.reshape(dense(fc6, 256, T*2, 'logits'), [B, T, 2])
        
        self.diff = (self.logits - self.y) * tf.to_float(self.y > 0)
        self.loss = tf.nn.l2_loss(self.diff)
        self.rmse = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.diff), 2)), 1))
        self.train_op = self.train_op_init(self.loss, self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())



if __name__ == '__main__':
    # Basic functionality test
    import sys
    sys.path.append("..")
    from GlobalOpts import GlobalOpts
    opts = GlobalOpts('basemodel')

    model = BaseModel(opts)
    for i in range(1):
        batchX = np.random.rand(opts.BATCH_SIZE, opts.IN_HEIGHT, opts.IN_WIDTH, 3)
        batchy = np.random.rand(opts.BATCH_SIZE, opts.NUM_JOINTS, 2)
        loss = model.train_step(batchX, batchy)
        print loss
        pred = model.predict(batchX)
        print np.array(pred).shape
