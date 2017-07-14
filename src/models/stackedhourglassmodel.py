from models.layer_utils import *
from models.model import Model
# from layer_utils import *
# from model import Model
from os.path import join

import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class stacked_hourglass_old():
    def __init__(self, nb_stack, name='stacked_hourglass'):
        self.nb_stack = nb_stack
        self.name = name

    def __call__(self, x):
        # print ('=============== x.get_shape()', x.get_shape())
        with tf.name_scope(self.name) as scope:
            padding = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], name='padding')
            with tf.name_scope("preprocessing") as sc:
                conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
                norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                r1 = self._residual_block(norm1, 128, 'r1')
                pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], 'VALID', scope=scope)
                r2 = self._residual_block(pool, 128, 'r2')
                r3 = self._residual_block(r2, 256, 'r3')
            hg = [None] * self.nb_stack
            ll = [None] * self.nb_stack
            ll_ = [None] * self.nb_stack
            out = [None] * self.nb_stack
            out_ = [None] * self.nb_stack
            sum_ = [None] * self.nb_stack
            with tf.name_scope('_hourglass_0_with_supervision') as sc:
                hg[0] = self._hourglass(r3, 4, 256, '_hourglass')
                ll[0] = self._conv_bn_relu(hg[0], 256, name='conv_1')
                ll_[0] = self._conv(ll[0],256,1,1,'VALID','ll')
                out[0] = self._conv(ll[0],16,1,1,'VALID','out')
                out_[0] = self._conv(out[0],256,1,1,'VALID','out_')
                sum_[0] = tf.add_n([ll_[0], out_[0], r3])
            for i in range(1, self.nb_stack - 1):
                with tf.name_scope('_hourglass_' + str(i) + '_with_supervision') as sc:
                    hg[i] = self._hourglass(sum_[i-1], 4, 256, '_hourglass')
                    ll[i] = self._conv_bn_relu(hg[i], 256, name='conv_1')
                    ll_[i] = self._conv(ll[i],256,1,1,'VALID','ll')
                    out[i] = self._conv(ll[i],16,1,1,'VALID','out')
                    out_[i] = self._conv(out[i],256,1,1,'VALID','out_')
                    sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i-1]])
            with tf.name_scope('_hourglass_' + str(self.nb_stack - 1) + '_with_supervision') as sc:
                hg[self.nb_stack-1] = self._hourglass(sum_[self.nb_stack - 2], 4, 256, '_hourglass')
                ll[self.nb_stack-1] = self._conv_bn_relu(hg[self.nb_stack - 1], 256, name='conv_1')
                out[self.nb_stack-1] = self._conv(ll[self.nb_stack-1],16,1,1,'VALID','out')
            return tf.stack(out)

    def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID', name='conv'):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                    kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            return conv

    def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1, name=None):
         with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                    kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='SAME', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, scope=scope)
            return norm

    def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
        with tf.name_scope(name) as scope:
            with tf.name_scope('norm_conv1') as sc:
                norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv1 = self._conv(norm1, int(nb_filter_out / 2), 1, 1, 'SAME', name='conv1')
            with tf.name_scope('norm_conv2') as sc:
                norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv2 = self._conv(norm2, int(nb_filter_out / 2), 3, 1, 'SAME', name='conv2')
            with tf.name_scope('norm_conv3') as sc:
                norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME', name='conv3')
            return conv3

    def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
        if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
            return inputs
        else:
            with tf.name_scope(name) as scope:
                conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME', name='conv')
                return conv

    def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
        with tf.name_scope(name) as scope:
            _conv_block = self._conv_block(inputs, nb_filter_out)
            _skip_layer = self._skip_layer(inputs, nb_filter_out)
            return tf.add(_skip_layer, _conv_block)

    def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass'):
        # print ('========== n =========', n)
        with tf.name_scope(name) as scope:
            # Upper branch
            # print ('============== inputs.shape', inputs.shape)
            up1 = self._residual_block(inputs, nb_filter_res, 'up1')
            # Lower branch
            pool = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], 'VALID', scope=scope)
            low1 = self._residual_block(pool, nb_filter_res, 'low1')
            if n > 1:
                low2 = self._hourglass(low1, n-1, nb_filter_res, 'low2')
            else:
                low2 = self._residual_block(low1, nb_filter_res, 'low2')
            low3 = self._residual_block(low2, nb_filter_res, 'low3')
            low4 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2,
                                                    name='upsampling')
            if n < 4:
                # print ('************* up1', up1.get_shape())
                # print ('************* low4', low4.get_shape())
                # print ("tf.add(up1, low4, name='merge')", tf.add(up1, low4, name='merge').get_shape())
                return tf.add(up1, low4, name='merge')
            else:
                return self._residual_block(tf.add(up1, low4), nb_filter_res, 'low4')

class stacked_hourglass(Model):
    def __init__(self, opts):

        B = opts.BATCH_SIZE
        H = opts.IN_HEIGHT
        W = opts.IN_WIDTH
        C = opts.IN_CHANNELS
        T = opts.NUM_JOINTS

        # self.nb_stack = 8
        self.nb_stack = 3
        self.opts = opts
        self.name = 'stacked_hourglass'
        self.x = tf.placeholder(tf.float32, [B, H, W, C], name="x")
        # print ('=============== self.x.get_shape()', self.x.get_shape()) # (2, 224, 224, 3)
        self.y = tf.placeholder(tf.float32, [B, T, 2], name="y")
        self.global_step = tf.Variable(0,trainable=False)

        with tf.name_scope(self.name) as scope:
            padding = tf.pad(self.x, [[0,0],[3,3],[3,3],[0,0]], name='padding')
            with tf.name_scope("preprocessing") as sc:
                conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
                norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                r1 = self._residual_block(norm1, 128, 'r1')
                pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], 'VALID', scope=scope)
                r2 = self._residual_block(pool, 128, 'r2')
                r3 = self._residual_block(r2, 256, 'r3')
            hg = [None] * self.nb_stack
            ll = [None] * self.nb_stack
            ll_ = [None] * self.nb_stack
            out = [None] * self.nb_stack
            out_ = [None] * self.nb_stack
            sum_ = [None] * self.nb_stack
            with tf.name_scope('_hourglass_0_with_supervision') as sc:
                hg[0] = self._hourglass(r3, 4, 256, '_hourglass')
                ll[0] = self._conv_bn_relu(hg[0], 256, name='conv_1')
                ll_[0] = self._conv(ll[0],256,1,1,'VALID','ll')
                out[0] = self._conv(ll[0],16,1,1,'VALID','out')
                out_[0] = self._conv(out[0],256,1,1,'VALID','out_')
                sum_[0] = tf.add_n([ll_[0], out_[0], r3])
            for i in range(1, self.nb_stack - 1):
                with tf.name_scope('_hourglass_' + str(i) + '_with_supervision') as sc:
                    hg[i] = self._hourglass(sum_[i-1], 4, 256, '_hourglass')
                    ll[i] = self._conv_bn_relu(hg[i], 256, name='conv_1')
                    ll_[i] = self._conv(ll[i],256,1,1,'VALID','ll')
                    out[i] = self._conv(ll[i],16,1,1,'VALID','out')
                    out_[i] = self._conv(out[i],256,1,1,'VALID','out_')
                    sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i-1]])
            with tf.name_scope('_hourglass_' + str(self.nb_stack - 1) + '_with_supervision') as sc:
                hg[self.nb_stack-1] = self._hourglass(sum_[self.nb_stack - 2], 4, 256, '_hourglass')
                ll[self.nb_stack-1] = self._conv_bn_relu(hg[self.nb_stack - 1], 256, name='conv_1')
                out[self.nb_stack-1] = self._conv(ll[self.nb_stack-1],16,1,1,'VALID','out')
            # self.logits = tf.stack(out)
            
            # p4 = maxpool2d(tf.stack(out), 2, "p5")
            # print ('=============== tf.stack(out).get_shape()', tf.stack(out).get_shape()) # (3, 2, 64, 64, 16)
            reshape = tf.reshape(tf.stack(out), [B, -1])
            # print ('=============== reshape.get_shape()', reshape.get_shape()) # (2, 150528)
            dim = reshape.get_shape()[1]
            fc5 = dense_relu_batch(reshape, dim, 256, [0], 'fc5')
            fc6 = dense_relu_batch(fc5, 256,256, [0], 'fc6')
            # return
            self.logits = tf.reshape(dense(fc6, 256, T*2, 'logits'), [B, T, 2])
            # print ('=============== self.logits.get_shape()', self.logits.get_shape()) # (2, 16, 2)

            self.diff = (self.logits - self.y) * tf.to_float(self.y > 0)
            self.loss = tf.nn.l2_loss(self.diff)
            self.rmse = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.diff), 2)), 1))
            self.train_op = self.train_op_init(self.loss, self.global_step)

            # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

            self.saver = tf.train.Saver(tf.global_variables())
            # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID', name='conv'):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                    kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            return conv

    def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1, name=None):
         with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,\
                                    kernel_size,inputs.get_shape().as_list()[3],nb_filter]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='SAME', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, scope=scope)
            return norm

    def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
        with tf.name_scope(name) as scope:
            with tf.name_scope('norm_conv1') as sc:
                norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv1 = self._conv(norm1, int(nb_filter_out / 2), 1, 1, 'SAME', name='conv1')
            with tf.name_scope('norm_conv2') as sc:
                norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv2 = self._conv(norm2, int(nb_filter_out / 2), 3, 1, 'SAME', name='conv2')
            with tf.name_scope('norm_conv3') as sc:
                norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5, 
                                    activation_fn=tf.nn.relu, scope=sc)
                conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME', name='conv3')
            return conv3

    def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
        if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
            return inputs
        else:
            with tf.name_scope(name) as scope:
                conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME', name='conv')
                return conv

    def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
        with tf.name_scope(name) as scope:
            _conv_block = self._conv_block(inputs, nb_filter_out)
            _skip_layer = self._skip_layer(inputs, nb_filter_out)
            return tf.add(_skip_layer, _conv_block)

    def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass'):
        # print ('========== n =========', n)
        with tf.name_scope(name) as scope:
            # Upper branch
            # print ('============== inputs.shape', inputs.shape)
            up1 = self._residual_block(inputs, nb_filter_res, 'up1')
            # Lower branch
            pool = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], 'VALID', scope=scope)
            low1 = self._residual_block(pool, nb_filter_res, 'low1')
            if n > 1:
                low2 = self._hourglass(low1, n - 1, nb_filter_res, 'low2')
            else:
                low2 = self._residual_block(low1, nb_filter_res, 'low2')
            low3 = self._residual_block(low2, nb_filter_res, 'low3')
            # print ('=========== low3', low3.get_shape())
            low4 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2,
                                                    name='upsampling')
            # print ('========== tf.shape(low3)[1:3] * 2 ==========', tf.shape(low3)[1:3] * 2)
            if n < 4:
                # print ('************* up1', up1.get_shape())
                # print ('************* low4', low4.get_shape())
                # print ("tf.add(up1, low4, name='merge')", tf.add(up1, low4, name='merge').get_shape())
                return tf.add(up1, low4, name='merge')
            else:
                return self._residual_block(tf.add(up1, low4), nb_filter_res, 'low4')

if __name__ == "__main__":

    # # Basic functionality test
    # import sys
    # sys.path.append("..")
    # from GlobalOpts import GlobalOpts
    # opts = GlobalOpts('basemodel')

    # model = stacked_hourglass(opts)
    # # print ('model', model)
    # for i in range(1):
    #     batchX = np.random.rand(opts.BATCH_SIZE, opts.IN_HEIGHT, opts.IN_WIDTH, 3)
    #     batchy = np.random.rand(opts.BATCH_SIZE, opts.NUM_JOINTS, 2)
    #     print ('batchX', batchX)
    #     loss = model.train_step(batchX, batchy)
    #     print ("loss => ", loss)
    #     pred = model.predict(batchX)
    #     print ("pred => ", pred)

    # Basic functionality test
    import sys
    sys.path.append("..")
    from GlobalOpts import GlobalOpts
    from reader import PoseReader
    opts = GlobalOpts('hourglass')
    valreader = PoseReader('valid', opts)
    filepath = join(str.encode(opts.DATA_DIR + '/image.png'))
    print ('filePath', filepath)
    print ("Loading Model ...")
    model = stacked_hourglass(opts)
    print ("Model Loaded Successfully!")
    batchX = valreader.get_image(filepath)
    print ('batchX', batchX)
    print ('batchX.shape', batchX.shape)
    
    pred = model.predict(batchX)
    print ("pred => ", pred)

    # # sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
    # with tf.Graph().as_default():
    #     DEVICE = '/gpu:0'
    #     with tf.device(DEVICE):
    #         print ("start build model...")
    #         _x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    #         y = tf.placeholder(tf.float32, [2, None, 64, 64, 16])
    #         output = stacked_hourglass_old(2, 'stacked_hourglass')(_x)
    #         loss = tf.reduce_mean(tf.square(output - y))
    #         print ('============ loss =============', loss)
    #         rmsprop = tf.train.RMSPropOptimizer(2.5e-4)
    #         print ("build finished...")
    #     train_step = tf.Variable(0, name='global_step', trainable=False)
    #     with tf.device(DEVICE):
    #         train_rmsprop = rmsprop.minimize(loss, train_step)
    #     init = tf.global_variables_initializer()
    #     with tf.Session() as sess:
    #         with tf.device(DEVICE):
    #             sess.run(init)
    #         print ("test...")
    #         xarr = np.random.rand(10, 24, 256, 256, 3)
    #         yarr = np.random.rand(10, 2, 24, 64, 64, 16)
    #         _time = time.clock()
    #         with tf.device(DEVICE):
    #             for u in range(0, 1):
    #                 sess.run(train_rmsprop, feed_dict={_x:xarr[u], y:yarr[u]})
    #         print ("test:", time.clock() - _time)