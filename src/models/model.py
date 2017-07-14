import tensorflow as tf
import numpy as np
import _pickle as cPickle
import os
from os.path import join


class Model:
         
    def train_step(self, batchX, batchY):
        '''
        Runs one step of stochastic gradient based method - updates model weights internally
        INPUTS : batchX - (batch_size, IN_HEIGHT, IN_WIDTH, IN_CHANNELS) input RGB Images
                 batchy - (batch_size, num_joints, 2) ground truth joint coordinates for each image
        OUTPUTS : loss - objective function loss
                  rmse - root mean squared error for batch
        '''
        # print ('============ batchX', batchX)
        # print ('============ batchY', batchY)
        # print ('============ batchX.shape', batchX.shape) # (2, 224, 224, 3)
        # print ('============ batchY.shape', batchY.shape) # (2, 16, 2)
        output, rmse, _ = self.sess.run([self.loss, self.rmse, self.train_op], feed_dict={
            self.x: batchX,
            self.y: batchY
        })
        return output, rmse

    def train_op_init(self, total_loss, global_step):
        MOVING_AVERAGE_DECAY       = 0.9999           # The decay to use for the moving average.
        INITIAL_LEARNING_RATE      = self.opts.INIT_LR
        LEARNING_RATE_DECAY_FACTOR = self.opts.LR_DECAY_FACTOR
        DECAY_STEPS                = self.opts.DECAY_STEPS

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        DECAY_STEPS,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        return apply_gradient_op


    def predict(self, batchX):
        '''
        Prediction for given batch of inputs
        INPUT : batchX - (batch_size, IN_HEIGHT, IN_WIDTH, IN_CHANNELS) input RGB Images
        OUTPUT : pred - (batch_size, num_joints, 2) prediction for each corresponding input image
        '''

        filler = np.zeros(self.y.get_shape())
        pred = self.sess.run(self.logits, feed_dict={self.x:batchX, self.y:filler})
        # For lstm model where shape is (T,B,2)
        pred = np.array(pred)
        if pred.shape[0] != self.opts.BATCH_SIZE:
            pred = np.swapaxes(pred,0,1)
        return pred

   
    def save_weights(self, it):
        if not os.path.exists(self.opts.ARCHLOG_DIR):
            os.makedirs(self.opts.ARCHLOG_DIR)
        checkpoint_path = join(self.opts.ARCHLOG_DIR, 'checkpoint.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=it)


    def restore_weights(self):
        all_ckpts = tf.train.get_checkpoint_state(self.opts.ARCHLOG_DIR)
        self.saver.restore(self.sess, all_ckpts.model_checkpoint_path)
