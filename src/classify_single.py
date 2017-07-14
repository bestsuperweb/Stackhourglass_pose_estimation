import tensorflow as tf
import numpy as np
import argparse
import pickle

from GlobalOpts import GlobalOpts
from reader import PoseReader
from os.path import join

def main(opts, runtype):
    network = gen_model(opts)

    if opts.resume:
        network.restore_weights()

    if runtype == 'train':
        trainreader = PoseReader('train', opts)
        valreader   = PoseReader('valid', opts)
        train_hist = []
        valid_hist = []
        
        for it in range(opts.MAX_ITERS):
            batchX, batchy = trainreader.sample()
            loss, rmse = network.train_step(batchX, batchy)

            # Debugging input
            # diff = network.sess.run(network.loss, feed_dict={network.x:batchX, network.y:batchy})
            # print diff

            train_hist.append((it, loss, rmse))
            if it % opts.SUMM_CHECK == 0:
                print ('\r Iteration %d Loss : %f RMSE : %f' % (it, loss, rmse))

            if it != 0 and it % opts.VAL_CHECK == 0:
                print ('Running Evaluation over datasets')
                train_rmse = evaluate(network, trainreader, opts.BATCH_SIZE)
                val_rmse = evaluate(network, valreader, opts.BATCH_SIZE)
                print ('RMSE - Train : %f Validation : %f' % (train_rmse, val_rmse))
                valid_hist.append((it, train_rmse, val_rmse))

            if (it != 0) and ((it % opts.CHECKPOINT == 0) or (it + 1) == opts.MAX_ITERS):
                network.save_weights(it)
        
        # Save output files
        with open(join(opts.ARCHLOG_DIR, 'train_data.pkl'), 'wb') as fid:
            pickle.dump({'train_hist':train_hist, 'valid_hist':valid_hist}, fid)
            
    elif runtype == 'test':
        # trainreader = PoseReader('train', opts)
        valreader   = PoseReader('valid', opts)
        # testreader  = PoseReader('test', opts)
        # readers = {'train':trainreader, 'val':valreader, 'test':testreader}
        readers = {'val':valreader}
        print ('restoring weights of Neural Network ...')
        network.restore_weights()
        print ('Finished restoring')

        for reader_type in readers:
            pred, gt = batch_predict(network, readers[reader_type], opts.BATCH_SIZE)
            # print ('pred', pred)
            # print ('gt', 'gt')
            rmse = np.sqrt(np.mean(np.mean((pred - gt)**2,2)))
            print ('Set : %s RMSE : %f' % (reader_type, rmse))

def gen_model(opts):
    if opts.model == 'base':
        from models.basemodel import BaseModel
        return BaseModel(opts)
    elif opts.model == 'base_pt':
        from models.basemodel_pt import BaseModel_pt
        return BaseModel_pt(opts)
    elif opts.model == 'base_small':
        from models.basemodel_small import BaseModel_small
        return BaseModel_small(opts)
    elif opts.model == 'lstm':
        from models.LSTMModel import LSTMModel
        return LSTMModel(opts)
    elif opts.model == 'lstm_pt':
        from models.LSTMModel_pt import LSTMModel_pt
        return LSTMModel_pt(opts)
    elif opts.model == 'lstm_interm':
        from models.LSTMModel_interm import LSTMModel_interm
        return LSTMModel_interm(opts)
    elif opts.model == 'attn':
        from models.attention import AttnModel
        return AttnModel(opts)
    elif opts.model == 'hourglass':
        from models.stackedhourglassmodel import stacked_hourglass
        return stacked_hourglass(opts)
    else:
        raise Exception('Model type not found')


def evaluate(network, reader, batch_size):
    pred, gt = batch_predict(network, reader, batch_size)
    visible = gt > 0
    dist = np.sqrt(np.sum(((pred - gt) * visible)**2, 2))
    rmse = np.mean(np.mean(dist, 1))
    return rmse

# Returns prediction matrix and gt matrix processed in batches of batch size
def batch_predict(network, reader, batch_size):
    for i in range(reader.num_batches):
        # print ('------------iteration------------', i)
        batchX, batchy = reader.get_batch(i)
        # print ('got batchX batchY ...')
        pred = network.predict(batchX)
        # print ('got Pred', pred)
        if i == 0:
            all_pred = pred
            all_gt = batchy
        else:
            all_pred = np.vstack((all_pred, pred))
            all_gt = np.vstack((all_gt, batchy))
    N = reader.num_examples
    all_pred = all_pred[0:N, ::]
    all_gt = all_gt[0:N, ::]
    return all_pred, all_gt


# Example Usage : python classify_single.py --runtype [train | test]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM based pose model for single pose estimation')
    parser.add_argument('--runtype', help='Either [train | test] mode', type=str, required=True)
    parser.add_argument('--model', help='Model type [stackhourglass, base, lstm, attn]', type=str, required=True)
    parser.add_argument('--name', help='suffix for output directory', type=str)
    parser.add_argument('-resume', action='store_true', help='resume from most recent checkpoint')
    args = parser.parse_args()
    assert args.runtype in ['train', 'test']

    opts = GlobalOpts(args.name)
    opts.model = args.model
    opts.resume = args.resume

    main(opts, args.runtype)
