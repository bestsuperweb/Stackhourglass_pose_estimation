from GlobalOpts import GlobalOpts
from os.path import join
from PIL import Image, ImageFile
import sys
import h5py
import scipy.io as sio
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class PoseReader:

    def __init__(self, runtype, opts):
        print ('Generating %s reader' % runtype)
        self.opts = opts
        self.runtype = runtype
        if runtype == 'train':
            data = h5py.File(opts.TRAIN_FILE)
        elif runtype == 'valid':
            data = h5py.File(opts.VALID_FILE)
        elif runtype == 'test':
            data = h5py.File(opts.TEST_FILE)
        else:
            raise Exception('Invalid runtype argument')
        # print ('data.pose', data['part'][:])
        # print data.keys()
        # self.pose = data['part'][:]
        # self.visible = data['visible'][:]
        # self.imgname = data['imgname'][:]

        self.pose = data['part'][0:22225]
        self.visible = data['visible'][0:22225]
        self.imgname = data['imgname'][0:22225]

        # Reorder pose joints with easier joints first
        # First pass - Head, upperneck, relbow,lelbow,rshoulder,lshoulder,thorax,pelvis
        order = [9,8,10,14,12,13,7,6]
        # Second pass - everthing else
        order += [0,1,2,3,4,5,11,15]

        self.pose = self.pose[:,order,:]


    @property
    def num_examples(self):
        return self.imgname.shape[0]

    @property
    def num_batches(self):
        return int(np.ceil(self.num_examples / self.opts.BATCH_SIZE))


    def sample(self, preprocess=True):
        inds = np.arange(self.num_examples)
        inds = np.random.choice(inds, self.opts.BATCH_SIZE)
        batchX, batchy = self.get_images(inds)
        return batchX.astype(np.float32), batchy.astype(np.float32)

    def get_batch(self, index, pad=True):
        start = index * self.opts.BATCH_SIZE
        inds = np.arange(start, start + self.opts.BATCH_SIZE)
        print ('inds', inds)
        batchX, batchy = self.get_images(inds)

        # pad if batch is smaller than batch size
        diff = self.opts.BATCH_SIZE - batchX.shape[0]
        if diff > 0:
            batchX = np.pad(batchX, ((0,diff),(0,0),(0,0)), mode='constant')
            batchy = np.pad(batchy, ((0,diff)), mode='constant')
        return batchX.astype(np.float32), batchy.astype(np.float32)

    def get_images(self, indices):
        padding = 20
        lstX = []
        lstY = []
        scale = []
        for ind in indices:
            imgn = self.imgname[ind]
            filepath = join(str.encode(self.opts.IMG_DIR), imgn)
            img = load_image(filepath)

            visible = np.array([ps for ps in self.pose[ind] if ps[0] > 0])
            ul = np.maximum(np.min(visible - padding, axis=0),0)
            br = np.minimum(np.max(visible + padding, axis=0), np.array([img.shape[1], img.shape[0]]))
            newpose = self.pose[ind] - ul

            ul = ul.astype(int)
            br = br.astype(int)
            img = img[ul[1]:br[1],ul[0]:br[0],:]
            
            scaleY = float(self.opts.IN_HEIGHT) / img.shape[0]
            scaleX = float(self.opts.IN_WIDTH) / img.shape[1]
            newpose = newpose * np.array([scaleX, scaleY])
            newpose[newpose < 0] = 0
            img = ImageScale(img, self.opts.IN_HEIGHT, self.opts.IN_WIDTH)
            lstX.append(img)
            lstY.append(newpose)
        return np.array(lstX), np.array(lstY)

    def get_image(self, path):
        padding = 20
        lstX = []
        scale = []
        img = load_image(path)
        scaleY = float(self.opts.IN_HEIGHT) / img.shape[0]
        scaleX = float(self.opts.IN_WIDTH) / img.shape[1]
        img = ImageScale(img, self.opts.IN_HEIGHT, self.opts.IN_WIDTH)
        lstX.append(img)
        return np.array(lstX)



def ImageScale(img, height, width):
    tmp = Image.fromarray(img).resize((height,width))
    return np.array(tmp)


def load_image(filepath):
    img = Image.open(filepath)
    return np.array(img)


if __name__ == '__main__':
    opts = GlobalOpts('testModel')
    reader = PoseReader('train', opts)
    print ('pose : %s' % str(reader.pose.shape))
    print ('visible : %s' % str(reader.visible.shape))
    print ('imgname : %s' % str(reader.imgname.shape))
    print ('sample()[0] : %s' % str(reader.sample()[0].shape))
    print ('sample()[1] : %s' % str(reader.sample()[1].shape))
    print ('get_batch()[0] : %s' % str(reader.get_batch(1)[0].shape))
    print ('get_batch()[1] : %s' % str(reader.get_batch(1)[1].shape))
    print ('get_images() : %s' % str(reader.get_images([1,2,3,4,5])[0].shape))
    batchY = reader.sample()[1]
    print ('Zero gt values : %d/%d' % (np.sum(batchY==0), np.prod(batchY.shape)))
