from os.path import dirname, join


class GlobalOpts:

    def __init__(self, name):
        # Directory Structure
        self.MODEL_NAME     = name
        self.SRC_DIR        = dirname(__file__)
        self.ROOT_DIR       = join(self.SRC_DIR, '..')
        self.LOG_DIR        = join(self.ROOT_DIR, 'checkpoints')
        self.DATA_DIR       = join(self.ROOT_DIR, 'data')
        self.PRETRAINED_DIR = join(self.DATA_DIR,'pretrained_models')
        self.IMG_DIR        = join(self.DATA_DIR, 'images')
        self.ANNOT_DIR      = join(self.DATA_DIR, 'annot')
        self.TRAIN_FILE     = join(self.ANNOT_DIR, 'train.h5')
        self.VALID_FILE     = join(self.ANNOT_DIR, 'valid.h5')
        self.TEST_FILE      = join(self.ANNOT_DIR, 'test.h5')
        self.ARCHLOG_DIR    = join(self.LOG_DIR, name)



        self.BATCH_SIZE = 25
        # self.BATCH_SIZE = 1
        #self.IN_HEIGHT = 480
        #self.IN_WIDTH = 640
        self.IN_HEIGHT = 256
        self.IN_WIDTH = 256
        self.IN_CHANNELS = 3
        self.NUM_JOINTS = 16
        #self.REG = 0.0001
        self.NUM_OUTPUTS = self.NUM_JOINTS * 2

        self.INIT_LR = 0.001
        self.MAX_ITERS = 2000
        # self.MAX_ITERS = 2000
        self.LR_DECAY_FACTOR = 0.5
        self.DECAY_STEPS = 1000

        # Training Globals
        self.VAL_CHECK = 2000
        self.SUMM_CHECK = 50
        # self.CHECKPOINT = 5000
        self.CHECKPOINT = 200

        # self.BATCH_SIZE = 1
        # #self.IN_HEIGHT = 480
        # #self.IN_WIDTH = 640
        # # self.IN_HEIGHT = 224
        # # self.IN_WIDTH = 224
        # self.IN_HEIGHT = 256
        # self.IN_WIDTH = 256
        # self.IN_CHANNELS = 3
        # self.NUM_JOINTS = 16
        # #self.REG = 0.0001
        # self.NUM_OUTPUTS = self.NUM_JOINTS * 2

        # self.INIT_LR = 0.001
        # # self.MAX_ITERS = 10000
        # self.MAX_ITERS = 500
        # self.LR_DECAY_FACTOR = 0.5
        # # self.DECAY_STEPS = 1000
        # self.DECAY_STEPS = 100

        # # Training Globals
        # self.VAL_CHECK = 500
        # # self.VAL_CHECK = 50
        # self.SUMM_CHECK = 50
        # # self.CHECKPOINT = 5000
        # self.CHECKPOINT = 50
