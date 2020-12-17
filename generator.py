import numpy as np
import random
import glob
import os
import random
import h5py

from keras.utils import np_utils

def generator(h5dir, batchsize):
    filenames = glob.glob(os.path.join(h5dir,"*.h5"))

    while True:
        # Select a file to batch from
        rfile = np.random.choice(filenames)
        currfile = h5py.File(rfile)
        nfeatures = currfile.get('pixelmaps').shape[0]
        # Randomly batch from the H5
        try:
            indices = np.sort(random.sample(range(nfeatures), batchsize))
            pass
            xbatch = currfile.get('pixelmaps')[indices, 0, :, :]
            ybatch = currfile.get('pixelmaps')[indices, 1, :, :]
            lbatch = np_utils.to_categorical(np.array(currfile.get('labels'))[indices], num_classes=5)
            pass

        except:
                print(".................",nfeatures)
                print(".................",batchsize)
                print(".....name......",rfile)
        yield {'xview' : xbatch, 'yview' : ybatch}, {'output' : lbatch}
