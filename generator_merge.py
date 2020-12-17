import numpy as np
import random
import glob
import os
import random
import h5py

from keras.utils import np_utils

# A generator that batches from a merged Numpy array
def generator(h5dir, batchsize):
    filenames = glob.glob(os.path.join(h5dir,"*.h5"))
    pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
    lb = np.empty((0), dtype=np.uint8)
    
    # Take all files in a directory and merge them into one dataset
    for fn in filenames:
        cf = h5py.File(fn)
        px = cf.get('pixelmaps')
        print("pixelmap shape ...", fn)
        print("pixelmap shape ...", cf.keys())
        print("pixelmap shape ...", px)

        pm = np.concatenate((pm, cf.get('pixelmaps')), axis = 0)
        lb = np.concatenate((lb, cf.get('labels')), axis = 0)
        print("pixelmap shape ...", pm.shape)

    nfeatures = pm.shape[0]
    while True:
        indices = np.sort(random.sample(range(nfeatures), batchsize))
        xbatch = pm[indices, 0, :, :]
        ybatch = pm[indices, 1, :, :]
        lbatch = np_utils.to_categorical(lb[indices], num_classes=5)
        yield {'xview' : xbatch, 'yview' : ybatch}, {'output' : lbatch}
