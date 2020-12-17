import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils

class DataGenerator(keras.utils.Sequence):

    def __init__(self, h5dir, batchsize):
        self.h5dir = h5dir
        self.batchsize = batchsize
        # List of files in the directory
        self.filelist = glob.glob(os.path.join(self.h5dir,"*.h5"))
        np.random.shuffle(self.filelist)
        # Array of sizes for each file
        self.filesizes = np.array([h5py.File(f).get('pixelmaps').shape[0] for f in self.filelist])
        # The number of batches per file
        self.batchesperfile = np.floor(self.filesizes/self.batchsize).astype('int64')
        self.batchintervals = [sum(self.batchesperfile[0:i])
                                    for i in range(len(self.batchesperfile)+1)][1:len(self.batchesperfile)+1]
        # Total number of features
        self.nfeatures = sum(self.filesizes)
        # The current file to use - iterated by checking the index
        self.currfileidx = 0

        print(self.filesizes)
        print(self.batchesperfile)
        print(self.nfeatures)
        print(self.currfileidx)
        print(self.batchintervals)

    # Should be the total number of batches - idx goes from 0 to len(DataGenerator)?
    def __len__(self):
        return sum(self.batchesperfile)

    def __getitem__(self, batchidx):
        # Get the current file to use
        currfileidx = next(x[0] for x in enumerate(self.batchintervals) if x[1] > batchidx)
        currfile = h5py.File(self.filelist[self.currfileidx])
        slicestart = (batchidx - sum(self.batchesperfile[0:self.currfileidx]))*self.batchsize
        sliceend = slicestart + self.batchsize

        # print("Getting batch no. %d in file %d at slice %d : %d" % (batchidx, currfileidx, slicestart, sliceend))

        if (sliceend / self.batchsize == self.batchesperfile[self.currfileidx]):
            self.currfileidx += 1

        pbatch = currfile.get('pixelmaps')[slicestart:sliceend]
        xbatch = pbatch[:, 0, :, :, :].astype('float32')
        ybatch = pbatch[:, 1, :, :, :].astype('float32')
        lbatch = np_utils.to_categorical(np.array(currfile.get('labels'))[slicestart:sliceend], num_classes=5)
        # print(pbatch.shape)
        # print(xbatch.shape)
        # print(ybatch.shape)
        # print(lbatch.shape)
        #
        return np.array(pbatch), np.array(lbatch)
        #return {'xview' : xbatch, 'yview' : ybatch}, {'output' : lbatch}

    def on_epoch_end(self):
        self.currfileidx = 0
        return
